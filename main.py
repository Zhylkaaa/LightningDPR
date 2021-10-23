import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Tuple, Optional, Dict
import json

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributed
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from torch import Tensor
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
)
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from callbacks import get_default_callbacks
from data_module import DPRDatasetModule
from dpr_model import DPRContextEncoder, DPRQuestionEncoder
from utils import (
    init_dpr_component_from_pretrained_model
)
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, questions: Tensor, contexts: Tensor,
                positive_ids: Tensor, is_valid: Tensor) -> Any:
        def gather_tensor(tensor: Tensor):
            gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_tensor, tensor)
            return gathered_tensor

        ctx.question_batch = questions.shape[0]
        ctx.context_batch = contexts.shape[0]
        ctx.batch_is_valid = is_valid

        gathered_questions = gather_tensor(questions)
        gathered_questions = torch.cat(gathered_questions, dim=0)

        gathered_contexts = gather_tensor(contexts)
        gathered_contexts = torch.cat(gathered_contexts, dim=0)

        gathered_positive_ids = gather_tensor(positive_ids)

        gathered_is_valid = gather_tensor(is_valid)

        shifted_positive_ids = []
        offset = 0
        offsets = {}
        for rank, (positive_ids, is_valid) in enumerate(zip(gathered_positive_ids, gathered_is_valid)):
            shifted_positive_ids.extend([int(sum(is_valid[:idx]) + offset) for idx in positive_ids])
            offsets[rank] = offset
            offset += int(sum(is_valid))
        offsets[torch.distributed.get_world_size()] = offset

        ctx.offsets = offsets
        gathered_is_valid = torch.cat(gathered_is_valid, dim=0)

        return gathered_questions, gathered_contexts[gathered_is_valid.type(torch.bool)], \
            torch.tensor(shifted_positive_ids, dtype=torch.long, device=gathered_contexts.device)

    @staticmethod
    def backward(ctx: Any, questions_grad, contexts_grad, *args) -> Any:
        questions_input_grad = questions_grad.clone()
        context_input_grad = torch.zeros((ctx.context_batch, *contexts_grad.shape[1:]),
                                         dtype=contexts_grad, device=contexts_grad.device)

        rank = torch.distributed.get_rank()
        questions_from = rank * ctx.question_batch
        questions_to = (rank + 1) * ctx.question_batch

        contexts_from = ctx.offsets[rank]
        contexts_to = ctx.offsets[rank + 1]

        contexts_grad_cp = contexts_grad.clone()
        context_input_grad[ctx.batch_is_valid.type(torch.bool)] = contexts_grad_cp[contexts_from:contexts_to]

        return questions_input_grad[questions_from:questions_to], context_input_grad, None, None


# For further optimization
class AggregateModel(torch.nn.Module):
    def __init__(self, q_encoder, ctx_encoder):
        super().__init__()
        self.q_encoder = q_encoder
        self.ctx_encoder = ctx_encoder

    def forward(self, *inputs):
        return self.q_encoder(*inputs[0])[0], self.ctx_encoder(*inputs[1])[0]


class DPRModel(pl.LightningModule):
    def __init__(self,
                 question_model_name_or_path: str,
                 question_projection_dim: int,
                 context_model_name_or_path: str,
                 context_projection_dim: int,
                 hparams: Namespace,
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.avg_train_num_correct = -1
        self.avg_train_loss = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.question_model_name_or_path = question_model_name_or_path
        self.context_model_name_or_path = context_model_name_or_path
        self.question_projection_dim = question_projection_dim
        self.context_projection_dim = context_projection_dim

        if not self.hparams.configure_sharded_model:
            self.q_encoder, self.q_tokenizer = init_dpr_component_from_pretrained_model(
                model_name_or_path=question_model_name_or_path,
                component_class=DPRQuestionEncoder,
                projection_dim=question_projection_dim,
                is_dpr_checkpoint=self.hparams.is_dpr_checkpoint
            )

            self.ctx_encoder, self.ctx_tokenizer = init_dpr_component_from_pretrained_model(
                model_name_or_path=context_model_name_or_path,
                component_class=DPRContextEncoder,
                projection_dim=context_projection_dim,
                is_dpr_checkpoint=self.hparams.is_dpr_checkpoint
            )

        if self.hparams.cpu_checkpointing or self.hparams.partition_activations or self.hparams.use_checkpointing:
            self.q_encoder.gradient_checkpointing_enable(use_deepspeed=True)
            self.ctx_encoder.gradient_checkpointing_enable(use_deepspeed=True)

    def configure_sharded_model(self) -> None:
        if self.hparams.configure_sharded_model:
            self.q_encoder, self.q_tokenizer = init_dpr_component_from_pretrained_model(
                model_name_or_path=self.question_model_name_or_path,
                component_class=DPRQuestionEncoder,
                projection_dim=self.question_projection_dim,
                is_dpr_checkpoint=self.hparams.is_dpr_checkpoint
            )
            self.q_encoder = auto_wrap(self.q_encoder)

            self.ctx_encoder, self.ctx_tokenizer = init_dpr_component_from_pretrained_model(
                model_name_or_path=self.context_model_name_or_path,
                component_class=DPRContextEncoder,
                projection_dim=self.context_projection_dim,
                is_dpr_checkpoint=self.hparams.is_dpr_checkpoint
            )

            self.ctx_encoder = auto_wrap(self.ctx_encoder)

    def forward(self, question_input, ctx_input) -> Tuple[Tensor, Tensor]:
        local_question_vectors, local_ctx_vectors = \
            self.q_encoder(**question_input)[0], self.ctx_encoder(**ctx_input)[0]

        return local_question_vectors.contiguous(), local_ctx_vectors.contiguous()

    def _calculate_loss(self,
                        q_vectors,
                        ctx_vectors,
                        positive_ids):
        scores = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            positive_ids.to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == positive_ids.to(max_idxs.device)
        ).sum()
        return loss, correct_predictions_count

    def shared_step(self, batch, sync_grads=False):
        local_question_vectors, local_ctx_vectors, positive_context_ids, is_valid = batch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            (global_question_vectors,
             global_ctx_vectors,
             global_positive_context_ids,
             is_valid) = self.all_gather((
                local_question_vectors,
                local_ctx_vectors,
                positive_context_ids,
                is_valid
             ), sync_grads=sync_grads)

            offset = 0
            shifted_positive_ids = []
            for idx, (valid, positive_context_ids) in enumerate(zip(is_valid, global_positive_context_ids)):
                shifted_positive_ids.extend([int(sum(valid[:idx]) + offset) for idx in positive_context_ids])
                offset += int(sum(valid))
            global_question_vectors = global_question_vectors.view(-1, global_question_vectors.shape[-1])
            global_ctx_vectors = global_ctx_vectors.view(-1, global_ctx_vectors.shape[-1])
            is_valid = is_valid.view(-1)
            global_ctx_vectors = global_ctx_vectors[is_valid.type(torch.bool)]
            positive_context_ids = torch.tensor(shifted_positive_ids, dtype=torch.long)
            # (global_question_vectors,
            #  global_ctx_vectors,
            #  positive_context_ids) = \
            #     SyncFunction.apply(local_question_vectors, local_ctx_vectors, positive_context_ids, is_valid)
        else:
            positive_context_ids = torch.tensor([int(sum(is_valid[:idx])) for idx in positive_context_ids],
                                                dtype=torch.long, device=local_ctx_vectors.device)
            global_question_vectors, global_ctx_vectors, positive_context_ids = \
                local_question_vectors, local_ctx_vectors[is_valid.type(torch.bool)], positive_context_ids

        loss, num_correct = self._calculate_loss(global_question_vectors, global_ctx_vectors, positive_context_ids)
        return loss, num_correct

    def training_step(self, batch, batch_idx):
        question_input, ctx_input, positive_context_ids, is_valid = batch
        # question_input = batch[:2]
        # ctx_input = batch[2:4]
        # positive_context_ids, is_valid = batch[4:]

        local_question_vetors, local_ctx_vectors = self(question_input, ctx_input)
        return local_question_vetors, local_ctx_vectors, positive_context_ids, is_valid

    def training_step_end(self, train_step_outputs):
        loss, num_correct = self.shared_step(train_step_outputs, sync_grads=True)

        if self.avg_train_num_correct == -1:
            self.avg_train_num_correct = num_correct
            self.avg_train_loss = loss
        else:
            self.avg_train_num_correct = 0.9 * self.avg_train_num_correct + 0.1 * num_correct
            self.avg_train_loss = 0.9 * self.avg_train_loss + 0.1 * loss

        if self.global_step % self.hparams.log_every_n_steps:
            self.log_dict({
                'train_avg_loss': self.avg_train_loss,
                'train_avg_num_correct': self.avg_train_num_correct,
                'global_step': self.global_step
            })
        return loss

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        question_input, ctx_input, positive_context_ids, is_valid = batch

        # question_input = batch[:2]
        # ctx_input = batch[2:4]
        # positive_context_ids, is_valid = batch[4:]

        local_question_vetors, local_ctx_vectors = self(question_input, ctx_input)
        return local_question_vetors, local_ctx_vectors, positive_context_ids, is_valid

    def validation_step_end(self, validation_step_outputs) -> Tuple[Tensor, Tensor]:
        loss, num_correct = self.shared_step(validation_step_outputs, sync_grads=False)
        return loss, num_correct

    def _aggregate_validation_metrics(self, outputs: Tuple[Tensor, Tensor]):
        losses, num_correct = list(zip(*outputs))
        losses = torch.tensor(losses)
        num_correct = torch.tensor(num_correct)

        loss = torch.mean(losses).detach().cpu()
        num_correct = torch.sum(num_correct).detach().cpu()
        return loss, num_correct

    def validation_epoch_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        loss, num_correct = self._aggregate_validation_metrics(outputs)

        self.log('val_loss', loss)
        self.log('val_num_correct', num_correct)
        # self.log_dict({
        #     'val_loss': loss,
        #     'val_num_correct': num_correct
        # }, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        question_input, ctx_input, positive_context_ids, is_valid = batch

        # question_input = batch[:2]
        # ctx_input = batch[2:4]
        # positive_context_ids, is_valid = batch[4:]

        local_question_vetors, local_ctx_vectors = self(question_input, ctx_input)
        return local_question_vetors, local_ctx_vectors, positive_context_ids, is_valid

    def test_step_end(self, test_step_outputs) -> Tuple[Tensor, Tensor]:
        return self.shared_step(test_step_outputs, sync_grads=False)

    def test_epoch_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        loss, num_correct = self._aggregate_validation_metrics(outputs)
        self.log('test_loss', loss)
        self.log('test_num_correct', num_correct)
        # self.log_dict({
        #     'test_loss': loss,
        #     'test_num_correct': num_correct
        # }, sync_dist=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != 'fit':
            return
        train_loader = self.train_dataloader()
        trainer = self.trainer
        num_devices = max(1, trainer.gpus)
        effective_batch_size = self.hparams.train_batch_size * num_devices * trainer.accumulate_grad_batches

        self.total_steps = int((len(train_loader.dataset) / effective_batch_size) * trainer.max_epochs)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.use_cpu_adam:
            optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        elif self.hparams.use_fused_adam:
            optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            },
        }

    # @pl.utilities.rank_zero_only
    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     base_save_path = self.output_dir.joinpath(f'checkpoint-{self.current_epoch}-{self.global_step}')
    #     self.q_encoder.save_pretrained(base_save_path.joinpath('question_encoder'))
    #     self.q_tokenizer.save_pretrained(base_save_path.joinpath('question_encoder'))
    #
    #     self.ctx_encoder.save_pretrained(base_save_path.joinpath('context_encoder'))
    #     self.ctx_tokenizer.save_pretrained(base_save_path.joinpath('context_encoder'))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DPRModel Params')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam epsilon')
        parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
        parser.add_argument('--num_train_epochs', dest="max_epochs", default=1, type=int)
        parser.add_argument('--learning_rate', default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument('--output_dir', default='dpr_model', type=str)


if __name__ == '__main__':
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)
    DPRModel.add_model_specific_args(parser)
    DPRDatasetModule.add_argparse_args(parser)
    parser.add_argument('--model_name_or_path', default=None,
                        help='Model name or path used to initialize both question and context encoder. '
                             'For more control use <question/context>_model_name_or_path')
    parser.add_argument('--question_model_name_or_path', default=None, help='Question encoder pretrained model')
    parser.add_argument('--context_model_name_or_path', default=None, help='Context encoder pretrained model')
    parser.add_argument('--question_projection_dim', default=0, type=int, help='Question encoder projection dim')
    parser.add_argument('--context_projection_dim', default=0, type=int, help='Context encoder projection dim')
    parser.add_argument('--seed', default=1, type=int, help='Seed for model and dataloaders')
    parser.add_argument('--fp16', action='store_true', help='Turn on AMP (adaptive mixed precision) training')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run prediction on test set')
    parser.add_argument('--tb_log_dir', default='dpr_experiment', help='Tensorboard log dir')
    parser.add_argument('--wandb_project', default=None, help='Wandb project to log to')
    parser.add_argument('--monitor_metric', default='val_num_correct')
    parser.add_argument('--use_cpu_adam', action='store_true')
    parser.add_argument('--use_fused_adam', action='store_true')
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--cpu_checkpointing', action='store_true')
    parser.add_argument('--offload_optimizer', action='store_true')
    parser.add_argument('--use_checkpointing', action='store_true')
    parser.add_argument('--offload_parameters', action='store_true')
    parser.add_argument('--partition_activations', action='store_true')
    parser.add_argument('--configure_sharded_model', action='store_true')
    parser.add_argument('--is_dpr_checkpoint', action='store_true')
    parser.add_argument('--no_save_full_weights', action='store_true')
    parser.add_argument('--ds_config_path', default=None, type=str)
    args: Namespace = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    if args.model_name_or_path is not None and args.question_model_name_or_path is not None \
            and args.context_model_name_or_path is not None:
        raise ValueError('Can\'t specify both model_name_or_path with '
                         'question_model_name_or_path and context_model_name_or_path')
    elif args.model_name_or_path is not None:
        question_model_name_or_path = args.model_name_or_path
        context_model_name_or_path = args.model_name_or_path
    elif args.question_model_name_or_path is not None and args.context_model_name_or_path:
        question_model_name_or_path = args.question_model_name_or_path
        context_model_name_or_path = args.context_model_name_or_path
    else:
        raise ValueError('Please specify model_name_or_path or '
                         'question_model_name_or_path with context_model_name_or_path')

    callbacks = get_default_callbacks(args)

    wandb_logger = None
    if args.wandb_project is not None:
        wandb_logger = WandbLogger(project=args.wandb_project)

    tensorboard_logger = TensorBoardLogger(save_dir=args.tb_log_dir)

    if args.plugins is not None and args.use_deepspeed:
        raise ValueError('--use_deepspeed is used to define custom behaviour of DeepSpeedPlugin '
                         'and can\'t be used with --plugins')

    if args.use_deepspeed:
        if args.ds_config_path:
            dpr_datamodule = DPRDatasetModule(q_tokenizer_path=question_model_name_or_path,
                                              ctx_tokenizer_path=context_model_name_or_path, args=args)
            dpr_datamodule.setup()
            with open(args.ds_config_path) as f:
                config = json.load(f)
            num_devices = max(1, args.gpus)
            effective_batch_size = num_devices * args.accumulate_grad_batches

            total_steps = int((len(dpr_datamodule.train_dataloader()) / effective_batch_size) * args.max_epochs)
            config['scheduler']['params']['warmup_num_steps'] = args.warmup_steps
            config['scheduler']['params']['total_num_steps'] = total_steps
            config['train_micro_batch_size_per_gpu'] = args.train_batch_size
            print(total_steps)
            print(config)
        else:
            config = args.ds_config_path

        if pl.__version__ == '1.4.9':
            plugin = DeepSpeedPlugin(
                stage=3,
                offload_optimizer=args.offload_optimizer,
                offload_parameters=args.offload_parameters,
                cpu_checkpointing=args.cpu_checkpointing,
                partition_activations=args.partition_activations,
                save_full_weights=(not args.no_save_full_weights),
                config=config
            )
        else:
            plugin = DeepSpeedPlugin(
                stage=3,
                offload_optimizer=args.offload_optimizer,
                offload_parameters=args.offload_parameters,
                cpu_checkpointing=args.cpu_checkpointing,
                partition_activations=args.partition_activations,
                config=config
            )
    else:
        plugin = None

    additional_args = {
        'precision': 16 if args.fp16 else 32,
        'amp_backend': 'native',
        'replace_sampler_ddp': False,
        'callbacks': callbacks,
        'logger': [tensorboard_logger, wandb_logger] if wandb_logger else tensorboard_logger,
        'plugins': plugin,
    }

    model = DPRModel(
        question_model_name_or_path=question_model_name_or_path,
        question_projection_dim=args.question_projection_dim,
        context_model_name_or_path=context_model_name_or_path,
        context_projection_dim=args.context_projection_dim,
        hparams=args
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        **additional_args,
    )

    if wandb_logger:
        pass #wandb_logger.watch(model)

    dpr_datamodule = DPRDatasetModule(q_tokenizer_path=question_model_name_or_path,
                                      ctx_tokenizer_path=context_model_name_or_path, args=args)
    if args.do_train:
        trainer.fit(model, datamodule=dpr_datamodule)

    if args.do_predict:
        trainer.test(model, datamodule=dpr_datamodule)
