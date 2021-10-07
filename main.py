from argparse import ArgumentParser, Namespace
from typing import Any, Tuple, List, Optional, Dict
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dpr_model import DPRContextEncoder, DPRQuestionEncoder
from dpr_config import DPRConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AdamW,
)
from utils import (
    init_dpr_component_from_pretrained_model
)
from data_module import DPRDatasetModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class DPRModel(pl.LightningModule):
    def __init__(self,
                 q_encoder: PreTrainedModel,
                 q_tokenizer: PreTrainedTokenizer,
                 ctx_encoder: PreTrainedModel,
                 ctx_tokenizer: PreTrainedTokenizer,
                 hparams: Namespace,
                 ):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.q_encoder = q_encoder
        self.q_tokenizer = q_tokenizer
        self.ctx_encoder = ctx_encoder
        self.ctx_tokenizer = ctx_tokenizer
        self.avg_train_num_correct = -1
        self.avg_train_loss = 0
        self.step = 0
        self.output_dir = Path(self.hparams.output_dir)

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        pass

    def _calculate_loss(self,
                        q_vectors,
                        ctx_vectors,
                        positive_ids):
        scores = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_ids).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_ids).to(max_idxs.device)
        ).sum()
        return loss, correct_predictions_count

    def _produce_question_ctx_vectors(self, question_input, ctx_input, positive_context_ids, is_valid):
        local_question_vectors = self.q_encoder(**question_input)[0]
        local_ctx_vectors = self.ctx_encoder(**ctx_input)[0]

        (global_question_vectors,
         global_ctx_vectors,
         global_positive_context_ids,
         is_valid) = self.all_gather((
            local_question_vectors,
            local_ctx_vectors,
            positive_context_ids,
            is_valid
        ))

        offset = 0
        for idx, valid in enumerate(is_valid):
            positive_context_ids[idx] += offset
            offset += torch.sum(valid)

        global_question_vectors = global_question_vectors.view(-1, global_question_vectors.shape[-1])
        global_ctx_vectors = global_ctx_vectors.view(-1, global_ctx_vectors.shape[-1])
        is_valid = is_valid.view(-1)
        global_ctx_vectors = global_ctx_vectors[is_valid]
        positive_context_ids = positive_context_ids.view(-1)

        return global_question_vectors, global_ctx_vectors, positive_context_ids

    def training_step(self, question_input, ctx_input, positive_context_ids, is_valid):
        global_question_vectors, global_ctx_vectors, positive_context_ids = \
            self._produce_question_ctx_vectors(question_input, ctx_input, positive_context_ids, is_valid)

        loss, num_correct = self._calculate_loss(global_question_vectors, global_ctx_vectors, positive_context_ids)
        if self.avg_train_num_correct == -1:
            self.avg_train_num_correct = num_correct
            self.avg_train_loss = loss
        else:
            self.avg_train_num_correct = 0.9 * self.avg_train_num_correct + 0.1 * num_correct
            self.avg_train_loss = 0.9 * self.avg_train_loss + 0.1 * loss

        self.step += 1
        if self.step % self.hparams.log_steps:
            self.log_dict({
                'avg_train_loss': self.avg_train_loss,
                'train_num_correct_avg': self.avg_train_num_correct,
                'global_step': self.step
            })
        return loss

    def validation_step(self, question_input, ctx_input, positive_context_ids, is_valid) -> Tuple[Tensor, Tensor]:
        global_question_vectors, global_ctx_vectors, positive_context_ids = \
            self._produce_question_ctx_vectors(question_input, ctx_input, positive_context_ids, is_valid)

        loss, num_correct = self._calculate_loss(global_question_vectors, global_ctx_vectors, positive_context_ids)
        return loss, num_correct

    def validation_epoch_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        losses, num_correct = list(zip(*outputs))
        loss = torch.mean(losses).detach().cpu()
        num_correct = torch.sum(num_correct).detach().cpu()

        self.log_dict({
            'val_loss': loss,
            'val_num_correct': num_correct
        })

    def test_step(self, question_input, ctx_input, positive_context_ids, is_valid) -> Tuple[Tensor, Tensor]:
        global_question_vectors, global_ctx_vectors, positive_context_ids = \
            self._produce_question_ctx_vectors(question_input, ctx_input, positive_context_ids, is_valid)

        loss, num_correct = self._calculate_loss(global_question_vectors, global_ctx_vectors, positive_context_ids)
        return loss, num_correct

    def test_epoch_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        losses, num_correct = list(zip(*outputs))
        loss = torch.mean(losses).detach().cpu()
        num_correct = torch.sum(num_correct).detach().cpu()

        self.log_dict({
            'test_loss': loss,
            'test_num_correct': num_correct
        })

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != 'fit':
            return
        train_loader = self.train_dataloader()
        trainer = self.trainer
        num_devices = max(1, trainer.gpus)
        effective_batch_size = self.hparams.train_batch_size * num_devices * trainer.accumulate_grad_batches

        self.total_steps = (len(train_loader.dataset) / effective_batch_size) * trainer.max_epochs

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

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        base_save_path = self.output_dir.joinpath("checkpoint{}".format(self.step_count))
        self.q_encoder.save_pretrained(base_save_path.joinpath('question_encoder'))
        self.q_tokenizer.save_pretrained(base_save_path.joinpath('question_encoder'))

        self.ctx_encoder.save_pretrained(base_save_path.joinpath('context_encoder'))
        self.ctx_tokenizer.save_pretrained(base_save_path.joinpath('context_encoder'))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DPRModel Params')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam epsilon')
        parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
        parser.add_argument('--num_train_epochs', dest="max_epochs", default=1, type=int)
        parser.add_argument('--learning_rate', default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument('--log_steps', default=50, type=int, help='number of steps before ')
        parser.add_argument('--output_dir', default='dpr_model', type=str)

if __name__ == '__main__':
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)
    DPRModel.add_model_specific_args(parser)
    DPRDatasetModule.add_argparse_args(parser)
    parser.add_argument('--model_name_or_path', default=None, help='Model name or path used to initialize both question and context encoder. '
                                                                   'For more control use <question/context>_model_name_or_path')
    parser.add_argument('--question_model_name_or_path', default=None, help='Question encoder pretrained model')
    parser.add_argument('--context_model_name_or_path', default=None, help='Context encoder pretrained model')
    parser.add_argument('--question_projection_dim', default=0, help='Question encoder projection dim')
    parser.add_argument('--context_projection_dim', default=0, help='Context encoder projection dim')
    parser.add_argument('--seed', default=1, type=int, help='Seed for model and dataloaders')
    parser.add_argument('--nofp16', action='store_true', help='Turn off AMP')
    args = parser.parse_args()

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

    question_model, question_tokenizer = init_dpr_component_from_pretrained_model(
        model_name_or_path=question_model_name_or_path,
        component_class=DPRQuestionEncoder,
        projection_dim=args.question_projection_dim
    )

    context_model, context_tokenizer = init_dpr_component_from_pretrained_model(
        model_name_or_path=context_model_name_or_path,
        component_class=DPRContextEncoder,
        projection_dim=args.context_projection_dim
    )

    # TODO
