import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Tuple, Optional, Dict
import json

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.distributed
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin, DDPPlugin
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
)
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from callbacks import get_default_callbacks
from reader_datamodule import DPRReaderDatasetModule
from utils import (
    init_dpr_reader
)


class ReaderModel(pl.LightningModule):
    def __init__(self,
                 encoder_name_or_path: str,
                 hparams: Namespace,
                 ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.avg_train_num_correct = -1
        self.avg_train_loss = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.reader_name_or_path = encoder_name_or_path

        self.reader, self.tokenizer = init_dpr_reader(
            model_name_or_path=encoder_name_or_path,
            is_reader_checkpoint=self.hparams.is_dpr_checkpoint
        )

        if self.hparams.cpu_checkpointing or self.hparams.partition_activations or self.hparams.use_checkpointing:
            self.reader.gradient_checkpointing_enable(use_deepspeed=True)

        self.avg_train_acc = -1
        self.avg_train_loss = -1

    def forward(self, inputs) -> Tensor:
        n, m, _ = inputs['input_ids'].shape
        inputs = {k: v.view(n*m, -1) for k, v in inputs.items()}
        return self.reader(**inputs).view(n, m)

    def _calculate_loss(self, logits):
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        if self.hparams.weight_examples:
            loss = F.cross_entropy(logits, labels, reduction='none')
            weights = (1 - torch.softmax(logits, dim=1)[:, 0]) ** 2
            loss = torch.sum(loss * weights) / torch.clamp(torch.sum(weights), min=1e-6)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def shared_step(self, batch) -> Tuple[Tensor, Tensor]:
        inputs, valid = batch
        logits = self(inputs)
        logits[~valid] = -float('inf')
        loss = self._calculate_loss(logits)
        predictions = torch.argmax(logits, dim=1).view(-1)
        return loss, predictions

    def training_step(self, batch, batch_idx):
        loss, predictions = self.shared_step(batch)
        acc = torch.mean((predictions == 0).float())
        if self.avg_train_loss == -1:
            self.avg_train_loss = loss
            self.avg_train_acc = acc
        else:
            self.avg_train_loss = 0.9 * self.avg_train_loss + 0.1 * loss
            self.avg_train_acc = 0.9 * self.avg_train_acc + 0.1 * acc

        if self.global_step % self.hparams.log_every_n_steps:
            self.log_dict({
                'train_avg_loss': self.avg_train_loss,
                'train_avg_acc': self.avg_train_acc,
                'global_step': self.global_step
            })
        return loss

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Tensor]:
        return self.shared_step(batch)

    def _aggregate_validation_metrics(self, outputs: Tuple[Tensor, Tensor]):
        losses, predictions = list(zip(*outputs))
        losses = torch.tensor(losses)
        loss = torch.mean(losses).detach().cpu()

        predictions = torch.cat(predictions, dim=0)
        acc = torch.mean((predictions == 0).float()).detach().cpu()
        return loss, acc

    def validation_epoch_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        loss, acc = self._aggregate_validation_metrics(outputs)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx) -> Tuple[Tensor, Tensor]:
        return self.shared_step(batch)

    def test_epoch_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        loss, acc = self._aggregate_validation_metrics(outputs)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DPRModel Params')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam epsilon')
        parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
        parser.add_argument('--num_train_epochs', dest="max_epochs", default=1, type=int)
        parser.add_argument('--learning_rate', default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument('--output_dir', default='dpr_model', type=str)
        parser.add_argument('--weight_examples', action='store_true')


if __name__ == '__main__':
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)
    ReaderModel.add_model_specific_args(parser)
    DPRReaderDatasetModule.add_argparse_args(parser)
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

    if args.model_name_or_path is None:
        raise ValueError('Specify model')

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
            reader_datamodule = DPRReaderDatasetModule(tokenizer=args.model_name_or_path, args=args)
            reader_datamodule.setup()
            with open(args.ds_config_path) as f:
                config = json.load(f)
            num_devices = max(1, args.gpus)
            effective_batch_size = num_devices * args.accumulate_grad_batches

            total_steps = int((len(reader_datamodule.train_dataloader()) / effective_batch_size) * args.max_epochs)
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

    if args.accelerator == 'ddp' and plugin is None:
        plugin = DDPPlugin(find_unused_parameters=False)
    else:
        plugin = args.accelerator

    additional_args = {
        'precision': 16 if args.fp16 else 32,
        'amp_backend': 'native',
        'replace_sampler_ddp': True,
        'callbacks': callbacks,
        'logger': [tensorboard_logger, wandb_logger] if wandb_logger else tensorboard_logger,
        'plugins': plugin,
        #'strategy': strategy
    }

    model = ReaderModel(
        encoder_name_or_path=args.model_name_or_path,
        hparams=args
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        **additional_args,
    )

    if wandb_logger:
        pass #wandb_logger.watch(model)

    reader_datamodule = DPRReaderDatasetModule(tokenizer=args.model_name_or_path, args=args)

    if args.do_train:
        trainer.fit(model, datamodule=reader_datamodule)

    if args.do_predict:
        trainer.test(model, datamodule=reader_datamodule)
