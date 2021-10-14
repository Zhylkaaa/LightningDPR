import logging
import random
from argparse import ArgumentParser, Namespace
import json
from typing import Union, List, Tuple, Dict, Optional
import math

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.distributed import T_co
import torch.distributed as dist

from transformers import PreTrainedTokenizer, AutoTokenizer
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class DPRDistributedSamplerWithValidation(Sampler[T_co]):
    """Sampler that restricts data loading to a subset of the dataset with additional validation for DPR model.

    Validations are aimed to help with model generalization and eliminate possibility of
    passing same positive and in-batch negative contexts in one batch.

    Code mainly copied from `torch.utils.data.distributed.DistributedSampler`
    for code references please refer to `DistributedSampler` documentation

    Added args:
    disjoint_window_size : int
        defines width of examples window with non-intersecting  (essentially `effective_batch_size`)
        Note: this means that sets 0..disjoint_window_size and disjoint_window_size..2*disjoint_window_size, but
              1..disjoint_window_size+1 may not be.
        default: 1 (means no validation)
    resampling_count : int
        defines number of random indexes permutations checked while trying to satisfy data constrains.
        Return first permutation that satisfies the constraint. If all checks fail, return last checked permutation
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 disjoint_window_size: int = 1, resampling_count: int = 100) -> None:
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
                logger.info('Detected single node usage (probably dp mode) using num_replicas = 1')
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
                logger.info('Detected single node usage (probably dp mode) using rank = 1')
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.disjoint_window_size = disjoint_window_size
        self.resampling_count = resampling_count

    def _is_valid(self, indexes):
        def concat_inputs(positives):
            return [((p['title'] + ' ') if p['title'] else '') + p['text'] for p in positives]

        current_contexts = set()

        for idx, i in enumerate(indexes):
            if (idx + 1) % self.disjoint_window_size == 0:
                current_contexts.clear()

            positives = concat_inputs(self.dataset[i]['positive_ctxs'])
            if len(current_contexts.intersection(set(positives))) > 0:
                return False
            else:
                current_contexts.update(set(positives))
        return True

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]

            if self.disjoint_window_size > 1:
                for _ in range(self.resampling_count + 1):
                    if self._is_valid(indices):
                        break
                    indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

            g = torch.Generator()
            g.manual_seed(self.seed)
            if self.disjoint_window_size > 1:
                for _ in range(self.resampling_count + 1):
                    if self._is_valid(indices):
                        break
                    indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]



        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        logger.debug('set_epoch is called')
        self.epoch = epoch


class DPRDatasetModule(pl.LightningDataModule):
    def __init__(
            self,
            q_tokenizer_path: PreTrainedTokenizer,
            ctx_tokenizer_path: PreTrainedTokenizer,
            args: Namespace,
    ):
        super(DPRDatasetModule, self).__init__()
        self.args = args
        self.q_tokenizer_path = q_tokenizer_path
        self.ctx_tokenizer_path = ctx_tokenizer_path
        self.dataset: Dict[str, Optional[List]] = {
            'train': None,
            'dev': None,
            'test': None
        }

    def setup(self, stage: Optional[str] = None) -> None:
        for data_split, data_file in [('train', self.args.train_data),
                                      ('dev', self.args.dev_data),
                                      ('test', self.args.test_data)]:
            if data_file is not None:
                with open(data_file) as f:
                    data = json.load(f)
                filtered_data = [
                    d for d in data
                    if len(d['positive_ctxs']) > 0
                ]
                self.dataset[data_split] = filtered_data

        self.q_tokenizer = AutoTokenizer.from_pretrained(self.q_tokenizer_path)
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(self.ctx_tokenizer_path)

        self.sep_token = self.ctx_tokenizer.sep_token \
            if self.ctx_tokenizer.sep_token is not None else self.ctx_tokenizer.eos_token
        self.pad_token = self.ctx_tokenizer.pad_token

    def prepare_data(self) -> None:
        for data_split, data_file in [('train', self.args.train_data),
                                      ('dev', self.args.dev_data),
                                      ('test', self.args.test_data)]:
            if data_file is not None:
                with open(data_file) as f:
                    data = json.load(f)
                filtered_data = [
                    d for d in data
                    if len(d['positive_ctxs']) > 0
                ]
                logger.info(f'{data_split.upper()} data size after filtering: {len(filtered_data)}')

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        if self.dataset['train'] is None:
            return None

        num_devices = max(1, self.args.gpus)
        effective_batch_size = self.args.train_batch_size * num_devices * self.args.accumulate_grad_batches

        sampler = DPRDistributedSamplerWithValidation(dataset=self.dataset['train'], seed=self.args.seed,
                                                      disjoint_window_size=effective_batch_size, resampling_count=100,
                                                      shuffle=True)

        return DataLoader(
            self.dataset['train'],
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=lambda x: self.collator(x, sample=True)
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.dataset['dev'] is None:
            return None

        num_devices = max(1, self.args.gpus)
        effective_batch_size = self.args.eval_batch_size * num_devices
        sampler = DPRDistributedSamplerWithValidation(dataset=self.dataset['dev'], seed=self.args.seed,
                                                      disjoint_window_size=effective_batch_size,
                                                      shuffle=False)

        return DataLoader(
            self.dataset['dev'],
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=lambda x: self.collator(x, sample=False)
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.dataset['test'] is None:
            return None

        num_devices = max(1, self.args.gpus)
        effective_batch_size = self.args.eval_batch_size * num_devices
        sampler = DPRDistributedSamplerWithValidation(dataset=self.dataset['test'], seed=self.args.seed,
                                                      disjoint_window_size=effective_batch_size,
                                                      shuffle=False)
        return DataLoader(
            self.dataset['test'],
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=lambda x: self.collator(x, sample=False)
        )

    def collator(self, questions_with_ctxs: List[Dict[str, Union[str, List[Dict]]]], sample=False):
        batch_questions = []
        batch_ctxs = []
        positive_context_ids = []
        batch_is_valid = []

        for question_with_ctxs in questions_with_ctxs:
            batch_questions.append(question_with_ctxs['question'])

            is_valid = [1 for _ in range(1 + self.args.num_negative_ctx + self.args.num_hard_negative_ctx)]
            positives = question_with_ctxs['positive_ctxs']
            negatives = question_with_ctxs['negative_ctxs']
            hard_negatives = question_with_ctxs['hard_negative_ctxs']
            if sample:
                random.shuffle(positives)
                random.shuffle(negatives)
                random.shuffle(hard_negatives)

            positive = positives[0]

            negatives = negatives[:self.args.num_negative_ctx]
            if len(negatives) < self.args.num_negative_ctx:
                for i in range(1 + len(negatives), 1 + self.args.num_negative_ctx):
                    is_valid[i] = 0

                negatives += [{'title': self.pad_token, 'text': self.pad_token}
                              for _ in range(self.args.num_negative_ctx - len(negatives))]

            hard_negatives = hard_negatives[:self.args.num_hard_negative_ctx]
            if len(hard_negatives) < self.args.num_hard_negative_ctx:
                for i in range(1 + self.args.num_negative_ctx + len(hard_negatives),
                               1 + self.args.num_negative_ctx + self.args.num_hard_negative_ctx):
                    is_valid[i] = 0

                hard_negatives += [{'title': self.pad_token, 'text': self.pad_token}
                                   for _ in range(self.args.num_hard_negative_ctx - len(hard_negatives))]

            positive_context_ids.append(len(batch_ctxs))
            batch_is_valid.extend(is_valid)
            batch_ctxs.append(positive)
            batch_ctxs.extend(negatives)
            batch_ctxs.extend(hard_negatives)

        question_input = self.q_tokenizer.batch_encode_plus(
            batch_questions,
            max_length=self.args.question_max_seq_len,
            truncation=True,
            padding='longest',
            return_tensors='pt'
        )

        ctx_input = self.ctx_tokenizer.batch_encode_plus(
            [f' {self.sep_token} '.join([ctx['title'], ctx['text']])
             if self.args.insert_titles and not ctx['text'] == self.pad_token
             else ctx['text']
             for ctx in batch_ctxs],
            max_length=self.args.ctx_max_seq_len,
            truncation=True,
            padding='longest',
            return_tensors='pt'
        )
        positive_context_ids = torch.tensor(positive_context_ids, dtype=torch.long)
        batch_is_valid = torch.tensor(batch_is_valid, dtype=torch.long)
        # this ugly data processing is necessary for Data Parallel as it's only capable of working with Tensors and Lists # noqa
        return question_input['input_ids'], question_input['attention_mask'], \
               ctx_input['input_ids'], ctx_input['attention_mask'], \
               positive_context_ids, batch_is_valid

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> Optional[ArgumentParser]:
        parser = parent_parser.add_argument_group('DPR Datamodule Params')
        parser.add_argument('--num_workers', default=1, type=int, help="kwarg passed to DataLoader")
        parser.add_argument('--train_batch_size', default=1, type=int)
        parser.add_argument('--eval_batch_size', default=1, type=int)
        parser.add_argument('--train_data', default=None, type=str, help='Path to json wile with training data')
        parser.add_argument('--dev_data', default=None, type=str, help='Path to json wile with dev data')
        parser.add_argument('--test_data', default=None, type=str, help='Path to json wile with test data')
        parser.add_argument('--num_negative_ctx', default=0, type=int,
                            help='Number of negative contexts for each example')
        parser.add_argument('--num_hard_negative_ctx', default=1, type=int,
                            help='Number of hard negative contexts for each example')
        parser.add_argument('--question_max_seq_len', default=256, type=int,
                            help='Maximum number of tokens per question')
        parser.add_argument('--ctx_max_seq_len', default=256, type=int,
                            help='Maximum number of tokens per context (title and text)')
        parser.add_argument('--insert_titles', action='store_true')
        return None
