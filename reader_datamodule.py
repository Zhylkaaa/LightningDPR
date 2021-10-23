import logging
import random
from argparse import ArgumentParser, Namespace
import json
from typing import Union, List, Tuple, Dict, Optional
import math
import re

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.distributed import T_co
import torch.distributed as dist

from transformers import PreTrainedTokenizer, AutoTokenizer
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class DPRReaderDatasetModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: str,
            args: Namespace,
    ):
        super(DPRReaderDatasetModule, self).__init__()
        self.args = args
        self.tokenizer_path = tokenizer
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.sep_token = self.tokenizer.sep_token \
            if self.tokenizer.sep_token is not None else self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token

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

        return DataLoader(
            self.dataset['train'],
            shuffle=False if dist.is_initialized() else True,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=lambda x: self.collator(x, sample=True)
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.dataset['dev'] is None:
            return None

        return DataLoader(
            self.dataset['dev'],
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=lambda x: self.collator(x, sample=False)
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.dataset['test'] is None:
            return None

        return DataLoader(
            self.dataset['test'],
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=lambda x: self.collator(x, sample=False)
        )

    def collator(self, questions_with_ctxs: List[Dict[str, Union[str, List[Dict]]]], sample=False):
        def normalize_text(text):
            return re.sub('\s+', ' ', text).strip()

        def convert_passages_to_string(passages):
            return [normalize_text(ctx['title']) + f' {self.sep_token} ' + normalize_text(ctx['text'])
                    if self.args.insert_titles
                    else normalize_text(ctx['text'])
                    for ctx in passages]

        def truncate_question(question):
            words = []
            words_len = 0
            for word in question.split():
                words_len += len(self.tokenizer.tokenize(word))
                if words_len > self.args.max_question_len:
                    break
                words.append(word)
            return ' '.join(words)

        examples = []
        valid = []

        for question_with_ctxs in questions_with_ctxs:
            is_valid = torch.ones(1 + self.args.num_negative_ctx + self.args.num_hard_negative_ctx)
            question = truncate_question(normalize_text(question_with_ctxs['question']))

            positives = question_with_ctxs['positive_ctxs']
            negatives = question_with_ctxs['negative_ctxs']
            hard_negatives = question_with_ctxs['hard_negative_ctxs']

            if sample:
                random.shuffle(positives)
                random.shuffle(negatives)
                random.shuffle(hard_negatives)
            positive_context = positives[0]

            negatives = negatives[:self.args.num_negative_ctx]
            if len(negatives) < self.args.num_negative_ctx:
                for i in range(len(negatives), self.args.num_negative_ctx):
                    is_valid[1 + i] = 0
                negatives += [{'title': self.pad_token, 'text': self.pad_token}
                              for _ in range(self.args.num_negative_ctx - len(negatives))]
            hard_negatives = hard_negatives[:self.args.num_hard_negative_ctx]

            if len(hard_negatives) < self.args.num_hard_negative_ctx:
                for i in range(len(hard_negatives), self.args.num_hard_negative_ctx):
                    is_valid[1 + self.args.num_negative_ctx + i] = 0
                hard_negatives += [{'title': self.pad_token, 'text': self.pad_token}
                                   for _ in range(self.args.num_hard_negative_ctx - len(hard_negatives))]

            positive_context = convert_passages_to_string([positive_context])[0]
            negative_ctxs = convert_passages_to_string(negatives)
            hard_negatives = convert_passages_to_string(hard_negatives)

            examples.append(self.tokenizer(
                [question] * (1 + len(negative_ctxs) + len(hard_negatives)),
                text_pair=[positive_context] + negative_ctxs + hard_negatives,
                truncation='only_second',
                padding='max_length',
                max_length=self.args.max_seq_len,
                return_tensors='pt'
            ))
            valid.append(is_valid.unsqueeze(0))

        valid = torch.cat(valid, dim=0)
        keys = examples[0].keys()
        examples = {k: torch.cat([example[k].unsqueeze(0) for example in examples], dim=0) for k in keys}
        return examples, valid.type(torch.bool)


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
        parser.add_argument('--max_question_len', default=128, type=int,
                            help='max len of question (should be strictly less than max_seq_len)')
        parser.add_argument('--max_seq_len', default=256, type=int,
                            help='Maximum number of tokens per passage')
        parser.add_argument('--insert_titles', action='store_true')
        return None
