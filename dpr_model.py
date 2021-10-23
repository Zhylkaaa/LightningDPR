from typing import Optional

import torch
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoConfig,
    AutoTokenizer
)
from torch import Tensor, nn
from dpr_config import DPRConfig
import transformers


class DPREncoderPreTrainedModel(PreTrainedModel):
    config_class = DPRConfig
    base_class_prefix = 'dpr_encoder'
    supports_gradient_checkpointing = True

    def __init__(self, config: DPRConfig):
        super(DPREncoderPreTrainedModel, self).__init__(config)
        self.config = config
        self.encoder: PreTrainedModel = AutoModel.from_config(config.encoder_config)

        if config.encoder_config.is_encoder_decoder and config.use_only_encoder:
            self.encoder: PreTrainedModel = self.encoder.get_encoder()

        self.supports_gradient_checkpointing = self.encoder.supports_gradient_checkpointing

        self.encode_proj = None
        if config.projection_dim > 0:
            self.encode_proj = nn.Linear(self.encoder.config.hidden_size, config.projection_dim)

        self.init_weights()

    def init_weights(self):
        if self.config.projection_dim > 0:
            self.encode_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.encode_proj.bias is not None:
                self.encode_proj.bias.data.zero_()
        self.encoder.init_weights()

    def _set_gradient_checkpointing(self, module, gradient_checkpointing=False, deepspeed_checkpointing=False):
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(use_deepspeed=deepspeed_checkpointing)
        else:
            self.encoder.gradient_checkpointing_disable()


class DPRContextEncoder(DPREncoderPreTrainedModel):
    base_model_prefix = "ctx_encoder"

    def forward(self,
                input_ids,
                attention_mask: Optional[Tensor],
                **kwargs):
        sequence_output, *other = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        ).values()  # NOTE: assume first output element is last_hidden_states

        sequence_representations = sequence_output[:, 0, :]
        if self.encode_proj:
            sequence_representations = self.encode_proj(sequence_representations)

        return (sequence_representations, sequence_output, *other)


class DPRQuestionEncoder(DPREncoderPreTrainedModel):
    base_model_prefix = "question_encoder"

    def forward(self,
                input_ids,
                attention_mask: Optional[Tensor] = None,
                **kwargs):
        sequence_output, *other = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        ).values()  # NOTE: assume first output element is last_hidden_states

        sequence_representations = sequence_output[:, 0, :]
        if self.encode_proj:
            sequence_representations = self.encode_proj(sequence_representations)

        return (sequence_representations, sequence_output, *other)


class DPRReaderModel(PreTrainedModel):
    config_class = DPRConfig
    base_class_prefix = 'dpr_reader'

    def __init__(self, config: DPRConfig):
        super(DPRReaderModel, self).__init__(config)
        self.config = config
        self.encoder: PreTrainedModel = AutoModel.from_config(config.encoder_config)

        if config.encoder_config.is_encoder_decoder and config.use_only_encoder:
            self.encoder: PreTrainedModel = self.encoder.get_encoder()

        classifier_dropout = (
            config.ranker_dropout
            if config.ranker_dropout is not None
            else config.encoder_config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.ranker = nn.Linear(self.encoder.config.hidden_size, 1)
        #self.span_predictor = nn.Linear(self.encoder.config.hidden_size, 2)

        self.init_weights()

    def init_weights(self):
        self.ranker.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.ranker.bias is not None:
            self.ranker.bias.data.zero_()

        # self.span_predictor.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # if self.span_predictor.bias is not None:
        #     self.span_predictor.bias.data.zero_()

        self.encoder.init_weights()

    def forward(self,
                input_ids,
                attention_mask: Optional[Tensor] = None,
                **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.ranker(pooled_output)

        return logits


if __name__ == '__main__':
    encoder_config = AutoConfig.from_pretrained('allegro/herbert-base-cased')
    config = DPRConfig.from_encoder_config(encoder_config, projection_dim=100)
    question_model = DPRQuestionEncoder(config)
    context_model = DPRContextEncoder(config)
    print(question_model)
    print(context_model)

    config = DPRConfig.from_encoder_config(encoder_config)
    question_model = DPRQuestionEncoder(config)
    context_model = DPRContextEncoder(config)
    reader_model = DPRReaderModel(config)
    print(question_model)
    print(context_model)
    print(reader_model)

    encoder_model = AutoModel.from_pretrained('allegro/herbert-base-cased')
    question_model.encoder = encoder_model
    context_model.encoder = encoder_model
    reader_model.encoder = encoder_model

    print('============ SAVE/LOAD TEST ===========')
    question_model.save_pretrained('test_question_encoder')
    context_model.save_pretrained('test_context_encoder')
    reader_model.save_pretrained('test_reader')

    question_model = DPRQuestionEncoder.from_pretrained('test_question_encoder')
    context_model = DPRContextEncoder.from_pretrained('test_context_encoder')
    reader_model = DPRReaderModel.from_pretrained('test_reader')

    print(question_model)
    print(context_model)
    print(reader_model)

    print('============ FORWARD PASS TEST ===========')

    tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')

    inp = tokenizer('Hi, my name is Dima.', return_tensors='pt')

    output_herbert = encoder_model(**inp)[0][:, 0, :]

    output_ctx = context_model(**inp)[0]
    output_question = question_model(**inp)[0]
    output_logits = reader_model(**inp)

    print('context model is correct:', torch.allclose(output_herbert, output_ctx))
    print('question model is correct:', torch.allclose(output_herbert, output_question))
    print('reader output shape is correct:', output_logits.shape == (1, 1))
