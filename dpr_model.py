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

    def __init__(self, config: DPRConfig):
        super(DPREncoderPreTrainedModel, self).__init__(config)
        self.config = config
        self.encoder: PreTrainedModel = AutoModel.from_config(config.encoder_config)

        if config.encoder_config.is_encoder_decoder and config.use_only_encoder:
            self.encoder: PreTrainedModel = self.encoder.get_encoder()

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
    print(question_model)
    print(context_model)

    encoder_model = AutoModel.from_pretrained('allegro/herbert-base-cased')
    question_model.encoder = encoder_model
    context_model.encoder = encoder_model

    print('============ SAVE/LOAD TEST ===========')
    question_model.save_pretrained('test_question_encoder')
    context_model.save_pretrained('test_context_encoder')

    question_model = DPRQuestionEncoder.from_pretrained('test_question_encoder')
    context_model = DPRContextEncoder.from_pretrained('test_context_encoder')

    print(question_model)
    print(context_model)

    print('============ FORWARD PASS TEST ===========')

    tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')

    inp = tokenizer('Hi, my name is Dima.', return_tensors='pt')

    output_herbert = encoder_model(**inp)[0][:, 0, :]

    output_ctx = context_model(**inp)[0]
    output_question = question_model(**inp)[0]

    print('context model is correct:', torch.allclose(output_herbert, output_ctx))
    print('question model is correct:', torch.allclose(output_herbert, output_question))
