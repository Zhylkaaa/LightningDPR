from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoConfig
)
from torch import nn
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

        if config.projection_dim > 0:
            self.encode_proj = nn.Linear(self.encoder.config.hidden_size, config.projection_dim)

        self.init_weights()

    def init_weights(self):
        if self.config.projection_dim > 0:
            self.encode_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.encode_proj.bias is not None:
                self.encode_proj.bias.data.zero_()
        self.encoder.init_weights()


if __name__ == '__main__':
    encoder_config = AutoConfig.from_pretrained('allegro/herbert-base-cased')
    config = DPRConfig.from_encoder_config(encoder_config)
    model = DPREncoderPreTrainedModel(config)
    print(model.config)
    print(model)

    config = DPRConfig.from_encoder_config(encoder_config, projection_dim=100)
    model = DPREncoderPreTrainedModel(config)
    print(model.config)
    print(model)

    encoder_model = AutoModel.from_pretrained('allegro/herbert-base-cased')
    model.encoder = encoder_model

    print('============ SAVE/LOAD TEST ===========')
    model.save_pretrained('test_dir')

    model = DPREncoderPreTrainedModel.from_pretrained('test_dir')
    print(model.config)
    print(model)
