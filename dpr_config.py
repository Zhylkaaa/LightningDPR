import copy

import transformers
from transformers import PretrainedConfig, AutoConfig


class DPRConfig(PretrainedConfig):
    model_type = 'dpr_encoder'
    is_composition = True

    def __init__(self, projection_dim=0, use_only_encoder=True, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        assert 'encoder_config' in kwargs
        encoder_config = kwargs.pop('encoder_config')
        self.projection_dim = projection_dim
        self.use_only_encoder = use_only_encoder
        self.initializer_range = initializer_range
        encoder_model_type = encoder_config.pop('model_type')
        self.encoder_config = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.is_encoder_decoder = False

    @classmethod
    def from_encoder_config(cls, encoder_config: PretrainedConfig, **kwargs):
        return cls(encoder_config=encoder_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['encoder_config'] = self.encoder_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output["transformers_version"] = transformers.__version__
        self.dict_torch_dtype_to_str(output)

        return output
