from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
from dpr_model import DPRContextEncoder, DPRQuestionEncoder, DPRReaderModel
from dpr_config import DPRConfig


def init_dpr_component_from_pretrained_model(model_name_or_path, component_class, projection_dim, is_dpr_checkpoint=False):
    if not is_dpr_checkpoint:
        encoder_config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        encoder = AutoModel.from_pretrained(model_name_or_path)
        if encoder_config.is_encoder_decoder:
            encoder = encoder.get_encoder()

        component_config = DPRConfig.from_encoder_config(encoder_config,
                                                         projection_dim=projection_dim)
        model = component_class(component_config)
        model.encoder = encoder
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        model = component_class.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer

def init_dpr_reader(model_name_or_path, is_reader_checkpoint=False):
    if not is_reader_checkpoint:
        encoder_config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        encoder = AutoModel.from_pretrained(model_name_or_path)
        if encoder_config.is_encoder_decoder:
            encoder = encoder.get_encoder()

        reader_config = DPRConfig.from_encoder_config(encoder_config)
        model = DPRReaderModel(reader_config)
        model.encoder = encoder
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        model = DPRReaderModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer
