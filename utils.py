from transformers import AutoModel, AutoTokenizer, AutoConfig
from dpr_model import DPRContextEncoder, DPRQuestionEncoder
from dpr_config import DPRConfig


def init_dpr_component_from_pretrained_model(model_name_or_path, component_class, projection_dim):
    encoder_config = AutoConfig.from_pretrained(model_name_or_path)
    encoder = AutoModel.from_pretrained(model_name_or_path)
    component_config = DPRConfig.from_encoder_config(encoder_config,
                                                     projection_dim=projection_dim)
    model = component_class(component_config)
    model.encoder = encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer
