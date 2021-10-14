from transformers import AutoTokenizer, AutoModel, AutoConfig
from dpr_model import DPRContextEncoder, DPRQuestionEncoder
from dpr_config import DPRConfig

if __name__ == '__main__':
    encoder_config = AutoConfig.from_pretrained('google/mt5-xl')
    config = DPRConfig.from_encoder_config(encoder_config, projection_dim=0)
    question_model = DPRQuestionEncoder(config)
    context_model = DPRContextEncoder(config)

    question_model.save_pretrained('mt5_question_encoder')
    context_model.save_pretrained('mt5_context_encoder')

    tokenizer = AutoTokenizer.from_pretrained('google/mt5-xl')
    tokenizer.save_pretrained('mt5_question_encoder')
    tokenizer.save_pretrained('mt5_context_encoder')
