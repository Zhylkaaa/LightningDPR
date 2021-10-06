from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from dpr_model import DPRContextEncoder, DPRQuestionEncoder
from dpr_config import DPRConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig
from utils import (
    init_dpr_component_from_pretrained_model
)


class DPRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, *args, **kwargs) -> Any:
        pass

    def _calculate_loss(self):
        pass

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def configure_optimizers(self):
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', default=None, help='Model name or path used to initialize both question and context encoder. '
                                                                   'For more control use <question/context>_model_name_or_path')
    parser.add_argument('--question_model_name_or_path', default=None, help='Question encoder pretrained model')
    parser.add_argument('--context_model_name_or_path', default=None, help='Context encoder pretrained model')
    parser.add_argument('--question_projection_dim', default=0, help='Question encoder projection dim')
    parser.add_argument('--context_projection_dim', default=0, help='Context encoder projection dim')
    args = parser.parse_args()

    if args.model_name_or_path is not None and args.question_model_name_or_path is not None \
            and args.context_model_name_or_path is not None:
        raise ValueError('Can\'t specify both model_name_or_path with '
                         'question_model_name_or_path and context_model_name_or_path')
    elif args.model_name_or_path is not None:
        question_model_name_or_path = args.model_name_or_path
        context_model_name_or_path = args.model_name_or_path
    elif args.question_model_name_or_path is not None and args.context_model_name_or_path:
        question_model_name_or_path = args.question_model_name_or_path
        context_model_name_or_path = args.context_model_name_or_path
    else:
        raise ValueError('Please specify model_name_or_path or '
                         'question_model_name_or_path with context_model_name_or_path')

    question_model, question_tokenizer = init_dpr_component_from_pretrained_model(
        model_name_or_path=question_model_name_or_path,
        component_class=DPRQuestionEncoder,
        projection_dim=args.question_projection_dim
    )

    context_model, context_tokenizer = init_dpr_component_from_pretrained_model(
        model_name_or_path=context_model_name_or_path,
        component_class=DPRContextEncoder,
        projection_dim=args.context_projection_dim
    )

    # TODO
