from src.config.config import TRAIN_MODES, MODEL_TYPES
from src.train.codet5.lora_trainer_codet5base import LoRATrainer_CodeT5Base
from src.train.codet5.lora_trainer_codet5large import LoRATrainer_CodeT5Large
from src.train.codet5.lora_trainer_codet5small import LoRATrainer_CodeT5Small
from src.train.full_finetune import FullFineTuneTrainer


def load_model_by_type(train_type, model_type):
    if train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.CODET5_SMALL:
        model, tokenizer = LoRATrainer_CodeT5Small.load_model()
    elif train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.CODET5_BASE:
        model, tokenizer = LoRATrainer_CodeT5Base.load_model()
    elif train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.CODET5_LARGE:
        model, tokenizer = LoRATrainer_CodeT5Large.load_model()
    elif train_type == TRAIN_MODES.FULL_FINETUNING:
        model, tokenizer = FullFineTuneTrainer.load_model()
    return model, tokenizer