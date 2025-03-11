from src.config.config import TRAIN_MODES, MODEL_TYPES
from src.train.codet5.lora_trainer_codet5base import LoRATrainer_CodeT5Base
from src.train.codet5.lora_trainer_codet5large import LoRATrainer_CodeT5Large
from src.train.codet5.lora_trainer_codet5small import LoRATrainer_CodeT5Small
from src.train.starcoder.lora_trainer_starcoder2 import LoRATrainer_StarCoder2

from src.train.full_finetune import FullFineTuneTrainer


def load_model_by_type(train_type, model_type, model_name):
    if train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.CODET5_SMALL:
        model, tokenizer = LoRATrainer_CodeT5Small.load_model(model_name)
    elif train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.CODET5_BASE:
        model, tokenizer = LoRATrainer_CodeT5Base.load_model(model_name)
    elif train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.CODET5_LARGE:
        model, tokenizer = LoRATrainer_CodeT5Large.load_model(model_name)
    elif train_type == TRAIN_MODES.LORA and model_type == MODEL_TYPES.STARCODER2_3B:
        model, tokenizer = LoRATrainer_StarCoder2.load_model(model_name)
    return model, tokenizer