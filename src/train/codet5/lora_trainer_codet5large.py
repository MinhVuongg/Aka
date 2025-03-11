import peft
import torch

from src.config.config import MODEL_NAME
from src.train.lora_trainer import LoRATrainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LoRATrainer_CodeT5Large(LoRATrainer):

    @staticmethod
    def load_model(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def add_lora(self):
        lora_config = peft.LoraConfig(
            r=32, lora_alpha=32, target_modules=["q", "k", "v", "o"],
            lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"
        )
        self.model = peft.get_peft_model(self.model, lora_config)