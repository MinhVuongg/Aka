import peft
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config.config import MODEL_NAME
from src.train.lora_trainer import LoRATrainer


class LoRATrainer_CodeT5Base(LoRATrainer):

    @staticmethod
    def load_model():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if torch.cuda.is_available() else None
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer

    def add_lora(self):
        lora_config = peft.LoraConfig(
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"
        )
        self.model = peft.get_peft_model(self.model, lora_config)