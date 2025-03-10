import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config.config import MODEL_NAME
from src.train.lora_trainer import LoRATrainer


class LoRATrainer_StarCoder2(LoRATrainer):

    @staticmethod
    def load_model():
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer

    def add_lora(self):
        lora_config = peft.LoraConfig(
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = peft.get_peft_model(self.model, lora_config)