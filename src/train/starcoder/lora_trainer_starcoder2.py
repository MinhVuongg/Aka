import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.train.lora_trainer import LoRATrainer


class LoRATrainer_StarCoder2(LoRATrainer):

    @staticmethod
    def load_model(model_name):
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
            # quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def add_lora(self):
        lora_config = peft.LoraConfig(
            r=32, lora_alpha=32,
            target_modules=["c_proj", "c_attn", "q_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = peft.get_peft_model(self.model, lora_config)