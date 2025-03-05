from src.config.config import BATCH_SIZE, EPOCHS, FP16
from src.train.base_trainer import BaseTrainer
import peft
from transformers import Trainer, TrainingArguments
import os
# from dotenv import load_dotenv
# from src.train.callbacks.accuracy_callback import AccuracyCallback

# Load biến môi trường
# load_dotenv()

# EPOCHS = int(os.getenv("EPOCHS", 3))
# BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
# FP16 = os.getenv("FP16", "False").lower() == "true"

class LoRATrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.add_lora()

    def add_lora(self):
        """Thêm LoRA vào mô hình"""
        lora_config = peft.LoraConfig(
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"
        )
        self.model = peft.get_peft_model(self.model, lora_config)

    def train(self):
        """Huấn luyện LoRA."""
        self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
        self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)

        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            num_train_epochs=EPOCHS,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=50,
            logging_strategy="epoch",
            fp16=FP16,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.save_model()
        self.plot_loss(trainer)

