from src.train.base_trainer import BaseTrainer
from transformers import Trainer, TrainingArguments
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

EPOCHS = int(os.getenv("EPOCHS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
FP16 = os.getenv("FP16", "False").lower() == "true"

class FullFineTuneTrainer(BaseTrainer):
    def train(self):
        """Huấn luyện full fine-tuning."""
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
