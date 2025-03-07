import logging

from src.config.config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from src.train.base_trainer import BaseTrainer
from transformers import Trainer, TrainingArguments, TrainerCallback
import os
# from dotenv import load_dotenv

# Load biến môi trường
# load_dotenv()
#
# EPOCHS = int(os.getenv("EPOCHS", 5))
# BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
# FP16 = os.getenv("FP16", "False").lower() == "true"
# import logging
logger = logging.getLogger(__name__)

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logger.info(f"[UET] {logs}")

class FullFineTuneTrainer(BaseTrainer):
    def train(self):
        """Huấn luyện full fine-tuning."""
        self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
        self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)

        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            num_train_epochs=EPOCHS,
            save_total_limit=1,
            logging_dir=self.log_dir,
            logging_steps=50,
            report_to="none",
            learning_rate=LEARNING_RATE,
            optim="adamw_torch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[CustomLoggingCallback()]
        )

        trainer.train()
        self.loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        self.val_loss_history = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
        self.save_model()
        self.plot_loss(trainer)