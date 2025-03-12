from src.config.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from src.train.base_trainer import BaseTrainer
from transformers import Trainer, TrainingArguments
from abc import ABC, abstractmethod


class LoRATrainer(BaseTrainer, ABC):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.add_lora()

    @abstractmethod
    def add_lora(self):
        """Thêm LoRA vào mô hình"""
        pass

    def train(self):
        """
        Huấn luyện
        """
        training_args = self._create_training_args()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=lambda data: {k: v for k, v in data.items() if k != "num_items_in_batch"}
        )


        trainer.train()
        self.train_loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        self.val_loss_history = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
        self.save_model()
        # self.plot_loss(trainer)

    def _create_training_args(self):
        """Tạo training arguments"""
        return TrainingArguments(
            output_dir=MODEL_SAVE_PATH,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            num_train_epochs=EPOCHS,
            save_total_limit=1,
            logging_dir="./logs",
            logging_steps=50,
            logging_strategy="epoch",
            report_to="none",
            learning_rate=LEARNING_RATE,
            optim="adamw_torch"
        )