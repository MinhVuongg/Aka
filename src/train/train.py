import logging
import os

# Định nghĩa thư mục log
LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(LOG_DIR, exist_ok=True)

# Cấu hình logging
LOG_FILE = os.path.join(LOG_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from transformers import Trainer, TrainingArguments
from base_trainer import BaseTrainer


class CustomTrainer(BaseTrainer):
    def train(self):
        """Huấn luyện mô hình với logging."""
        logger.info(" Bắt đầu quá trình huấn luyện...")
        try:
            self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)

            training_args = TrainingArguments(
                output_dir=self.model_save_path,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                evaluation_strategy="epoch",
                num_train_epochs=5,
                save_total_limit=2,
                logging_dir="./log",
                logging_steps=50,
                fp16=False,
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
            logger.info(" Huấn luyện hoàn tất!")
        except Exception as e:
            logger.error(f" Lỗi trong quá trình huấn luyện: {e}")


if __name__ == "__main__":
    trainer = CustomTrainer()
    trainer.train()
