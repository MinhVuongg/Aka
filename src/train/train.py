import logging
import os
import sys
# from dotenv import load_dotenv
from transformers import Trainer, TrainingArguments, TrainerCallback
from base_trainer import BaseTrainer
from transformers import AutoTokenizer

from src.config.config import BATCH_SIZE, EPOCHS, TRAIN_TYPE, LOG_DIR

# Load biến môi trường từ file .env
# load_dotenv()

# Định nghĩa thư mục log
# LOG_DIR = os.path.join(os.path.dirname(__file__), "../..", "log")
# os.makedirs(LOG_DIR, exist_ok=True)  # Tạo thư mục nếu chưa có
#
# # Cấu hình file log
# LOG_FILE = os.path.join(LOG_DIR, "training.log")
#
# # Xoá nội dung file training.log
# with open(LOG_FILE, encoding="utf-8", mode="w") as f:
#     pass

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    # ,
    # handlers=[
    #     logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),  # Append log vào file
    #     logging.StreamHandler(sys.stdout)  # Hiển thị log trên console
    # ]
)

logger = logging.getLogger(__name__)
# logger.addHandler(logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"))
# logger.addHandler(logging.StreamHandler(sys.stdout))
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.flush = lambda: handler.stream.flush()  # Đảm bảo flush log ngay sau mỗi ghi

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logger.info(f"{logs}")
            logger.handlers[0].flush()

class CustomTrainer(BaseTrainer):
    def __init__(self):
        """Khởi tạo trainer với tokenizer"""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_source_length = 256
        self.max_target_length = 256

    def preprocess_function_with_masking(self, examples):
        """Tiền xử lý dữ liệu với Masking"""
        sources = [str(source) + " <SEP>" for source in examples.get("source", [])]
        targets = [str(target) for target in examples.get("target", [])]

        model_inputs = self.tokenizer(
            sources, max_length=self.max_source_length, truncation=True, padding="max_length"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self):
        """Huấn luyện mô hình với logging"""
        logger.info("[UET] Bắt đầu quá trình huấn luyện...")
        logger.handlers[0].flush()

        try:
            if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
                logger.error("[UET] Dữ liệu huấn luyện hoặc validation bị rỗng!")
                logger.handlers[0].flush()
                return

            logger.info("[UET] Đang tiền xử lý dữ liệu với Masking...")
            self.train_dataset = self.train_dataset.map(self.preprocess_function_with_masking, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess_function_with_masking, batched=True)
            logger.info(f"[UET] Dữ liệu sau tiền xử lý - Train: {len(self.train_dataset)}, Validation: {len(self.val_dataset)}")
            logger.handlers[0].flush()

            training_args = TrainingArguments(
                output_dir=self.model_save_path,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                num_train_epochs=EPOCHS,
                save_total_limit=2,
                logging_dir=LOG_DIR,
                logging_steps=1,
                report_to="none"
            )

            logger.info(f"[UET] Training Arguments: {training_args}")
            logger.handlers[0].flush()

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                tokenizer=self.tokenizer,
                callbacks=[CustomLoggingCallback()]
            )

            trainer.train()

            # Ghi lại train loss sau mỗi epoch
            for log in trainer.state.log_history:
                if "loss" in log:
                    logger.info(f"[UET] Train loss: {log['loss']} - Epoch: {log.get('epoch', 'N/A')}")
                    logger.handlers[0].flush()

            self.save_model()
            logger.info("[UET] Huấn luyện hoàn tất!")
            logger.handlers[0].flush()

        except Exception as e:
            logger.error(f"[UET] Lỗi trong quá trình huấn luyện: {str(e)}", exc_info=True)
            logger.handlers[0].flush()

if __name__ == "__main__":
    trainer = CustomTrainer()
    trainer.train()
