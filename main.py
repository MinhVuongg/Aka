import json
import logging
import os
import sys

from src.config.config import DATA_PATH_RAW, DATA_PATH_PROCESS, OUTPUT_PATH, TRAIN_TYPE, LOG_DIR
from src.predict.evaluate import load_model

# Xác định đường dẫn thư mục gốc của dự án
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
# LOG_DIR = os.path.join(ROOT_DIR, "log")

# Thêm `src` vào sys.path để có thể import module
sys.path.insert(0, SRC_DIR)

# Import các module sau khi đã thêm vào sys.path
from src.data.preprocess import preprocess_dataset
from src.train.full_finetune import FullFineTuneTrainer
from src.train.lora_trainer import LoRATrainer
from src.predict import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    # ,
    # handlers=[
    #     logging.FileHandler(os.path.join(LOG_DIR, "training.log"), encoding="utf-8", mode="a"),  # Append log vào file
    #     logging.StreamHandler(sys.stdout)  # Hiển thị log trên console
    # ]
)

logger = logging.getLogger(__name__)

def main():
    """Hàm chính để chạy pipeline"""
    #  Tiền xử lý dữ liệu
    # raw_data_path = os.path.join(DATA_DIR, "raw")
    # processed_data_path = os.path.join(DATA_DIR, "processed.json")

    logger.info("[UET] Bắt đầu tiền xử lý dữ liệu...")

    # if os.path.exists(OUTPUT_PATH):
    #     # os.remove(OUTPUT_PATH)
    #     os.makedirs(OUTPUT_PATH)
    history = {"train_loss": []}

    preprocess_dataset(DATA_PATH_RAW, DATA_PATH_PROCESS)
    # preprocess_dataset(DATA_PATH_RAW, DATA_PATH_PROCESS, overwrite=False)
    logger.info("[UET] Tiền xử lý hoàn tất!")

    #  Huấn luyện mô hình
    # trainer_type = os.getenv("TRAIN_TYPE", "full")  # full hoặc lora

    logger.info(f"[UET] Bắt đầu huấn luyện mô hình ({TRAIN_TYPE})...")

    if TRAIN_TYPE == "lora":
        logger.info("[UET] LoRATrainer")
        trainer = LoRATrainer()
    else:
        logger.info("[UET] FullFineTuneTrainer")
        trainer = FullFineTuneTrainer()

    trainer.train()
    logger.info("[UET] Huấn luyện hoàn tất!")

    history["train_loss"] = trainer.loss_history
    history["val_loss"] = trainer.val_loss_history
    logger.info(f"[UET] training log: {json.dumps(history, indent=4)}")
    with open(LOG_DIR, "w") as f:
        json.dump(history, f, indent=4)

    #  Đánh giá mô hình
    logger.info("[UET] Đánh giá mô hình...")
    model, dataset, tokenizer = load_model()
    evaluate_model(dataset, tokenizer, model)
    logger.info("[UET] Hoàn tất đánh giá!")


if __name__ == "__main__":
    main()
