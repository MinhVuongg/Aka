import json
import logging
import os
import sys

from src.config.config import TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS, TRAIN_TYPE, LOG_DIR, \
    VALIDATIONSET_RAW, \
    VALIDATIONSET_DATA_PATH_PROCESS, OUTPUT_TRAINSET_CSV, TRAIN_MODE
from src.predict.analysis import analysis_result
from src.predict.evaluate import load_model

# Xác định đường dẫn thư mục gốc của dự án
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
# LOG_DIR = os.path.join(ROOT_DIR, "log")

# Thêm `src` vào sys.path để có thể import module
sys.path.insert(0, SRC_DIR)

# Import các module sau khi đã thêm vào sys.path
from src.data.preprocess import preprocess_dataset, preprocess_dataset2
from src.train.full_finetune import FullFineTuneTrainer
from src.train.lora_trainer import LoRATrainer
from src.predict import evaluate_model
from src.config.config import OUTPUT_VALIDATIONSET_CSV

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
    # TIEN XU LY DU LIEU
    logger.info("[UET] Bắt đầu tiền xử lý dữ liệu cho training set và test set.")
    logger.info(f"\t[UET] Training set path = {TRAINSET_RAW}")
    preprocess_dataset2(TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS)
    logger.info(f"\t[UET] Test set path = {VALIDATIONSET_RAW}")
    preprocess_dataset2(VALIDATIONSET_RAW, VALIDATIONSET_DATA_PATH_PROCESS)
    logger.info("[UET] Tiền xử lý hoàn tất!")

    logger.info(f"[UET] Bắt đầu huấn luyện mô hình ({TRAIN_TYPE})...")

    # TRAIN MODEL
    history = {"train_loss": []}

    if TRAIN_TYPE == TRAIN_MODE.LORA:
        logger.info("[UET] LoRATrainer")
        trainer = LoRATrainer()
    elif TRAIN_TYPE == TRAIN_MODE.FULL_FINETUNING:
        logger.info("[UET] FullFineTuneTrainer")
        trainer = FullFineTuneTrainer()

    trainer.train()
    logger.info("[UET] Huấn luyện hoàn tất!")

    history["train_loss"] = trainer.train_loss_history
    history["val_loss"] = trainer.val_loss_history
    logger.info(f"[UET] training log: {json.dumps(history, indent=4)}")
    with open(LOG_DIR, "w") as f:
        json.dump(history, f, indent=4)

    #  Đánh giá mô hình
    logger.info("[UET] Đánh giá mô hình tren tap validation...")
    model, dataset, tokenizer = load_model(datapath=VALIDATIONSET_DATA_PATH_PROCESS)
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_VALIDATIONSET_CSV)
    analysis_result(OUTPUT_VALIDATIONSET_CSV)
    logger.info("[UET] Hoàn tất đánh giá trên tập validation!")

    logger.info("[UET] Đánh giá mô hình tren tap training...")
    model, dataset, tokenizer = load_model(datapath=TRAINSET_DATA_PATH_PROCESS)
    dataset = dataset[:100]
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_TRAINSET_CSV)
    analysis_result(OUTPUT_TRAINSET_CSV)
    logger.info("[UET] Hoàn tất đánh giá trên tập training!")

if __name__ == "__main__":
    main()
