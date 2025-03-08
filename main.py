import json
import os
import sys

from src.config.config import TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS, TRAIN_TYPE, LOG_DIR, \
    VALIDATIONSET_RAW, \
    VALIDATIONSET_DATA_PATH_PROCESS, OUTPUT_TRAINSET_CSV, TRAIN_MODES, MODEL_TYPES, MODEL_TYPE, \
    TRAININGSET_REPORT_LIMIT, VALIDATIONSET_REPORT_LIMIT, OUTPUT_VALIDATIONSET_HTML, OUTPUT_TRAINSET_HTML
from src.predict.analysis import analysis_result
from src.predict.evaluate import load_model
from src.train.codet5.lora_trainer_codet5base import LoRATrainer_CodeT5Base
from src.train.codet5.lora_trainer_codet5large import LoRATrainer_CodeT5Large
from src.train.codet5.lora_trainer_codet5small import LoRATrainer_CodeT5Small
from src.utils.mylogger import logger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
# LOG_DIR = os.path.join(ROOT_DIR, "log")

# Thêm `src` vào sys.path để có thể import module
sys.path.insert(0, SRC_DIR)

# Import các module sau khi đã thêm vào sys.path
from src.data.preprocess import preprocess_dataset2
from src.train.full_finetune import FullFineTuneTrainer
from src.predict import evaluate_model
from src.config.config import OUTPUT_VALIDATIONSET_CSV

def main():
    # TIEN XU LY DU LIEU
    logger.info("[UET] Bắt đầu tiền xử lý dữ liệu cho training set và test set.")
    logger.info(f"[UET] Training set path = {TRAINSET_RAW}")
    preprocess_dataset2(TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS)
    logger.info(f"[UET] Test set path = {VALIDATIONSET_RAW}")
    preprocess_dataset2(VALIDATIONSET_RAW, VALIDATIONSET_DATA_PATH_PROCESS)
    logger.info("[UET] Tiền xử lý hoàn tất!")

    logger.info("\n")
    logger.info(f"[UET] Bắt đầu huấn luyện mô hình ({TRAIN_TYPE})...")

    # TRAIN MODEL
    history = {"train_loss": []}
    logger.info(f"[UET] {TRAIN_TYPE}")

    if TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_SMALL:
        trainer = LoRATrainer_CodeT5Small()
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_BASE:
        trainer = LoRATrainer_CodeT5Base()
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_LARGE:
        trainer = LoRATrainer_CodeT5Large()
    elif TRAIN_TYPE == TRAIN_MODES.FULL_FINETUNING:
        trainer = FullFineTuneTrainer()

    trainer.train()
    logger.info("[UET] Huấn luyện hoàn tất!")

    history["train_loss"] = trainer.train_loss_history
    history["val_loss"] = trainer.val_loss_history
    logger.info(f"[UET] training log: {json.dumps(history, indent=4)}")
    with open(LOG_DIR, "w") as f:
        json.dump(history, f, indent=4)

    #  Đánh giá mô hình
    logger.info("\n")
    logger.info("[UET] Đánh giá mô hình tren tap validation...")
    model, dataset, tokenizer = load_model(datapath=VALIDATIONSET_DATA_PATH_PROCESS)
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_VALIDATIONSET_CSV, limit=VALIDATIONSET_REPORT_LIMIT)
    analysis_result(OUTPUT_VALIDATIONSET_CSV, OUTPUT_VALIDATIONSET_HTML)
    logger.info("[UET] Hoàn tất đánh giá trên tập validation!")

    logger.info("\n")
    logger.info("[UET] Đánh giá mô hình tren tap training...")
    model, dataset, tokenizer = load_model(datapath=TRAINSET_DATA_PATH_PROCESS)
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_TRAINSET_CSV, limit=TRAININGSET_REPORT_LIMIT)
    analysis_result(OUTPUT_TRAINSET_CSV, OUTPUT_TRAINSET_HTML)
    logger.info("[UET] Hoàn tất đánh giá trên tập training!")


if __name__ == "__main__":
    main()
