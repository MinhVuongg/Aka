import json
import os
import sys

from src.config.config import TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS, TRAIN_TYPE, LOG_DIR, \
    VALIDATIONSET_RAW, \
    VALIDATIONSET_DATA_PATH_PROCESS, OUTPUT_TRAINSET_CSV, TRAIN_MODES, MODEL_TYPES, MODEL_TYPE, \
    TRAININGSET_REPORT_LIMIT, VALIDATIONSET_REPORT_LIMIT, OUTPUT_VALIDATIONSET_HTML, OUTPUT_TRAINSET_HTML, MODEL_NAME, \
    OPTIMIZE_TARGET_STRATEGY, TARGET_SELETCTION_STRATEGIES, MODEL_SAVE_PATH
from src.predict.analysis import analysis_result
from src.predict.evaluate import load_model
from src.train.codet5.lora_trainer_codet5base import LoRATrainer_CodeT5Base
from src.train.codet5.lora_trainer_codet5large import LoRATrainer_CodeT5Large
from src.train.codet5.lora_trainer_codet5small import LoRATrainer_CodeT5Small
from src.train.starcoder.lora_trainer_starcoder2 import LoRATrainer_StarCoder2
from src.train.codet5.lora_trainer_codet5p_2b import LoRATrainer_CodeT5P_2B
from src.utils.model_utils import load_model_by_type
from src.utils.mylogger import logger, GREEN, BLUE
from src.utils.token_statistics import count_tokens

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
    logger.info(f"[UET] Load mô hình {MODEL_NAME} và tokenizer tương ứng")
    model, tokenizer = load_model_by_type(TRAIN_TYPE, MODEL_TYPE, MODEL_NAME)

    # TIEN XU LY DU LIEU
    # logger.info("[UET] Bắt đầu tiền xử lý dữ liệu cho training set và test set.")
    logger.info(f"[UET] Phân tích tập học thô {TRAINSET_RAW} để tạo tập học tinh chế. Quá trình này sẽ thực hiện tiền xử lý dữ liệu để mô hình học tốt hơn.", color=BLUE)
    preprocess_dataset2(TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS, tokenizer, optimize_target_strategy=OPTIMIZE_TARGET_STRATEGY)

    logger.info("")
    logger.info(f"[UET] Phân tích tập test thô {VALIDATIONSET_RAW} để tạo tập test tinh chế", color=BLUE)
    preprocess_dataset2(VALIDATIONSET_RAW, VALIDATIONSET_DATA_PATH_PROCESS, tokenizer, optimize_target_strategy=TARGET_SELETCTION_STRATEGIES.NONE)
    logger.info("[UET] Tiền xử lý hoàn tất!")

    logger.info("\n")

    # THONG KE
    logger.info(f"[UET] Thống kê tập học của mô hình ({TRAINSET_DATA_PATH_PROCESS}). Việc thống kê giúp hiểu bản chất dữ liệu.", color=BLUE)
    count_tokens(TRAINSET_DATA_PATH_PROCESS, tokenizer)

    logger.info(f"[UET] Thống kê tập test của mô hình ({VALIDATIONSET_DATA_PATH_PROCESS}). Việc thống kê giúp hiểu bản chất dữ liệu.", color=BLUE)
    count_tokens(VALIDATIONSET_DATA_PATH_PROCESS, tokenizer)

    # TRAIN MODEL
    history = {"train_loss": []}
    logger.info("\n")
    logger.info(f"[UET] {TRAIN_TYPE}", color=BLUE)

    if TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_SMALL:
        trainer = LoRATrainer_CodeT5Small(MODEL_NAME)
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_BASE:
        trainer = LoRATrainer_CodeT5Base(MODEL_NAME)
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_LARGE:
        trainer = LoRATrainer_CodeT5Large(MODEL_NAME)
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.STARCODER2_3B:
        trainer = LoRATrainer_StarCoder2(MODEL_NAME)
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5P_2B:
        model, tokenizer = LoRATrainer_CodeT5P_2B.load_model(MODEL_NAME)

    trainer.train()
    logger.info("[UET] Huấn luyện hoàn tất!")

    history["train_loss"] = trainer.train_loss_history
    history["val_loss"] = trainer.val_loss_history
    logger.info(f"[UET] training log: {json.dumps(history, indent=4)}")
    with open(LOG_DIR, "w") as f:
        json.dump(history, f, indent=4)

    #  Đánh giá mô hình
    logger.info("\n")
    logger.info("[UET] Đánh giá mô hình tren tap validation...", color=BLUE)
    model, dataset, tokenizer = load_model(datapath=VALIDATIONSET_DATA_PATH_PROCESS, model_name=MODEL_SAVE_PATH)
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_VALIDATIONSET_CSV, limit=VALIDATIONSET_REPORT_LIMIT)
    analysis_result(OUTPUT_VALIDATIONSET_CSV, OUTPUT_VALIDATIONSET_HTML)
    logger.info("[UET] Hoàn tất đánh giá trên tập validation!")

    logger.info("\n")
    logger.info("[UET] Đánh giá mô hình tren tap training...", color=BLUE)
    model, dataset, tokenizer = load_model(datapath=TRAINSET_DATA_PATH_PROCESS, model_name=MODEL_SAVE_PATH)
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_TRAINSET_CSV, limit=TRAININGSET_REPORT_LIMIT)
    analysis_result(OUTPUT_TRAINSET_CSV, OUTPUT_TRAINSET_HTML)
    logger.info("[UET] Hoàn tất đánh giá trên tập training!")


if __name__ == "__main__":
    main()
