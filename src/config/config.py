# Mô hình và dữ liệu
import logging
import os
from enum import Enum
import datetime
from src.utils.utils import normalize_path


class Mode(Enum):
    VAST = 0
    FSOFT_SERVER = 1
    LINH_LOCAL = 2
    DA_LOCAL = 3


class COMMENT_REMOVAL(Enum):
    AST = 0
    REGREX = 1


class MASKING_STRATEGIES(Enum):
    NONE = 0
    RANDOM = 1


class TRAIN_MODES(Enum):
    FULL_FINETUNING = 0
    LORA = 1


class MODEL_TYPES(Enum):
    CODET5_SMALL = "Salesforce/codet5-small"
    CODET5_BASE = "Salesforce/codet5-base"
    CODET5_LARGE = "Salesforce/codet5-large"
    STARCODER2_3B = "bigcode/starcoder2-3b"
    CODET5P_2B = "Salesforce/codet5p-2b"


class TARGET_SELETCTION_STRATEGIES:
    NONE = "Không "  # Không xử lý
    SORT_BY_TOKEN_AND_CUTOFF = "SORT_BY_TOKEN_AND_CUTOFF"  # Khi tiền xử lý, với một danh sách test case -> sort theo số token tăng dần, và lấy n test case để tổng token <= max_target_length

# --------------------------------------------------------------------------------
# Môi trường & Tham số huấn luyện
# --------------------------------------------------------------------------------

mode = Mode.VAST  # <= CHOOSE DEPLOYMENT HERE
TRAIN_TYPE = TRAIN_MODES.LORA
MODEL_TYPE = MODEL_TYPES.CODET5P_2B
MODEL_NAME = str(MODEL_TYPE.value)

MASKING_SOURCE = MASKING_STRATEGIES.NONE
OPTIMIZE_TARGET_STRATEGY = TARGET_SELETCTION_STRATEGIES.SORT_BY_TOKEN_AND_CUTOFF
max_source_length = 512
max_target_length = 512
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.0005
LOGGING_STEP = 50

SAVE_TOTAL_LIMIT = 1
LOGGING_DIR = "./logs"
TRAININGSET_REPORT_LIMIT = 10
VALIDATIONSET_REPORT_LIMIT = 10
REMOVE_COMMENT_MODE = COMMENT_REMOVAL.REGREX  # CHOOSE COMMENT REMOVAL

# --------------------------------------------------------------------------------
# Đường dẫn
# --------------------------------------------------------------------------------
PROJECT_PATH = "/root/Aka"

MAIN_OUTPUT_PATH = normalize_path(f"{PROJECT_PATH}/aka-output")
os.makedirs(MAIN_OUTPUT_PATH, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_PATH = normalize_path(f"{MAIN_OUTPUT_PATH}/{timestamp}")
os.makedirs(OUTPUT_PATH, exist_ok=True)

TRAINSET_RAW = normalize_path(f"{PROJECT_PATH}/data/trainset/raw")
TRAINSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_trainset.json")
VALIDATIONSET_RAW = normalize_path(f"{PROJECT_PATH}/data/validation/raw")
VALIDATIONSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_validationset.json")

model_name_only = str(MODEL_TYPE.value).replace("/", "-")
MODEL_SAVE_PATH = normalize_path(
    f"{OUTPUT_PATH}/model{model_name_only}_ep{EPOCHS}_sour{max_source_length}_tar{max_target_length}_bat{BATCH_SIZE}_train{TRAIN_TYPE.name}"
)

OUTPUT_VALIDATIONSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_validationset.csv")
OUTPUT_TRAINSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_trainingset.csv")
LOG_DIR = normalize_path(f"{OUTPUT_PATH}/training_history.json")
OUTPUT_VALIDATIONSET_HTML = normalize_path(f"{OUTPUT_PATH}/output_validationset_compare.html")
OUTPUT_TRAINSET_HTML = normalize_path(f"{OUTPUT_PATH}/output_trainingset_compare.html")
LOGGER_OUTPUT = normalize_path(f"{OUTPUT_PATH}/log.txt")
