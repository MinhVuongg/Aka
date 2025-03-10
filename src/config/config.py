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


# --------------------------------------------------------------------------------
# 
#                           Môi trường & Tham số huấn luyện (MODIFY HERE)
# 
# --------------------------------------------------------------------------------

# -----          ---------------------          -------------          -------------          ----------------------
# EXAMPLE 1: GIÁ TRỊ SOURCE VÀ LENGTH KHÁ LỚN => CẦN GIẢM BATCH_SIZE THÌ MỚI TRAIN ĐƯỢC TRÊN A100 80GB
# mode = Mode.VAST
# MODEL_NAME = "Salesforce/codet5-base"
# MASKING_SOURCE = MASKING.NONE
# MODEL_TYPE = ModelType.SEQ2SEQ
# max_source_length = 512
# max_target_length = 1028 
# EPOCHS = 100
# BATCH_SIZE = 2
# FP16 = False
# TRAIN_TYPE = "lora"  # full hoặc lora

# -----          ---------------------          -------------          -------------          ----------------------
# EXAMPLE 2: GIÁ TRỊ SOURCE VÀ LENGTH KHÁ NHỎ => CẦN TĂNG BATCH_SIZE ĐỂ TỐI ƯU TÀI NGUYÊN TRÊN A100 80GB
# mode = Mode.VAST
# MODEL_NAME = "Salesforce/codet5-base"
# MASKING_SOURCE = MASKING.NONE
# MODEL_TYPE = ModelType.SEQ2SEQ
# max_source_length = 256
# max_target_length = 256
# EPOCHS = 100
# BATCH_SIZE = 64
# FP16 = False
# TRAIN_TYPE = "lora"  # full hoặc lora

mode = Mode.DA_LOCAL  # <= --------------------------------- CHOOSE DEPLOYMENT HERE ---------------------------------
TRAIN_TYPE = TRAIN_MODES.LORA

MODEL_TYPE = MODEL_TYPES.CODET5_SMALL
MODEL_NAME = str(MODEL_TYPE.value)

MASKING_SOURCE = MASKING_STRATEGIES.NONE

max_source_length = 32
max_target_length = 32
EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
LOGGING_STEP = 50

SAVE_TOTAL_LIMIT = 1
LOGGING_DIR = "./logs"

# Khi train mô hình xong, ta sẽ đánh giá trên tập training set. Nếu set None thì đánh giá toàn bộ. Nếu không, hãy set một số cụ thể.
TRAININGSET_REPORT_LIMIT = 10

# Khi train mô hình xong, ta sẽ đánh giá trên tập validation set. Nếu set None thì đánh giá toàn bộ. Nếu không, hãy set một số cụ thể.
VALIDATIONSET_REPORT_LIMIT = 10


# Mot vai machine ko chay duoc AST option. Neu chay duoc AST thi remove comment tot hon.
REMOVE_COMMENT_MODE = COMMENT_REMOVAL.REGREX  # <= --------------------------------- CHOOSE COMMENT REMOVAL ---------------------------------

# --------------------------------------------------------------------------------
# Đường dẫn (SHOULD NOT MODIFY)
# --------------------------------------------------------------------------------
if mode == Mode.VAST:
    PROJECT_PATH = "/workspace/aka-llm"
elif mode == Mode.DA_LOCAL:
    PROJECT_PATH = "/Users/ducanhnguyen/Documents/aka-llm"
elif mode == Mode.LINH_LOCAL:
    PROJECT_PATH = "D:/Workspace/AKA/March04/aka-llm"
elif mode == Mode.FSOFT_SERVER:
    PROJECT_PATH = r"C:\Users\CuongPN8.IVI\Documents\uet-llm\version0703"

# Tạo đường dẫn aka-output
MAIN_OUTPUT_PATH = normalize_path(f"{PROJECT_PATH}/aka-output")
if not os.path.exists(MAIN_OUTPUT_PATH):
    os.makedirs(MAIN_OUTPUT_PATH)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_PATH = normalize_path(f"{MAIN_OUTPUT_PATH}/{timestamp}")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

TRAINSET_RAW = normalize_path(f"{PROJECT_PATH}/data/trainset/raw")
TRAINSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_trainset.json")

VALIDATIONSET_RAW = normalize_path(f"{PROJECT_PATH}/data/validation/raw")
VALIDATIONSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_validationset.json")


# 'Salesforce/codet5-small' => 'Salesforce-codet5-small'
model_name_only = str(MODEL_TYPE.value).replace("/", "-")  # Lấy ten model sau dấu "/"

# Ten model duoc luu cung voi thong tin lien quan.
MODEL_SAVE_PATH = normalize_path(
    f"{OUTPUT_PATH}/model{model_name_only}_ep{EPOCHS}_sour{max_source_length}_tar{max_target_length}_bat{BATCH_SIZE}_train{TRAIN_TYPE.name}")

OUTPUT_VALIDATIONSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_validationset.csv")
OUTPUT_TRAINSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_trainingset.csv")

LOG_DIR = normalize_path(f"{OUTPUT_PATH}/training_history.json")

OUTPUT_VALIDATIONSET_HTML = normalize_path(f"{OUTPUT_PATH}/output_validationset_compare.html")
OUTPUT_TRAINSET_HTML = normalize_path(f"{OUTPUT_PATH}/output_trainingset_compare.html")

LOGGER_OUTPUT = normalize_path(f"{OUTPUT_PATH}/log.txt")