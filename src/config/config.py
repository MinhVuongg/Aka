# Mô hình và dữ liệu
import os
from enum import Enum

from src.utils.model_utils import ModelType
from src.utils.utils import normalize_path

class Mode(Enum):
    VAST = 0
    FSOFT_SERVER = 1
    LINH_LOCAL = 2
    DA_LOCAL = 3

class COMMENT_REMOVAL(Enum):
    AST = 0
    REGREX = 1

class MASKING(Enum):
    NONE = 0
    RANDOM = 1

class TRAIN_MODE(Enum):
    LORA = 0
    FULL_FINETUNING = 1

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

mode = Mode.VAST  # <= --------------------------------- CHOOSE DEPLOYMENT HERE ---------------------------------
MODEL_NAME = "Salesforce/codet5-base"
MASKING_SOURCE = MASKING.NONE
MODEL_TYPE = ModelType.SEQ2SEQ
max_source_length = 512
max_target_length = 1028
EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
TRAIN_TYPE = TRAIN_MODE.LORA

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

OUTPUT_PATH = normalize_path(f"{PROJECT_PATH}/aka-output")

TRAINSET_RAW = normalize_path(f"{PROJECT_PATH}/data/trainset/raw")
TRAINSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_trainset.json")

VALIDATIONSET_RAW = normalize_path(f"{PROJECT_PATH}/data/validation/raw")
VALIDATIONSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_validationset.json")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

model_name_only = MODEL_NAME.split("/")[-1]  # Lấy ten model sau dấu "/"

# Ten model duoc luu cung voi thong tin lien quan.
MODEL_SAVE_PATH = normalize_path(
    f"{OUTPUT_PATH}/model{model_name_only}_epoch{EPOCHS}_traintype{TRAIN_TYPE}_maxsourcelen{max_source_length}_maxtargetlen{max_target_length}_batch{BATCH_SIZE}")

OUTPUT_VALIDATIONSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_validationset.csv")
OUTPUT_TRAINSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_trainingset.csv")

LOG_DIR = normalize_path(f"{OUTPUT_PATH}/training_history.json")
