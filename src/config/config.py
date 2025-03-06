# Mô hình và dữ liệu
import os
from enum import Enum
from src.utils.utils import normalize_path

class Mode(Enum):
    VAST = 0
    FSOFT_SERVER = 1
    LINH_LOCAL = 2
    DA_LOCAL = 3

mode = Mode.VAST  # <= --------------------------------- CHOOSE DEPLOYMENT HERE ---------------------------------

if mode == Mode.VAST:
    PROJECT_PATH = "/workspace/aka-llm"
elif mode == Mode.DA_LOCAL:
    PROJECT_PATH = "/Users/ducanhnguyen/Documents/aka-llm"
elif mode == Mode.LINH_LOCAL:
    PROJECT_PATH = "D:/Workspace/AKA/March04/aka-llm"

#--------------------------------------------------------------------------------
# Tham số huấn luyện
#--------------------------------------------------------------------------------
MODEL_NAME = "Salesforce/codet5-small"
max_source_length = 256
max_target_length = 256
EPOCHS = 3
BATCH_SIZE = 16
FP16 = False
TRAIN_TYPE = "lora"  # full hoặc lora

#--------------------------------------------------------------------------------
# Đường dẫn
#--------------------------------------------------------------------------------
OUTPUT_PATH = normalize_path(f"{PROJECT_PATH}/aka-output")

TRAINSET_RAW = normalize_path(f"{PROJECT_PATH}/data/trainset/raw")
TRAINSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_trainset.json")

VALIDATIONSET_RAW = normalize_path(f"{PROJECT_PATH}/data/validation/raw")
VALIDATIONSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_validationset.json")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

MODEL_SAVE_PATH = normalize_path(f"{OUTPUT_PATH}/model_epoch{EPOCHS}_maxsourcelen{max_source_length}_maxtargetlen{max_target_length}_batch{BATCH_SIZE}")
OUTPUT_CSV = normalize_path(f"{OUTPUT_PATH}/output.csv")
LOG_DIR = normalize_path(f"{OUTPUT_PATH}/training_history.json")