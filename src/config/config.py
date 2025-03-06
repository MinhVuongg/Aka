# Mô hình và dữ liệu
import os
from enum import Enum

from src.utils.utils import normalize_path


class Mode(Enum):
    VAST = 0
    FSOFT_SERVER = 1
    LINH_LOCAL = 2
    DA_LOCAL = 3

class COMMENT_REMOVAL(Enum):
    AST = 0
    REGREX = 1

# --------------------------------------------------------------------------------
# Môi trường & Tham số huấn luyện (MODIFY HERE)
# --------------------------------------------------------------------------------
mode = Mode.VAST  # <= --------------------------------- CHOOSE DEPLOYMENT HERE ---------------------------------
MODEL_NAME = "Salesforce/codet5-small"
max_source_length = 256
max_target_length = 256
EPOCHS = 3
BATCH_SIZE = 16
FP16 = False
TRAIN_TYPE = "lora"  # full hoặc lora

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

LOG_DIR = normalize_path(f"{OUTPUT_PATH}/training_history.json")
