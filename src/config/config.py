# Mô hình và dữ liệu
import os

MODEL_NAME="Salesforce/codet5-base"
DATA_PATH_RAW="D:\\Workspace\\AKA\\March04\\aka-llm\\data\\raw"

OUTPUT_PATH= "D:\\Workspace\\AKA\\March04\\aka-llm\\outLinh"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

DATA_PATH_PROCESS= OUTPUT_PATH + "\\processed.json"
MODEL_SAVE_PATH=OUTPUT_PATH+"\\model"
OUTPUT_CSV=OUTPUT_PATH + "\\output.csv"
LOG_DIR =OUTPUT_PATH + "\\logLinh"

# Tham số huấn luyện
EPOCHS=1
BATCH_SIZE=16
FP16=False
TRAIN_TYPE="full"  # full hoặc lora


