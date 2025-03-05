# Mô hình và dữ liệu
import os

# Tham số đánh giá
max_source_length = 256
max_target_length = 256

# ------
VAST = False
if VAST:
    MODEL_NAME = "Salesforce/codet5-large"
    DATA_PATH_RAW = "/workspace/aka-llm/data/raw"

    OUTPUT_PATH = "/workspace/aka-llm/outLinh"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    DATA_PATH_PROCESS = OUTPUT_PATH + "/processed.json"
    MODEL_SAVE_PATH = OUTPUT_PATH + "/model"
    OUTPUT_CSV = OUTPUT_PATH + "/output.csv"
    LOG_DIR = OUTPUT_PATH + "/training_history.json"

    # Tham số huấn luyện
    EPOCHS = 50
    BATCH_SIZE = 512
    FP16 = False
    TRAIN_TYPE = "lora"  # full hoặc lora

    TEST_SIZE = 0.2
else:
    MODEL_NAME="Salesforce/codet5-small"
    DATA_PATH_RAW="D:\\Workspace\\AKA\\March04\\aka-llm\\data\\raw1"

    OUTPUT_PATH= "D:\\Workspace\\AKA\\March04\\aka-llm\\outLinh"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    DATA_PATH_PROCESS= OUTPUT_PATH + "\\processed.json"
    MODEL_SAVE_PATH=OUTPUT_PATH+"\\model"
    OUTPUT_CSV=OUTPUT_PATH + "\\output.csv"
    LOG_DIR = OUTPUT_PATH + "\\training_history.json"

    # Tham số huấn luyện
    EPOCHS=3
    BATCH_SIZE=16
    FP16=False
    TRAIN_TYPE="lora"  # full hoặc lora

    TEST_SIZE = 0.2
