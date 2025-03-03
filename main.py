import os
import sys

# Xác định đường dẫn thư mục gốc của dự án
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "log")

# Thêm `src` vào sys.path để có thể import module
sys.path.insert(0, SRC_DIR)

# Import các module sau khi đã thêm vào sys.path
from src.data.preprocess import preprocess_dataset
from src.train.full_finetune import FullFineTuneTrainer
from src.train.lora_trainer import LoRATrainer
from src.predict import evaluate_model

def main():
    """Hàm chính để chạy pipeline"""

    #  Tiền xử lý dữ liệu
    raw_data_path = os.path.join(DATA_DIR, "raw")
    processed_data_path = os.path.join(DATA_DIR, "processed.json")

    print(" Bắt đầu tiền xử lý dữ liệu...")
    preprocess_dataset(raw_data_path, processed_data_path)
    print(" Tiền xử lý hoàn tất!")

    #  Huấn luyện mô hình
    trainer_type = os.getenv("TRAIN_TYPE", "full")  # full hoặc lora

    print(f" Bắt đầu huấn luyện mô hình ({trainer_type})...")

    if trainer_type == "lora":
        trainer = LoRATrainer()
    else:
        trainer = FullFineTuneTrainer(processed_data_path)

    trainer.train()
    print(" Huấn luyện hoàn tất!")

    #  Đánh giá mô hình
    print(" Đánh giá mô hình...")
    evaluate_model()
    print(" Hoàn tất đánh giá!")


if __name__ == "__main__":
    main()
