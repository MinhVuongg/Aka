import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import datasets
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv


# Load các biến môi trường từ file .env
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Salesforce/codet5-base")
DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../..", "data/processed.json")
MODEL_SAVE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../..", "model")
LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../..", "log")

DATA_PATH = os.getenv("DATA_PATH", "E:\AKA_AI\aka-llm\data\processed1.json")
class BaseTrainer(ABC):
    """Lớp cơ sở cho việc huấn luyện mô hình."""
    def __init__(self, data_path=None):
        self.model_name = MODEL_NAME
        self.data_path = data_path if data_path else DATA_PATH  # Nếu không truyền, dùng mặc định
        self.model_save_path = MODEL_SAVE_PATH
        self.log_dir = LOG_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.load_model()
        self.train_dataset, self.val_dataset = self.load_data()

    def load_model(self):
        """Tải mô hình gốc để fine-tune."""
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def load_data(self):
        """Load và xử lý dữ liệu."""
        dataset = datasets.load_dataset("json", data_files=self.data_path, split="train")
        train_val = dataset.train_test_split(test_size=0.1)
        return train_val["train"], train_val["test"]

    def preprocess(self, examples):
        """Tiền xử lý dữ liệu."""
        inputs = self.tokenizer(examples["source"], max_length=256, truncation=True, padding="max_length")
        targets = self.tokenizer(examples["target"], max_length=256, truncation=True, padding="max_length")
        inputs["labels"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in targets["input_ids"]]
        return inputs

    @abstractmethod
    def train(self):
        """Phương thức train, sẽ được triển khai trong lớp con."""
        pass

    def save_model(self):
        """Lưu mô hình đã huấn luyện."""
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        print(" Model saved successfully!")

    def plot_loss(self, trainer):
        """Vẽ đồ thị training loss và validation loss theo epoch."""
        log_history = trainer.state.log_history

        train_losses = []
        val_losses = []
        epochs = []

        for entry in log_history:
            if "loss" in entry:
                train_losses.append(entry["loss"])
                epochs.append(entry["epoch"])
            if "eval_loss" in entry:
                val_losses.append(entry["eval_loss"])

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Training Loss", marker="o")
        plt.plot(epochs, val_losses, label="Validation Loss", marker="s")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
