import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import datasets
import os
from abc import ABC, abstractmethod
# from dotenv import load_dotenv

from src.config.config import MODEL_NAME, TRAINSET_DATA_PATH_PROCESS, MODEL_SAVE_PATH, LOG_DIR, \
    VALIDATIONSET_DATA_PATH_PROCESS, ModelType, MODEL_TYPE, MASKING_SOURCE, MASKING
from src.data.RandomTokenMasker import RandomTokenMasker
from src.utils.model_utils import load_model_by_type


class BaseTrainer(ABC):
    """Lớp cơ sở cho việc huấn luyện mô hình."""

    def __init__(self, data_path=None):
        self.model_name = MODEL_NAME
        self.model_save_path = MODEL_SAVE_PATH
        self.log_dir = LOG_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.load_model()
        self.train_dataset, self.val_dataset = self.load_data()
        self.train_loss_history = []
        self.token_masker = RandomTokenMasker(mask_rate_min=0.10, mask_rate_max=0.15)

        if MASKING_SOURCE == MASKING.NONE:
            logging.info("[UET] Masking")
            self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)
        elif MASKING_SOURCE == MASKING.RANDOM:
            self.train_dataset = self.train_dataset.map(self.preprocess_with_masking, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess_with_masking, batched=True)

    def load_model(self):
        """Tải mô hình gốc để fine-tune với khả năng tùy chỉnh loại mô hình sử dụng ENUM."""
        return load_model_by_type(MODEL_NAME, MODEL_TYPE)

    # def load_model(self):
    #     """Tải mô hình gốc để fine-tune."""
    #     return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def load_data(self):
        """Load và xử lý dữ liệu."""
        dataset_trainset = datasets.load_dataset("json", data_files=TRAINSET_DATA_PATH_PROCESS, split="train")
        print(f"[UET] Số lượng mẫu trong train set: {len(dataset_trainset)}")

        dataset_validationset = datasets.load_dataset("json", data_files=VALIDATIONSET_DATA_PATH_PROCESS, split="train")
        print(f"[UET] Số lượng mẫu trong validation set: {len(dataset_validationset)}")

        return dataset_trainset, dataset_validationset

    def preprocess_with_masking(self, examples):
        """Preprocess data with random token masking"""
        sources = []
        targets = []

        # Apply random token masking to each example
        for source, target in zip(examples.get("source", []), examples.get("target", [])):
            # Mask tokens in the source code randomly
            masked_source = self.token_masker.mask_tokens(str(source))

            sources.append(masked_source)
            targets.append(str(target))

            # Reset the masker for the next example
            self.token_masker.reset()

        # Tokenize the masked inputs
        inputs = self.tokenizer(sources, max_length=256, truncation=True, padding="max_length")
        outputs = self.tokenizer(targets, max_length=256, truncation=True, padding="max_length")

        inputs["labels"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in outputs["input_ids"]]
        return inputs

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
