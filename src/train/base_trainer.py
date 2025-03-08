import logging

from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import datasets
from abc import ABC, abstractmethod
from src.utils.mylogger import logger
from src.config.config import MODEL_NAME, TRAINSET_DATA_PATH_PROCESS, MODEL_SAVE_PATH, VALIDATIONSET_DATA_PATH_PROCESS, MASKING_SOURCE, MASKING_STRATEGIES, max_target_length, \
    max_source_length
from src.data.masking.RandomTokenMasker import RandomTokenMasker


class BaseTrainer(ABC):
    """Lớp cơ sở cho việc huấn luyện mô hình."""

    def __init__(self):
        self.model, self.tokenizer = self.load_model()
        self.train_dataset, self.val_dataset = self.load_data()
        self.train_loss_history = []

        if MASKING_SOURCE == MASKING_STRATEGIES.NONE:
            logging.info("[UET] Masking: None")
            self.token_masker = RandomTokenMasker(mask_rate_min=0.10, mask_rate_max=0.15)
            self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)
        elif MASKING_SOURCE == MASKING_STRATEGIES.RANDOM:
            logging.info("[UET] Masking: Random")
            self.train_dataset = self.train_dataset.map(self.preprocess_with_masking, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess_with_masking, batched=True)

    @staticmethod
    @abstractmethod
    def load_model():
        pass

    @abstractmethod
    def train(self):
        """Phương thức train, sẽ được triển khai trong lớp con."""
        pass

    def load_data(self):
        """Load và xử lý dữ liệu."""
        dataset_trainset = datasets.load_dataset("json", data_files=TRAINSET_DATA_PATH_PROCESS, split="train")
        logger.info(f"[UET] Số lượng mẫu trong train set: {len(dataset_trainset)}")

        dataset_validationset = datasets.load_dataset("json", data_files=VALIDATIONSET_DATA_PATH_PROCESS, split="train")
        logger.info(f"[UET] Số lượng mẫu trong validation set: {len(dataset_validationset)}")

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
        inputs = self.tokenizer(sources, max_length=max_source_length, truncation=True, padding="max_length")
        outputs = self.tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

        inputs["labels"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in outputs["input_ids"]]
        return inputs

    def preprocess(self, examples):
        """Tiền xử lý dữ liệu."""
        inputs = self.tokenizer(examples["source"], max_length=max_source_length, truncation=True, padding="max_length")
        targets = self.tokenizer(examples["target"], max_length=max_target_length, truncation=True,
                                 padding="max_length")
        inputs["labels"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in targets["input_ids"]]
        return inputs

    def save_model(self):
        """Lưu mô hình đã huấn luyện."""
        self.model.save_pretrained(MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(MODEL_SAVE_PATH)
        logger.info("[UET] Model saved successfully!")

    def plot_loss(self, trainer):
        """Vẽ đồ thị training loss và validation loss theo epoch."""
        try:
            logger.info("[UET] plot_loss")
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
        except Exception as e:
            logger.error(f"[UET] Error in plot_loss: {e}")
            logger.exception("Stack trace:")
