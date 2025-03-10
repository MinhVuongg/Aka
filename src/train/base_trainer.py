import logging
from abc import ABC, abstractmethod

import datasets
import matplotlib.pyplot as plt

from src.config.config import TRAINSET_DATA_PATH_PROCESS, MODEL_SAVE_PATH, VALIDATIONSET_DATA_PATH_PROCESS, \
    MASKING_SOURCE, MASKING_STRATEGIES, max_target_length, \
    max_source_length
from src.data.masking.RandomTokenMasker import RandomTokenMasker
from src.utils.mylogger import logger


class BaseTrainer(ABC):
    """Lớp cơ sở cho việc huấn luyện mô hình."""

    def __init__(self):
        self.model, self.tokenizer = self.load_model()
        self.train_dataset, self.val_dataset = self.load_data()
        self.train_loss_history = []
        self.val_loss_history = []

        if MASKING_SOURCE == MASKING_STRATEGIES.NONE:
            logging.info("[UET] Masking: None")
            self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)

        elif MASKING_SOURCE == MASKING_STRATEGIES.RANDOM:
            logging.info("[UET] Masking: Random")
            self.token_masker = RandomTokenMasker(mask_rate_min=0.10, mask_rate_max=0.15)
            self.train_dataset = self.train_dataset.map(self.preprocess_with_random_masking, batched=True)
            self.val_dataset = self.val_dataset.map(self.preprocess_with_random_masking, batched=True)

    @staticmethod
    @abstractmethod
    def load_model():
        pass

    @abstractmethod
    def train(self):
        """Phương thức train, sẽ được triển khai trong lớp con."""
        pass

    def remove_long_samples_from_dataset(self, dataset, max_source_length, max_target_length):
        """
        Bỏ các phần tử source >= max_source_length và target >= max_target_length

        Args:
            dataset: Dataset cần lọc

        Returns:
            filtered_dataset: Dataset sau khi lọc
        """
        initial_size = len(dataset)
        logger.info(f"[UET] Số lượng mẫu ban đầu: {initial_size}")

        def filter_short_samples(example):
            return len(str(example["source"])) <= max_source_length and len(str(example["target"])) <= max_target_length

        filtered_dataset = dataset.filter(filter_short_samples)

        filtered_size = len(filtered_dataset)
        logger.info(f"[UET] Số lượng mẫu sau khi lọc: {filtered_size}")

        if initial_size > 0:
            removal_rate = (initial_size - filtered_size) / initial_size * 100
            logger.info(f"[UET] Tỷ lệ loại bỏ: {removal_rate:.2f}%")

        return filtered_dataset

    # def load_data(self):
    #     """Load và xử lý dữ liệu."""
    #     dataset_trainset = datasets.load_dataset("json", data_files=TRAINSET_DATA_PATH_PROCESS, split="train")
    #     logger.info(f"[UET] Số lượng mẫu trong train set: {len(dataset_trainset)}")
    #
    #     dataset_validationset = datasets.load_dataset("json", data_files=VALIDATIONSET_DATA_PATH_PROCESS, split="train")
    #     logger.info(f"[UET] Số lượng mẫu trong validation set: {len(dataset_validationset)}")
    #
    #     return dataset_trainset, dataset_validationset

    def load_data(self):
        """Load và xử lý dữ liệu."""

        # Load datasets
        dataset_trainset = datasets.load_dataset("json", data_files=TRAINSET_DATA_PATH_PROCESS, split="train")
        logger.info(f"[UET] Số lượng mẫu trong train set: {len(dataset_trainset)}")
        dataset_validationset = datasets.load_dataset("json", data_files=VALIDATIONSET_DATA_PATH_PROCESS, split="train")
        logger.info(f"[UET] Số lượng mẫu trong validation set: {len(dataset_validationset)}")

        # if OPTIMIZE_TRAININGSET_STRATEGY == TARGET_SELETCTION_STRATEGIES.SORT_BY_TOKEN_AND_CUTOFF:
        #     logger.info(f"[UET] Đang lọc train set...")
        #     filtered_trainset = self.remove_long_samples_from_dataset(dataset_trainset, max_source_length, max_target_length)
        #     logger.info(f"[UET] Số lượng mẫu trong train set: {len(dataset_trainset)}")
        #
        #     logger.info(f"[UET] Đang lọc validation set...")
        #     filtered_validationset = self.remove_long_samples_from_dataset(dataset_validationset, max_source_length, max_target_length)
        #     logger.info(f"[UET] Số lượng mẫu trong validation set: {len(dataset_validationset)}")

        return dataset_trainset, dataset_validationset

    def preprocess_with_random_masking(self, examples):
        """Preprocess data with random token masking"""
        sources = []
        targets = []

        # Apply random token masking to each example
        i = 0
        for source, target in zip(examples.get("source", []), examples.get("target", [])):
            i += 1
            # Mask tokens in the source code randomly
            masked_source = self.token_masker.mask_tokens(str(source))

            sources.append(masked_source)
            targets.append(str(target))

            # Reset the masker for the next example
            self.token_masker.reset()

            if i == 1:
                logger.info(f"[UET] Before masking:"
                            f"\n------------------------------------\n"
                            f"{source} \n "
                            f"\n------------------------------------\n"
                            f"After masking: "
                            f"\n{masked_source}"
                            f"\n------------------------------------\n")


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
