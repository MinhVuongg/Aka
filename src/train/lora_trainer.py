from src.config.config import BATCH_SIZE, EPOCHS, FP16
from src.data.RandomTokenMasker import RandomTokenMasker
from src.train.base_trainer import BaseTrainer
import peft
from transformers import Trainer, TrainingArguments

class LoRATrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.add_lora()
        self.token_masker = RandomTokenMasker(mask_rate_min=0.10, mask_rate_max=0.15)

    def add_lora(self):
        """Thêm LoRA vào mô hình"""
        lora_config = peft.LoraConfig(
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"
        )
        self.model = peft.get_peft_model(self.model, lora_config)

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

    def train(self):
        """Huấn luyện LoRA."""
        self.train_dataset = self.train_dataset.map(self.preprocess_with_masking, batched=True)
        self.val_dataset = self.val_dataset.map(self.preprocess_with_masking, batched=True)

        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            num_train_epochs=EPOCHS,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=50,
            logging_strategy="epoch",
            fp16=FP16,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.train_loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        self.val_loss_history = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
        self.save_model()
        self.plot_loss(trainer)

