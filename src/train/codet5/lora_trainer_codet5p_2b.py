import peft
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from src.config.config import MODEL_NAME
from src.train.lora_trainer import LoRATrainer
import transformers
transformers.utils.logging.enable_explicit_format()
transformers.trainer.USE_TF = False 

class LoRATrainer_CodeT5P_2B(LoRATrainer):
    def __init__(self, model_name):
        """Khởi tạo mô hình với LoRA"""
        super().__init__(model_name)
        self.add_lora()  # Thêm LoRA vào mô hình

    @staticmethod
    def load_model(model_name):
        """Tải mô hình CodeT5p và tokenizer."""
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",  # Tự động nhận diện thiết bị
                trust_remote_code=True,  # Bắt buộc cho CodeT5p
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, safe_serialization=True)
            print(f" Đã tải mô hình {model_name} thành công.")
            return model, tokenizer
        except Exception as e:
            print(f" Lỗi khi tải mô hình: {e}")
            raise

    def add_lora(self, r=16, lora_alpha=32, lora_dropout=0.1):
        """Thêm LoRA vào CodeT5p-2B."""
        target_modules = self.get_lora_target_modules()

        lora_config = peft.LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",  
            task_type="SEQ_2_SEQ_LM"
        )

        self.model = peft.get_peft_model(self.model, lora_config)
        print(" LoRA đã được thêm vào mô hình CodeT5p.")

        self.print_trainable_parameters()

    def get_lora_target_modules(self):
        """Xác định các module cần áp dụng LoRA."""
        target_modules = []

        # Encoder layers
        num_encoder_layers = 20  
        for i in range(num_encoder_layers):
            target_modules.extend([
                f"encoder.h.{i}.attn.qkv_proj",
                f"encoder.h.{i}.attn.out_proj"
            ])

        # Decoder layers (Self-Attention)
        num_decoder_layers = 32  
        for i in range(num_decoder_layers):
            target_modules.extend([
                f"decoder.transformer.h.{i}.attn.qkv_proj",
                f"decoder.transformer.h.{i}.attn.out_proj"
            ])

        # Cross-Attention (Chỉ có ở lớp 31, cần xác nhận)
        target_modules.extend([
            f"decoder.transformer.h.31.crossattention.qkv_proj",
            f"decoder.transformer.h.31.crossattention.out_proj"
        ])

        print(f" Target Modules: {target_modules}")
        return target_modules

    def print_trainable_parameters(self):
        """In số lượng tham số có thể huấn luyện."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())

        print(f"🔹 Trainable params: {trainable_params} || Total params: {all_params} || Ratio: {100 * trainable_params / all_params:.2f}%")

    def train(self):
        """Huấn luyện mô hình với LoRA"""
        # Gọi phương thức huấn luyện từ Trainer
        super().train()

        # Đường dẫn lưu mô hình sau khi train
        save_path = "/workspace/aka-llm/aka-output/2025-03-11-18-39/model_checkpoint/"
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(save_path, exist_ok=True)

        # Lưu mô hình và tokenizer
        self.model.save_pretrained(save_path, safe_serialization=True)
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id 
        self.tokenizer.save_pretrained(save_path)

        print(f"Mô hình đã được lưu tại: {save_path}")
