## Cấu trúc thư mục

```
project-root/
│
├── config/
│   ├── __init__.py
│   └── config.py                    # Cấu hình tham số  
│
├── data/
│   ├── masking/
│   │   ├── __init__.py
│   │   └── RandomTokenMasker.py     # Masking token ngẫu nhiên
│   │
│   ├── __init__.py
│   └── preprocess.py                # Xử lý dữ liệu
│
├── predict/
│   ├── __init__.py
│   ├── analysis.py                  # Phân tích kết quả
│   └── evaluate.py                  # Đánh giá mô hình
│
├── train/
│   ├── codet5/
│   │   ├── __init__.py
│   │   ├── lora_trainer_codet5small.py
│   │   ├── lora_trainer_codet5base.py
│   │   └── lora_trainer_codet5large.py
│   │
│   ├── __init__.py
│   ├── base_trainer.py              # Lớp trainer cơ sở
│   ├── full_finetune.py             # Full fine-tuning
│   └── lora_trainer.py              # LoRA fine-tuning
│
└── utils/
    ├── __init__.py
    ├── model_utils.py               # Tiện ích model
    ├── mylogger.py                  # Logging
    ├── token_statistics.py          # Thống kê token
    └── utils.py                     # Tiện ích chung
```

## Cấu hình tham số trong config.py

### Enum Classes
- `Mode`: Môi trường chạy (VAST, FSOFT_SERVER, LINH_LOCAL, DA_LOCAL)
- `COMMENT_REMOVAL`: Phương thức xóa comment (AST, REGREX)
- `MASKING_STRATEGIES`: Chiến lược masking (NONE, RANDOM) 
- `TRAIN_MODES`: Phương pháp huấn luyện (FULL_FINETUNING, LORA)
- `MODEL_TYPES`: Loại mô hình (CODET5_SMALL, CODET5_BASE, CODET5_LARGE)

### Tham số huấn luyện chính
```python
mode = Mode.DA_LOCAL  # Môi trường chạy
TRAIN_TYPE = TRAIN_MODES.LORA  # Phương pháp huấn luyện

MODEL_TYPE = MODEL_TYPES.CODET5_SMALL  # Loại mô hình
MODEL_NAME = str(MODEL_TYPE.value)  # Tên mô hình

MASKING_SOURCE = MASKING_STRATEGIES.NONE  # Chiến lược masking

max_source_length = 32  # Độ dài tối đa của source
max_target_length = 32  # Độ dài tối đa của target
EPOCHS = 1  # Số epoch huấn luyện
BATCH_SIZE = 2  # Kích thước batch
LEARNING_RATE = 0.0005  # Tốc độ học
LOGGING_STEP = 50  # Tần suất log

TRAININGSET_REPORT_LIMIT = 10  # Số lượng mẫu đánh giá trên tập train
VALIDATIONSET_REPORT_LIMIT = 10  # Số lượng mẫu đánh giá trên tập validation

REMOVE_COMMENT_MODE = COMMENT_REMOVAL.REGREX  # Phương thức xóa comment
```