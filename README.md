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

# Cấu trúc JSON Training Sets

## 1. JSON Training Set Thô (AKA)

Dữ liệu thô được sinh ra từ AKA có cấu trúc chi tiết như sau:

```json
[
  {
    "f": [],                          // DANH SÁCH THUỘC TÍNH
    "fm": "",                         // HÀM ĐANG KIỂM THỬ
    
    "datatest": [
      {
        "id": 0,
        "dt": {},                     // DATA TREE
        "td": "",                     // TEST DRIVER TƯƠNG ỨNG (CẦN ĐẢM BẢO PARSE ĐƯỢC AST)
        "simplified_t": "...",        // TEST DRIVER RÚT GỌN
        "isAutomated": false,         // SINH TỰ ĐỘNG HAY THỦ CÔNG
        "testpath": [],               // TEST PATH
        "executed_fm": "",            // FM VỚI ĐÁNH DẤU /*EXECUTED*/ Ở CUỐI DÒNG ĐƯỢC THỰC THI
        "executed_fm_masked": "",     // FM NHƯNG CHỈ HIỂN THỊ CODE ĐƯỢC THỰC THI
        "executed_m": "",             // M VỚI ĐÁNH DẤU /*EXECUTED*/ Ở CUỐI DÒNG ĐƯỢC THỰC THI
        "executed_m_masked": ""       // M NHƯNG CHỈ HIỂN THỊ CODE ĐƯỢC THỰC THI
      }
    ],
    "m": {                           // DANH SÁCH HÀM LIÊN QUAN
      "called_m": [                  // CHỨA CÁC HÀM ĐƯỢC GỌI BỞI FM VÀ KHÔNG STUB
        {
          "path_fm": "",             // ĐỊA CHỈ
          "fm": ""                   // CODE (CẦN ĐẢM BẢO PARSE ĐƯỢC AST)
        }
      ],     
      "stub_called_m": [],           // CHỨA CÁC HÀM ĐƯỢC GỌI VÀ STUB (CẦN ĐẢM BẢO PARSE ĐƯỢC AST)
      "callee_m": []                 // CHỨA CÁC HÀM GỌI ĐẾN FM (CẦN ĐẢM BẢO PARSE ĐƯỢC AST)
    },        
    "fc": "",                        // CLASS ĐƯỢC KIỂM THỬ
    "c": [],                         // DANH SÁCH CONSTRUCTOR (CẦN ĐẢM BẢO PARSE ĐƯỢC AST)
    "path_fm": ""                    // ĐƯỜNG DẪN ĐẾN HÀM KIỂM THỬ
  }
]
```

### Giải thích chi tiết
- **f**: Mảng chứa danh sách các thuộc tính
- **fm**: Hàm đang được kiểm thử
- **datatest**: Mảng chứa các test case
  - **id**: ID của test case
  - **dt**: Data tree
  - **td**: Test driver
  - **simplified_t**: Test driver đã được rút gọn
  - **executed_fm/executed_m**: Code với annotation /*EXECUTED*/ ở cuối dòng đã thực thi
  - **executed_fm_masked/executed_m_masked**: Chỉ hiển thị code đã thực thi
- **m**: Các phương thức liên quan
  - **called_m**: Hàm được gọi bởi FM và không stub
  - **stub_called_m**: Hàm được gọi và stub
  - **callee_m**: Hàm gọi đến FM
- **fc**: Class được kiểm thử
- **c**: Danh sách constructor
- **path_fm**: Đường dẫn đến hàm kiểm thử

## 2. JSON Training Set Mịn

Dữ liệu mịn được sinh ra từ training set thô, có cấu trúc đơn giản hơn:

```json
[
  {
    "source": "...",
    "target": "..."
  }
]
```

### Giải thích
- **source**: Dữ liệu nguồn dùng làm input cho model
- **target**: Dữ liệu đích dùng làm output mong muốn của model

Dataset mịn được tối ưu hóa để huấn luyện các mô hình Machine Learning, trong khi dataset thô chứa thông tin chi tiết hơn về cấu trúc code và quá trình kiểm thử.