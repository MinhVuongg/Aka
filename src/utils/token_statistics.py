import json
import logging
import numpy as np
from src.config.config import MODEL_TYPE, TRAIN_TYPE
from src.utils.model_utils import load_model_by_type
from src.utils.mylogger import logger

def count_tokens(json_path, tokenizer):
    # Đọc file JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ Đọc thành công file JSON, tổng số phần tử: {len(data)}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc file JSON: {str(e)}")
        return None

    if not data:
        logger.warning("⚠️ File JSON không có dữ liệu! Bỏ qua xử lý.")
        return None

    # Khởi tạo danh sách để lưu token
    all_source_tokens = []
    all_target_tokens = []

    # Phân tích từng mục
    for i, item in enumerate(data):
        source = item.get("source", "").strip()
        target = item.get("target", "").strip()
        
        if source:
            source_tokens = len(tokenizer.encode(source))
            all_source_tokens.append(source_tokens)
        else:
            logger.warning(f"⚠️ Dữ liệu source tại index {i} rỗng!")
        
        if target:
            target_tokens = len(tokenizer.encode(target))
            all_target_tokens.append(target_tokens)
        else:
            logger.warning(f"⚠️ Dữ liệu target tại index {i} rỗng!")
    
    # Chuyển sang numpy array để tính toán thống kê
    source_tokens_np = np.array(all_source_tokens)
    target_tokens_np = np.array(all_target_tokens)

    source_stats, target_stats = None, None
    
    if source_tokens_np.size > 0:
        source_stats = {
            "min": np.min(source_tokens_np),
            "max": np.max(source_tokens_np),
            "avg": np.mean(source_tokens_np),
            "q25": np.percentile(source_tokens_np, 25),
            "q50": np.percentile(source_tokens_np, 50),
            "q75": np.percentile(source_tokens_np, 75),
            "q90": np.percentile(source_tokens_np, 90),
            "q95": np.percentile(source_tokens_np, 95),
            "q99": np.percentile(source_tokens_np, 99)
        }
    else:
        logger.warning("⚠️ Không có token nào trong source! Bỏ qua thống kê.")

    if target_tokens_np.size > 0:
        target_stats = {
            "min": np.min(target_tokens_np),
            "max": np.max(target_tokens_np),
            "avg": np.mean(target_tokens_np),
            "q25": np.percentile(target_tokens_np, 25),
            "q50": np.percentile(target_tokens_np, 50),
            "q75": np.percentile(target_tokens_np, 75),
            "q90": np.percentile(target_tokens_np, 90),
            "q95": np.percentile(target_tokens_np, 95),
            "q99": np.percentile(target_tokens_np, 99)
        }
    else:
        logger.warning("⚠️ Không có token nào trong target! Bỏ qua thống kê.")

    return {"source_stats": source_stats, "target_stats": target_stats}

if __name__ == '__main__':
    model, tokenizer = load_model_by_type(TRAIN_TYPE, MODEL_TYPE)
    token_stats = count_tokens("/Users/ducanhnguyen/Documents/aka-llm/aka-output/2025-03-08-13-31/processed_trainset.json", tokenizer)