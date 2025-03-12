"""
Đếm token trong file JSON dùng để train model.
Đếm source và target
"""
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
    except Exception as e:
        logger.info(f"Lỗi khi đọc file JSON: {str(e)}")
        return

    logger.info(f"Tổng số mục trong file: {len(data)}")

    # Khởi tạo danh sách để lưu token
    all_source_tokens = []
    all_target_tokens = []

    # Phân tích từng mục
    for i, item in enumerate(data):

        # Đếm token cho source
        source = item.get("source", "")
        source_tokens = len(tokenizer.encode(source))
        all_source_tokens.append(source_tokens)

        # Đếm token cho target
        target = item.get("target", "")
        target_tokens = len(tokenizer.encode(target))
        all_target_tokens.append(target_tokens)

    # Chuyển sang numpy array để tính toán thống kê
    # Chuyển sang numpy array để tính toán thống kê
    source_tokens_np = np.array(all_source_tokens)
    target_tokens_np = np.array(all_target_tokens)

    source_stats = {}
    target_stats = {}

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
        logging.warning("⚠️ Không có token nào trong dữ liệu source!")

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
        logging.warning("⚠️ Không có token nào trong dữ liệu target!")


    # Tính các chỉ số thống kê cơ bản
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

    # In kết quả tổng hợp
    logging.info("===== THỐNG KÊ TOKEN =====")

    logging.info("--- SOURCE TOKENS ---")
    logging.info(f"Min: {source_stats['min']}")
    logging.info(f"Max: {source_stats['max']}")
    logging.info(f"Avg: {source_stats['avg']:.2f}")
    logging.info(f"Q25: {source_stats['q25']}")
    logging.info(f"Q50 (trung vị): {source_stats['q50']}")
    logging.info(f"Q75: {source_stats['q75']}")
    logging.info(f"Q90: {source_stats['q90']}")
    logging.info(f"Q95: {source_stats['q95']}")
    logging.info(f"Q99: {source_stats['q99']}")

    logging.info("--- TARGET TOKENS ---")
    logging.info(f"Min: {target_stats['min']}")
    logging.info(f"Max: {target_stats['max']}")
    logging.info(f"Avg: {target_stats['avg']:.2f}")
    logging.info(f"Q25: {target_stats['q25']}")
    logging.info(f"Q50 (trung vị): {target_stats['q50']}")
    logging.info(f"Q75: {target_stats['q75']}")
    logging.info(f"Q90: {target_stats['q90']}")
    logging.info(f"Q95: {target_stats['q95']}")
    logging.info(f"Q99: {target_stats['q99']}")

    logging.info("===========================")

    return {
        "source_stats": source_stats,
        "target_stats": target_stats
    }

if __name__ == '__main__':
    model, tokenizer = load_model_by_type(TRAIN_TYPE, MODEL_TYPE)
    token_stats = count_tokens(
        "/Users/ducanhnguyen/Documents/aka-llm/aka-output/2025-03-08-13-31/processed_trainset.json",
        tokenizer)