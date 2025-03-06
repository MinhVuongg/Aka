import logging
import os

import torch
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from src.config.config import VALIDATIONSET_DATA_PATH_PROCESS, MODEL_SAVE_PATH, OUTPUT_VALIDATIONSET_CSV, \
    max_source_length, \
    max_target_length, MODEL_NAME, MODEL_TYPE
from src.utils import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Load model đã train
def load_model():
    logger.info(f"[UET] Download model from %s - start", MODEL_SAVE_PATH)
    model = utils.load_model(MODEL_NAME, MODEL_TYPE)
    logger.info(f"[UET] Download model from %s - done", MODEL_SAVE_PATH)

    logger.info(f"[UET] Load Tokenizer from %s- start", MODEL_SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    logger.info(f"[UET] Load Tokenizer from %s- done", MODEL_SAVE_PATH)

    logger.info(f"[UET] Moving model to GPU/CPU - start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"[UET] Moving model to GPU/CPU - done, using: {device}")

    if torch.cuda.is_available():
        logger.info(f"[UET] Converting model to half precision (float16) - start")
        model = model.half()
        logger.info(f"[UET] Converting model to half precision (float16) - done")

    logger.info(f"[UET] Run model.eval() - start")
    model.eval()
    logger.info(f"[UET] Run model.eval() - done")

    # Load raw test
    logger.info(f"[UET] load_dataset from %s - start", VALIDATIONSET_DATA_PATH_PROCESS)
    dataset = load_dataset("json", data_files=VALIDATIONSET_DATA_PATH_PROCESS, split="train")
    num_samples = len(dataset)
    print(f"Số lượng mẫu trong dataset: {num_samples}")
    logger.info(f"[UET] load_dataset - done")

    # Kiểm tra raw
    if len(dataset) == 0:
        raise ValueError("[UET] Dataset rỗng! Kiểm tra lại dữ liệu đầu vào.")

    return model, dataset, tokenizer


# Hàm sinh dự đoán từ mô hình
def generate_target_from_sample(sample, tokenizer, model):
    source_text = str(sample["source"]) + " <SEP>"
    inputs = tokenizer(source_text, return_tensors="pt", max_length=max_source_length, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # print("\t with torch.no_grad(): - before")
    with torch.no_grad():
        outputs = model.generate(inputs=inputs["input_ids"], max_length=max_target_length)
    # print("\t with torch.no_grad(): - after")
    return source_text, tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split(),
                         smoothing_function=SmoothingFunction().method4)


# Hàm đánh giá mô hình
def evaluate_model(dataset, tokenizer, model):
    logger.info("[UET] Đang đánh giá mô hình...")

    # Xóa file cũ nếu có
    if os.path.exists(OUTPUT_VALIDATIONSET_CSV):
        os.remove(OUTPUT_VALIDATIONSET_CSV)

    em_scores = []
    bleu_scores = []

    with open(OUTPUT_VALIDATIONSET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Source", "Expected Target", "Predicted Target", "BLEU Score", "Exact Match"])

        pbar = tqdm(dataset, desc="Evaluating", unit="sample")
        for sample in pbar:
            # print("generate_target_from_sample - before");
            source_text, predicted = generate_target_from_sample(sample, tokenizer, model)
            # print("generate_target_from_sample - after");
            ground_truth = str(sample["target"])

            em_score = int(predicted.strip() == ground_truth.strip())  # Exact Match (0 hoặc 1)
            # print("compute_bleu - before");
            bleu_score = compute_bleu(predicted, ground_truth)
            # print("compute_bleu - before");

            writer.writerow([source_text, ground_truth, predicted, bleu_score, em_score])
            f.flush()

            em_scores.append(em_score)
            bleu_scores.append(bleu_score)

            pbar.set_postfix(EM=f"{100 * sum(em_scores) / len(em_scores):.2f}%",
                             BLEU=f"{100 * sum(bleu_scores) / len(bleu_scores):.2f}%")

    avg_em = sum(em_scores) / len(em_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    logger.info(f"\n[UET] **Kết quả đánh giá:**")
    logger.info(f"[UET] Exact Match: {avg_em * 100:.2f}%")
    logger.info(f"[UET] BLEU Score trung bình: {avg_bleu * 100:.2f}%")
    logger.info(f"[UET] Kết quả được lưu vào: {OUTPUT_VALIDATIONSET_CSV}")


# Chạy đánh giá
if __name__ == "__main__":
    model, dataset, tokenizer = load_model()
    evaluate_model(dataset, tokenizer, model)
