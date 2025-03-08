import csv
import os

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config.config import VALIDATIONSET_DATA_PATH_PROCESS, MODEL_SAVE_PATH, OUTPUT_VALIDATIONSET_CSV, \
    max_source_length, \
    max_target_length, TRAIN_TYPE, TRAIN_MODES, MODEL_TYPES, MODEL_TYPE
from src.train.codet5.lora_trainer_codet5base import LoRATrainer_CodeT5Base
from src.train.codet5.lora_trainer_codet5large import LoRATrainer_CodeT5Large
from src.train.codet5.lora_trainer_codet5small import LoRATrainer_CodeT5Small
from src.train.full_finetune import FullFineTuneTrainer
from src.utils.mylogger import logger


# Load model đã train
def load_model(datapath):
    logger.info(f"[UET] Download model from %s - start", MODEL_SAVE_PATH)

    model = None
    if TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_SMALL:
        model = LoRATrainer_CodeT5Small.load_model()
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_BASE:
        model = LoRATrainer_CodeT5Base.load_model()
    elif TRAIN_TYPE == TRAIN_MODES.LORA and MODEL_TYPE == MODEL_TYPES.CODET5_LARGE:
        model = LoRATrainer_CodeT5Large.load_model()
    elif TRAIN_TYPE == TRAIN_MODES.FULL_FINETUNING:
        model = FullFineTuneTrainer.load_model()

    if model is None:
        logger.error(f"[UET] Chưa support model ở {datapath}")
        return

    logger.info(f"[UET] Download model from %s - done", MODEL_SAVE_PATH)

    logger.info(f"[UET] Load Tokenizer from %s", MODEL_SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

    logger.info(f"[UET] Moving model to GPU/CPU - start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"[UET] Moving model to GPU/CPU - done, using: {device}")

    logger.info(f"[UET] Run model.eval() - start")
    model.eval()
    logger.info(f"[UET] Run model.eval() - done")

    # Load raw test
    logger.info(f"[UET] Đang load dataset từ from %s - start", datapath)
    dataset = load_dataset("json", data_files=datapath, split="train")
    num_samples = len(dataset)
    logger.info(f"[UET] Số lượng mẫu trong dataset: {num_samples}")

    # Kiểm tra raw
    if len(dataset) == 0:
        raise ValueError("[UET] Dataset rỗng! Kiểm tra lại dữ liệu đầu vào.")

    return model, dataset, tokenizer


# Hàm sinh dự đoán từ mô hình
def generate_target_from_sample(sample, tokenizer, model):
    source_text = str(sample["source"]) + " <SEP>"
    inputs = tokenizer(source_text, return_tensors="pt", max_length=max_source_length, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(inputs=inputs["input_ids"], max_length=max_target_length)
    return source_text, tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split(),
                         smoothing_function=SmoothingFunction().method4)


def evaluate_model(dataset, tokenizer, model, outputFolder, limit=None):
    logger.info("[UET] Đang đánh giá mô hình...")
    logger.info(f"[UET] Số sample đánh giá: {limit}")
    # Xóa file cũ nếu có
    if os.path.exists(outputFolder):
        os.remove(outputFolder)

    em_scores = []
    bleu_scores = []

    with open(outputFolder, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Source", "Expected Target", "Predicted Target", "BLEU Score", "Exact Match"])
        count = 0

        # Tao progress bar
        total = len(dataset) if limit is None else min(limit, len(dataset))
        pbar = tqdm(total=total, desc="Evaluating", unit="sample")

        for i, sample in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            try:
                source_text, predicted = generate_target_from_sample(sample, tokenizer, model)
                ground_truth = str(sample["target"])

                em_score = int(predicted.strip() == ground_truth.strip())
                bleu_score = compute_bleu(predicted, ground_truth)

                writer.writerow([source_text, ground_truth, predicted, bleu_score, em_score])
                f.flush()

                em_scores.append(em_score)
                bleu_scores.append(bleu_score)

                count += 1
                pbar.update(1)
                pbar.set_postfix(EM=f"{100 * sum(em_scores) / len(em_scores):.2f}%",
                                 BLEU=f"{100 * sum(bleu_scores) / len(bleu_scores):.2f}%")

            except Exception as e:
                logger.error(f"Error processing sample at position {i}: {str(e)}")
                continue

        pbar.close()

    if em_scores:
        avg_em = sum(em_scores) / len(em_scores)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        logger.info(f"\n[UET] **Kết quả đánh giá:**")
        logger.info(f"[UET] Exact Match: {avg_em * 100:.2f}%")
        logger.info(f"[UET] BLEU Score trung bình: {avg_bleu * 100:.2f}%")
        logger.info(f"[UET] Kết quả được lưu vào: {outputFolder}")
    else:
        logger.warning("[UET] Không có mẫu nào được đánh giá thành công!")

# Chạy đánh giá
if __name__ == "__main__":
    model, dataset, tokenizer = load_model(datapath=VALIDATIONSET_DATA_PATH_PROCESS)
    evaluate_model(dataset, tokenizer, model, outputFolder=OUTPUT_VALIDATIONSET_CSV)
