import os
import torch
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Cấu hình đường dẫn
MODEL_PATH = "D:\Code\Aka\codet5"  # Thay bằng nơi lưu mô hình
DATA_PATH = "data/processed.json"
OUTPUT_CSV = "output/evaluation_results.csv"

# Kiểm tra file dữ liệu
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f" Không tìm thấy file dữ liệu: {DATA_PATH}")

# Load model đã train
print("Đang tải mô hình...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Load dataset test
dataset = load_dataset("json", data_files=DATA_PATH, split="train").train_test_split(test_size=0.1)["test"]

# Kiểm tra dataset
if len(dataset) == 0:
    raise ValueError(" Dataset rỗng! Kiểm tra lại dữ liệu đầu vào.")

# Tham số đánh giá
max_source_length = 256
max_target_length = 256


# Hàm sinh dự đoán từ mô hình
def generate_target_from_sample(sample):
    source_text = str(sample["source"]) + " <SEP>"
    inputs = tokenizer(source_text, return_tensors="pt", max_length=max_source_length, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(inputs=inputs["input_ids"], max_length=max_target_length)

    return source_text, tokenizer.decode(outputs[0], skip_special_tokens=True)


# Hàm tính BLEU Score
smoothie = SmoothingFunction().method4


def compute_bleu(generated, ground_truth):
    return sentence_bleu([ground_truth.split()], generated.split(), smoothing_function=smoothie)


# Hàm đánh giá mô hình
def evaluate_model():
    print(" Đang đánh giá mô hình...")

    # Xóa file cũ nếu có
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    em_scores = []
    bleu_scores = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Source", "Expected Target", "Predicted Target", "BLEU Score", "Exact Match"])

        pbar = tqdm(dataset, desc="Evaluating", unit="sample")
        for sample in pbar:
            source_text, predicted = generate_target_from_sample(sample)
            ground_truth = str(sample["target"])

            em_score = int(predicted.strip() == ground_truth.strip())  # Exact Match (0 hoặc 1)
            bleu_score = compute_bleu(predicted, ground_truth)

            writer.writerow([source_text, ground_truth, predicted, bleu_score, em_score])
            f.flush()

            em_scores.append(em_score)
            bleu_scores.append(bleu_score)

            pbar.set_postfix(EM=f"{100 * sum(em_scores) / len(em_scores):.2f}%",
                             BLEU=f"{100 * sum(bleu_scores) / len(bleu_scores):.2f}%")

    avg_em = sum(em_scores) / len(em_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print(f"\n **Kết quả đánh giá:**")
    print(f" Exact Match: {avg_em * 100:.2f}%")
    print(f" BLEU Score trung bình: {avg_bleu * 100:.2f}%")
    print(f" Kết quả được lưu vào: {OUTPUT_CSV}")


# Chạy đánh giá
if __name__ == "__main__":
    evaluate_model()
