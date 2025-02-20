import os
import glob
import json
from clang.cindex import Index, TokenKind
import re
from transformers import AutoTokenizer
from src.train.base_trainer import MODEL_NAME
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Load tokenizer của mô hình đang học
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_target_range(target):
    """
    Trích xuất phần code từ target, bỏ các dòng không cần thiết.
    """
    # Chia target thành từng dòng bằng dấu `;` hoặc xuống dòng (hỗ trợ PyCharm)
    lines = re.split(r'[;\n]+', target)
    code_lines = []
    capture = False

    for line in lines:
        line = line.strip()  # Loại bỏ khoảng trắng thừa

        # Bỏ qua các dòng chứa từ khóa không cần thiết
        if any(keyword in line for keyword in ["AKA_mark", "AKA_EXPECTED_OUTPUT", "AKA_fCall"]):
            continue

        # Nếu gặp 'AKA_test_case_name', bắt đầu ghi dữ liệu
        if "AKA_test_case_name" in line:
            capture = True
            continue

        # Nếu gặp 'AKA_ACTUAL_OUTPUT', dừng lại
        if "AKA_ACTUAL_OUTPUT" in line:
            break

        if capture:
            code_lines.append(line)

    # Nếu không có dòng hợp lệ, trả về target ban đầu
    if not code_lines:
        return target.strip()

    # Gộp lại thành chuỗi, đảm bảo không còn dấu `;` thừa
    return " ".join(code_lines).strip("; ")


# Loại bỏ comment trong code
def remove_comments_ast(code):
    if isinstance(code, list):  # Nếu là danh sách, chuyển thành chuỗi
        code = " ".join(map(str, code))

    index = Index.create()
    tu = index.parse('temp.cpp', unsaved_files=[('temp.cpp', code)], args=['-x', 'c++'])

    comments = [(token.extent.start.offset, token.extent.end.offset)
                for token in tu.cursor.get_tokens() if token.kind == TokenKind.COMMENT]

    if not comments:
        return code

    result = "".join(
        [code[last:end] for last, end in zip([0] + [c[1] for c in comments], [c[0] for c in comments] + [len(code)])]
    )
    return result


# Xoá những kí tự dư thừa như xuống dòng, nhiều dấu cách thừa ...
def clean_code(code):
    code = re.sub(r'[\r\n\t]+', ' ', code)
    code = re.sub(r'\s+', ' ', code).strip()
    return code


# Load và xử lý dữ liệu
def preprocess_dataset(input_folder, output_file):
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    all_data = []

    # Load dữ liệu cũ nếu tệp đã tồn tại
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = []

    existing_entries = {(entry["source"], entry["target"]) for entry in all_data}
    new_data = []

    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for entry in raw_data:
            source = clean_code(remove_comments_ast(entry.get("fm", "")))
            target = clean_code(remove_comments_ast(extract_target_range(entry.get("t", ""))))

            if (source, target) not in existing_entries:
                new_data.append({"source": source, "target": target})
                existing_entries.add((source, target))

    if new_data:
        all_data.extend(new_data)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"Added {len(new_data)} new samples. Total samples: {len(all_data)}")
    else:
        print("No new data to add.")


if __name__ == "__main__":
    preprocess_dataset("data/raw", "data/processed.json")