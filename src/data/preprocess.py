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
            target = clean_code(remove_comments_ast(entry.get("t", "")))

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