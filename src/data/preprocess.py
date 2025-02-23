import os
import glob
import json
import re
import sys
from clang.cindex import Index, TokenKind, Config
from transformers import AutoTokenizer
from src.train.base_trainer import MODEL_NAME

# Đảm bảo module có thể tìm thấy đúng thư mục
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Config clang path
Config.set_library_path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "clang+llvm-19.1.7-x86_64-pc-windows-msvc/bin")))

# Load tokenizer của mô hình
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_target_range(target):
    """
    Trích xuất đoạn mã cần thiết từ target, bỏ qua các dòng không liên quan.
    """
    try:
        if isinstance(target, list):
            target = " ".join(map(str, target))

        lines = re.split(r'[;\n]+', target)
        code_lines = []
        capture = False

        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ["set up", "AKA_mark", "AKA_EXPECTED_OUTPUT", "AKA_fCall"]):
                continue
            if "AKA_test_case_name" in line:
                capture = True
                continue
            if "AKA_ACTUAL_OUTPUT" in line:
                code_lines.append(line)
                break
            if capture:
                code_lines.append(line)

        return ";".join(code_lines).strip(";") if code_lines else target.strip()
    except Exception as e:
        print(f"[ERROR] extract_target_range: {e}")
        return target


def remove_comments_ast(code):
    """
    Loại bỏ comment trong mã C++ bằng Clang AST.
    """
    try:
        if not code:
            return ""

        if isinstance(code, list):
            code = " ".join(map(str, code))

        index = Index.create()
        tu = index.parse('temp.cpp', unsaved_files=[('temp.cpp', code)], args=['-x', 'c++'])
        comments = [(token.extent.start.offset, token.extent.end.offset) for token in tu.cursor.get_tokens() if
                    token.kind == TokenKind.COMMENT]

        if not comments:
            return code

        result = []
        last_end = 0
        for start, end in comments:
            result.append(code[last_end:start])
            last_end = end
        result.append(code[last_end:])

        return re.sub(r';+', '; ', "".join(result).strip(";")) + ";"
    except Exception as e:
        print(f"[ERROR] remove_comments_ast: {e}")
        return code


def clean_code(code):
    """
    Chuẩn hóa mã nguồn bằng cách loại bỏ ký tự không cần thiết.
    """
    try:
        if not code:
            return ""

        code = re.sub(r'[\r\n\t]+', ' ', code)
        code = re.sub(r'\s+', ' ', code).strip()
        return code
    except Exception as e:
        print(f"[ERROR] clean_code: {e}")
        return code


def preprocess_dataset(input_folder, output_file, overwrite=False):
    """
    Xử lý tập dữ liệu từ thư mục đầu vào và lưu kết quả vào file JSON đầu ra.
    """
    try:
        json_files = glob.glob(os.path.join("../..", input_folder, "*.json"))
        new_data = []

        # Kiểm tra nếu file đầu ra đã tồn tại
        all_data = []
        if not overwrite and os.path.exists(os.path.join("../..", output_file)):
            with open(os.path.join("../..", output_file), "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSONDecodeError in {output_file}: {e}")
                    print("Warning: Output file contains invalid JSON, starting fresh.")

        existing_entries = {(entry["source"], entry["target"]) for entry in all_data}

        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSONDecodeError in {file}: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] Failed to read file {file}: {e}")
                continue

            for entry in raw_data:
                try:
                    source = clean_code(remove_comments_ast(entry.get("fm", "")))
                    target = clean_code(remove_comments_ast(extract_target_range(entry.get("t", ""))))

                    if (source, target) and (source, target) not in existing_entries:
                        new_data.append({"source": source, "target": target})
                        existing_entries.add((source, target))
                except Exception as e:
                    print(f"[ERROR] Processing entry in {file}: {e}")
                    continue

        if new_data:
            all_data = new_data if overwrite else all_data + new_data
            with open(os.path.join("../..", output_file), "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            print(f"Processed {len(new_data)} new samples. Total samples: {len(all_data)}")
        else:
            print("No new data added.")

    except Exception as e:
        print(f"[ERROR] preprocess_dataset: {e}")


if __name__ == "__main__":
    preprocess_dataset("data/raw", "data/processed.json", overwrite=False)  # True nếu ghi đè file cũ
