import os
import glob
import json
import re
import sys
from clang.cindex import Index, TokenKind, Config
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.config.config import MODEL_NAME, TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS, REMOVE_COMMENT_MODE, COMMENT_REMOVAL

# from src.train.base_trainer import Salesforce/codet5-base
#
# # Đảm bảo module có thể tìm thấy đúng thư mục
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging

logger = logging.getLogger(__name__)

# Config clang path
Config.set_library_path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "clang+llvm-19.1.7-x86_64-pc-windows-msvc/bin")))

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
        logger.error(f"[ERROR] extract_target_range: {e}")
        logger.error(f"[ERROR] extract_target_range: {e}")
        return target

def remove_comments(code):
    if REMOVE_COMMENT_MODE == COMMENT_REMOVAL.AST:
        return remove_comments_ast(code)
    elif REMOVE_COMMENT_MODE == COMMENT_REMOVAL.REGREX:
        return remove_comments_regex(code)
    else:
        return code;

def remove_comments_regex(code):
    try:
        if not code:
            return ""

        if isinstance(code, list):
            code = " ".join(map(str, code))

        # Loại bỏ comment dạng /* ... */
        import re
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)

        # Loại bỏ comment dạng //
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)

        # Xu ly dấu chấm phẩy dư thừa
        return re.sub(r';+', '; ', code.strip(";")) + ";"
    except Exception as e:
        logger.error(f"[UET] [ERROR] remove_comments_regex: {e}")
        return code


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
        logger.error(f"[ERROR] remove_comments_ast: {e}")
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
        logger.error(f"[ERROR] clean_code: {e}")
        return code


def extract_class_declaration(focal_class):
    match = re.search(r'\b(?:final\s+)?(?:class)\s+\w+\s*{', focal_class)

    if match:
        return match.group(0).strip(" {")
    else:
        return ""


"""
XU LY CAU TRUC TRAINING SET RAW MOI
"""
def preprocess_dataset2(input_data, output_file, overwrite=False):
    try:
        logger.info("[UET] Processing data: {} items".format(len(input_data)))
        new_data = []

        # Check if output file already exists
        all_data = []
        if not overwrite and os.path.exists(os.path.join("../..", output_file)):
            with open(os.path.join("../..", output_file), "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"[UET] [ERROR] JSONDecodeError in {output_file}: {e}")
                    logger.error("[UET] Warning: Output file contains invalid JSON, starting fresh.")

        existing_entries = {(entry["source"], entry["target"]) for entry in all_data}

        for entry in input_data:
            try:
                focal_method = clean_code(remove_comments(entry.get("fm", "")))
                focal_class = entry.get("fc", "")
                focal_class_name = ""
                if focal_class != "":
                    focal_class_name = extract_class_declaration(focal_class)

                # Process other fields
                constructor_signatures = ""
                if "c" in entry:
                    if isinstance(entry["c"], list):
                        constructor_signatures = clean_code(remove_comments(" ".join(entry["c"])))
                    else:
                        constructor_signatures = clean_code(remove_comments(entry["c"]))

                method_signatures = ""
                if "m" in entry:
                    if isinstance(entry["m"], list):
                        method_signatures = clean_code(remove_comments(" ".join(entry["m"])))
                    else:
                        method_signatures = clean_code(remove_comments(entry["m"]))

                fields = ""
                if "f" in entry:
                    if isinstance(entry["f"], list):
                        fields = clean_code(remove_comments(" ".join(entry["f"])))
                    else:
                        fields = clean_code(remove_comments(entry["f"]))

                # Use datatest field for target if available, otherwise use 't' field
                target = ""
                if "datatest" in entry and entry["datatest"]:
                    for test_case in entry["datatest"]:
                        if "simplified_t" in test_case:
                            target = clean_code(remove_comments(extract_target_range(test_case["simplified_t"])))
                            break
                        elif "td" in test_case:
                            target = clean_code(remove_comments(extract_target_range(test_case["td"])))
                            break

                # If no target found in datatest, use 't' if available
                if not target and "t" in entry:
                    target = clean_code(remove_comments(extract_target_range(entry.get("t", ""))))

                # Construct source string
                source = "\n".join([
                    "FC:", focal_class_name,
                    "FM:", focal_method,
                    "C:", constructor_signatures,
                    "M:", method_signatures,
                    "F:", fields
                ])

                # Only add if both source and target exist and this pair hasn't been added before
                if source and target and (source, target) not in existing_entries:
                    new_data.append({"source": source, "target": target})
                    existing_entries.add((source, target))
            except Exception as e:
                logger.error(f"[UET] [ERROR] Processing entry: {e}")
                continue

        if new_data:
            all_data = new_data if overwrite else all_data + new_data
            with open(os.path.join("../..", output_file), "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            logging.info(f"[UET] Processed {len(new_data)} new samples. Total samples: {len(all_data)}")
        else:
            logging.info("[UET] No new data added.")

        return all_data

    except Exception as e:
        logger.error(f"[UET] [ERROR] preprocess_dataset2: {e}")
        return []

def preprocess_dataset(input_folder, output_file, overwrite=False):
    """
    Xử lý tập dữ liệu từ thư mục đầu vào và lưu kết quả vào file JSON đầu ra.
    """
    try:
        json_files = glob.glob(os.path.join(input_folder, "**", "*.json"), recursive=True)
        # logger.info(os.path.join(input_folder, "**", "*.json"))
        logger.info("[UET] Total json files: {}".format(len(json_files)))
        new_data = []

        # Kiểm tra nếu file đầu ra đã tồn tại
        all_data = []
        if not overwrite and os.path.exists(os.path.join("../..", output_file)):
            with open(os.path.join("../..", output_file), "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"[UET] [ERROR] JSONDecodeError in {output_file}: {e}")
                    logger.error("[UET] Warning: Output file contains invalid JSON, starting fresh.")

        existing_entries = {(entry["source"], entry["target"]) for entry in all_data}

        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"[UET] [ERROR] JSONDecodeError in {file}: {e}")
                continue
            except Exception as e:
                logger.error(f"[UET] [ERROR] Failed to read file {file}: {e}")
                continue

            for entry in raw_data:
                try:
                    focal_method = clean_code(remove_comments(entry.get("fm", "")))
                    focal_class = entry.get("fc", "")
                    focal_class_name = ""
                    if focal_class != "":
                        focal_class_name = extract_class_declaration(focal_class)

                    constructor_signatures = clean_code(remove_comments(entry.get("c", "")))
                    method_signatures = clean_code(remove_comments(entry.get("m", "")))
                    fields = clean_code(remove_comments(entry.get("f", "")))
                    source = "\n".join(
                        ["FC:", focal_class_name, "FM:", focal_method, "C:", constructor_signatures, "M:",
                         method_signatures,
                         "F:", fields])
                    target = clean_code(remove_comments(extract_target_range(entry.get("t", ""))))

                    if (source, target) and (source, target) not in existing_entries:
                        new_data.append({"source": source, "target": target})
                        existing_entries.add((source, target))
                except Exception as e:
                    logger.error(f"[UET] [ERROR] Processing entry in {file}: {e}")
                    continue

        if new_data:
            all_data = new_data if overwrite else all_data + new_data
            with open(os.path.join("../..", output_file), "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            logging.info(f"[UET] Processed {len(new_data)} new samples. Total samples: {len(all_data)}")
        else:
            logging.info("[UET] No new data added.")

    except Exception as e:
        logger.error(f"[UET] [ERROR] preprocess_dataset: {e}")


def main():
    """
    Main function to test preprocess_dataset2 with a single file.
    """
    # Define input and output files
    input_file = "/Users/ducanhnguyen/Documents/aka-llm/training-set (1).json"  # Replace with your file path
    output_file = "/Users/ducanhnguyen/Documents/aka-llm/training-set (1)_pre.json"

    # Check if the input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        return

    try:
        logger.info(f"Processing file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = preprocess_dataset2(data, output_file, overwrite=True)
        logger.info(f"Processed {len(processed_data)} items")
        logger.info(f"Data has been saved to {os.path.abspath(output_file)}")

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {input_file}: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()