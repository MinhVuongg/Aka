import glob
import json
import logging
import os
import re

from src.config.config import REMOVE_COMMENT_MODE, COMMENT_REMOVAL, max_target_length, OPTIMIZE_TARGET_STRATEGY, \
    TARGET_SELETCTION_STRATEGIES

logger = logging.getLogger(__name__)


# Config clang path - Khong xoa
# from clang.cindex import Index, TokenKind, Config
# Config.set_library_path(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "clang+llvm-19.1.7-x86_64-pc-windows-msvc/bin")))

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
        return code


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
    # Tam thoi comment do ko chay duoc tren VAST
    # try:
    #     if not code:
    #         return ""
    #
    #     if isinstance(code, list):
    #         code = " ".join(map(str, code))
    #
    #     index = Index.create()
    #     tu = index.parse('temp.cpp', unsaved_files=[('temp.cpp', code)], args=['-x', 'c++'])
    #     comments = [(token.extent.start.offset, token.extent.end.offset) for token in tu.cursor.get_tokens() if
    #                 token.kind == TokenKind.COMMENT]
    #
    #     if not comments:
    #         return code
    #
    #     result = []
    #     last_end = 0
    #     for start, end in comments:
    #         result.append(code[last_end:start])
    #         last_end = end
    #     result.append(code[last_end:])
    #
    #     return re.sub(r';+', '; ', "".join(result).strip(";")) + ";"
    # except Exception as e:
    #     logger.error(f"[ERROR] remove_comments_ast: {e}")
    #     return code

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


def preprocess_dataset2(input_folder, output_file, tokenizer, optimize_target_strategy):
    """
    XU LY CAU TRUC TRAINING SET RAW MOI
    """
    # Tim json file
    json_files = glob.glob(os.path.join(input_folder, "**", "*.json"), recursive=True)
    logger.info(f"[UET] Tìm thấy {len(json_files)} JSON trong {input_folder}")
    logger.info(f"[UET] Chiến thuật xử lý target: {optimize_target_strategy}")
    new_data = []
    total_processed = 0
    total_target_items = 0
    included_target_items = 0

    # Duyet tung json file
    for file_idx, json_file in enumerate(json_files):
        logger.info(f"[UET] Phân tích json thứ {file_idx + 1}/{len(json_files)}: {json_file}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            if not isinstance(file_data, list):
                file_data = [file_data]

            for entry_idx, entry in enumerate(file_data):
                try:
                    # Lấy focal method (fm)
                    focal_method = ""
                    if "fm" in entry and entry["fm"]:
                        focal_method = clean_code(remove_comments(entry["fm"]))

                    # Lấy focal class
                    focal_class = entry.get("fc", "")
                    focal_class_name = ""
                    if focal_class != "":
                        focal_class_name = extract_class_declaration(focal_class)

                    # Parse constructor signatures - c là một mảng các String
                    constructor_signatures = ""
                    if "c" in entry and entry["c"]:
                        if isinstance(entry["c"], list):
                            constructor_list = []
                            for constructor in entry["c"]:
                                if isinstance(constructor, str):
                                    constructor_list.append(constructor)
                                else:
                                    # Chuyển đổi bất kỳ kiểu dữ liệu nào khác sang string
                                    constructor_list.append(str(constructor))
                            constructor_signatures = clean_code(remove_comments(" ".join(constructor_list)))

                    # Parse method signatures - m là một đối tượng chứa ba mảng string
                    method_signatures = ""
                    if "m" in entry and entry["m"]:
                        all_methods = []

                        # m là một đối tượng với ba thuộc tính mảng
                        if isinstance(entry["m"], dict):
                            # Xử lý called_m (luôn là mảng string)
                            if "called_m" in entry["m"] and entry["m"]["called_m"]:
                                for method in entry["m"]["called_m"]:
                                    if isinstance(method, str):
                                        all_methods.append(method)
                                    else:
                                        all_methods.append(str(method))

                            # Xử lý stub_called_m (luôn là mảng string)
                            if "stub_called_m" in entry["m"] and entry["m"]["stub_called_m"]:
                                for method in entry["m"]["stub_called_m"]:
                                    if isinstance(method, str):
                                        all_methods.append(method)
                                    else:
                                        all_methods.append(str(method))

                            # Xử lý callee_m (luôn là mảng string)
                            if "callee_m" in entry["m"] and entry["m"]["callee_m"]:
                                for method in entry["m"]["callee_m"]:
                                    if isinstance(method, str):
                                        all_methods.append(method)
                                    else:
                                        all_methods.append(str(method))

                        method_signatures = clean_code(remove_comments(" ".join(map(str, all_methods))))

                    # Parse fields - f là một mảng các String
                    fields = ""
                    if "f" in entry and entry["f"]:
                        if isinstance(entry["f"], list):
                            field_list = []
                            for field in entry["f"]:
                                if isinstance(field, str):
                                    field_list.append(field)
                                else:
                                    # Chuyển đổi bất kỳ kiểu dữ liệu nào khác sang string
                                    field_list.append(str(field))
                            fields = clean_code(remove_comments(" ".join(field_list)))

                    # Parse targets from datatest
                    targets = []
                    if "datatest" in entry and entry["datatest"]:
                        total_target_items += len(entry["datatest"])
                        for test_case in entry["datatest"]:
                            if isinstance(test_case, dict) and "td" in test_case and test_case["td"]:
                                test_target = clean_code(remove_comments(extract_target_range(test_case["td"])))
                                if test_target:  # Chỉ thêm nếu target không rỗng
                                    targets.append(test_target)

                            # Xử lý các trường thông tin thực thi
                            executed_fm = ""
                            executed_fm_masked = ""
                            executed_m = ""
                            executed_m_masked = ""
                            testpath = []

                            if isinstance(test_case, dict):
                                if "executed_fm" in test_case and test_case["executed_fm"]:
                                    executed_fm = clean_code(remove_comments(test_case["executed_fm"]))

                                if "executed_fm_masked" in test_case and test_case["executed_fm_masked"]:
                                    executed_fm_masked = clean_code(remove_comments(test_case["executed_fm_masked"]))

                                if "executed_m" in test_case and test_case["executed_m"]:
                                    executed_m = clean_code(remove_comments(test_case["executed_m"]))

                                if "executed_m_masked" in test_case and test_case["executed_m_masked"]:
                                    executed_m_masked = clean_code(remove_comments(test_case["executed_m_masked"]))

                                # Xử lý testpath là một mảng string
                                if "testpath" in test_case and test_case["testpath"]:
                                    if isinstance(test_case["testpath"], list):
                                        for path in test_case["testpath"]:
                                            if isinstance(path, str):
                                                testpath.append(path)
                                            else:
                                                testpath.append(str(path))

                            # Tại đây bạn có thể sử dụng các biến này cho mục đích phân tích

                        # Sắp xếp targets theo độ dài tăng dần
                        targets.sort(key=len)

                    # Source
                    source = "\n".join([
                        "/*FC*/", focal_class_name,
                        "\n{",
                        "/*FM*/ ", focal_method,
                        "/*C*/", constructor_signatures,
                        "/*M*/:", method_signatures,
                        "/*F*/:", fields,
                        "\n}"
                    ])

                    if optimize_target_strategy == TARGET_SELETCTION_STRATEGIES.SORT_BY_TOKEN_AND_CUTOFF:
                        """
                        Ghép tất cả target thành một target duy nhất với <TC> làm dấu phân cách.
                        Chọn những test case tốt.
                        Ưu tiên manual test case (chưa xử lý) và automated test case (ngắn nhất)
                        """
                        final_target = ""
                        included_items_in_current_entry = 0
                        for target_item in targets:
                            # Kiểm tra nếu đây là phần tử đầu tiên
                            if not final_target:
                                temp_target = target_item
                            else:
                                # Thêm dấu phân cách <TC> nếu không phải phần tử đầu tiên
                                temp_target = final_target + " <TC> " + target_item

                            # Kiểm tra độ dài token sau khi thêm
                            if len(tokenizer.encode(temp_target)) <= max_target_length:
                                final_target = temp_target
                                included_items_in_current_entry += 1
                            else:
                                break
                        included_target_items += included_items_in_current_entry

                        # Chỉ thêm vào dữ liệu nếu có ít nhất một phần tử trong final_target
                        if final_target and source:
                            new_data.append({"source": source, "target": final_target})
                            total_processed += 1

                            if total_processed % 100 == 0:
                                logger.info(f"[UET] Thu thập được {total_processed} hàm tới lúc này")

                    elif optimize_target_strategy == TARGET_SELETCTION_STRATEGIES.NONE:
                        """
                        Ghép tất cả target thành một target duy nhất với <TC> làm dấu phân cách
                        """
                        if source and targets:
                            combined_target = " <TC> ".join(targets)
                            new_data.append({"source": source, "target": combined_target})
                            total_processed += 1

                            if total_processed % 100 == 0:
                                logger.info(f"[UET] Thu thập được {total_processed} hàm tới lúc này")

                except Exception as e:
                    logger.error(f"[UET] [ERROR] Processing entry in file {json_file}, index {entry_idx}: {str(e)}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"[UET] [ERROR] Failed to parse JSON file {json_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"[UET] [ERROR] Failed to process file {json_file}: {e}")
            continue

    # Tính và hiển thị tỷ lệ
    if optimize_target_strategy == TARGET_SELETCTION_STRATEGIES.SORT_BY_TOKEN_AND_CUTOFF:
        if total_target_items > 0:
            inclusion_ratio = (included_target_items / total_target_items) * 100
            logger.info(
                f"[UET] Tỷ lệ test case chọn được thêm vào tập training set mô hình: {included_target_items}/{total_target_items} ({inclusion_ratio:.2f}%).")

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    logger.info(f"[UET] Processed {len(new_data)} new samples. Total samples: {len(new_data)}")
    logger.info(f"[UET] Data successfully saved to {output_file}")

    return new_data