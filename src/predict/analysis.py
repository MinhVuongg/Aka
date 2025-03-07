import pandas as pd
import difflib
import re


def format_cpp_code(code: str) -> str:
   if isinstance(code, str):
       # Thêm xuống dòng sau dấu chấm phẩy nếu chưa có
       code = re.sub(r";\s*", ";\n", code)

       # # Thêm xuống dòng sau dấu ngoặc mở nếu chưa có
       # code = re.sub(r"(\{)\s*", r"\1\n", code)
       #
       # # Thêm thụt lề cơ bản nếu cần
       # formatted_code = "\n".join(line.strip() for line in code.split("\n"))
   return code

def format_newline_for_web(text: str) -> str:
    if isinstance(text, str):
        return text.replace("\n", "<br>")
    else:
        return text

def highlight_diff(expected, predicted):
    # expected = format_cpp_code(expected)
    # predicted = format_cpp_code(predicted)
    diff = difflib.ndiff(expected, predicted)
    result_expected, result_predicted = "", ""

    for d in diff:
        if d.startswith(" "):
            result_expected += d[2]
            result_predicted += d[2]
        elif d.startswith("-"):
            result_expected += f'<span class="text-danger bg-light-red">{d[2]}</span>'
        elif d.startswith("+"):
            result_predicted += f'<span class="text-success bg-light-green">{d[2]}</span>'
    result_expected = format_newline_for_web(result_expected)
    result_predicted = format_newline_for_web(result_predicted)
    return result_expected, result_predicted

def analysis_result(file_path):
    df = pd.read_csv(file_path)
    if 'Expected Target' not in df.columns or 'Predicted Target' not in df.columns:
        raise ValueError("File CSV phải có hai cột: 'Expected Target' và 'Predicted Target'")

    columns_to_normalize = ['Source', 'Expected Target', 'Predicted Target']

    df[columns_to_normalize] = df[columns_to_normalize].map(format_cpp_code)
    df[['Expected Target', 'Predicted Target']] = df.apply(
        lambda row: highlight_diff(str(row['Expected Target']), str(row['Predicted Target'])), axis=1,
        result_type='expand'
    )

    df[columns_to_normalize] = df[columns_to_normalize].map(format_newline_for_web)
    df = df.astype(str)

    # Tạo HTML với Bootstrap
    html_table = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Comparison Result</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <style>
            body {{
                font-family: monospace;
            }}
            .table-container {{ 
                max-width: 100%;
                overflow-x: auto;
                padding: 20px;
            }}
            .bg-light-red {{ 
                background-color: #ffcccc;
                display: inline;  /* Đảm bảo chữ không bị tách */
            }}
            .bg-light-green {{ 
                background-color: #ccffcc;
                display: inline;
            }}
            table {{
                width: 100%;
                table-layout: fixed;
                word-break: break-word;
                white-space: pre-wrap; /* Giữ nguyên định dạng */
                font-size: 16px;
            }}
            th {{
                text-align: center !important;
                font-weight: bold;
                background-color: #f8f9fa;
            }}
            td {{
                white-space: pre-wrap; /* Tránh lỗi giãn khoảng cách */
            }}
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h2 class="text-center">Comparison Result</h2>
            <div class="table-container">
                {df.to_html(classes='table table-bordered table-striped', escape=False, index=False)}
            </div>
        </div>
    </body> 
    </html>
    '''

    with open("comparison_result.html", "w", encoding="utf-8") as f:
        f.write(html_table)

    print("Kết quả đã được lưu vào comparison_result.html")

# Chạy đánh giá
if __name__ == "__main__":
    analysis_result(OUTPUT_VALIDATIONSET_CSV)