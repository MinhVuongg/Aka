import json
from typing import List

from src.data.object.CodeNode import CodeNode
from src.data.object.TestCase import TestCase


def parse_json_to_nodes(json_file_path: str) -> List[CodeNode]:
    """
    Parse một file JSON thành danh sách các CodeNode.

    Args:
        json_file_path: Đường dẫn đến file JSON

    Returns:
        List[CodeNode]: Danh sách các node mã nguồn
    """
    nodes = []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Đảm bảo data là một list
        if not isinstance(data, list):
            data = [data]

        for item in data:
            # Tạo node mới sử dụng thuộc tính private
            node = CodeNode(
                _fm=item.get("fm", ""),
                _fc=item.get("fc", ""),
                _path_fm=item.get("path_fm", ""),
                _m=item.get("m", []),
                _c=item.get("c", []),
                _f=item.get("f", [])
            )

            # Xử lý test cases
            if "datatest" in item and isinstance(item["datatest"], list):
                datatest_list = []
                for test_case_data in item["datatest"]:
                    test_case = TestCase(
                        _id=test_case_data.get("id", 0),
                        _td=test_case_data.get("td", ""),
                        _simplified_t=test_case_data.get("simplified_t", ""),
                        _isAutomated=test_case_data.get("isAutomated", False),
                        _dt=test_case_data.get("dt", {})
                    )
                    datatest_list.append(test_case)

                # Sử dụng setter để thiết lập danh sách datatest
                node.datatest = datatest_list

            nodes.append(node)

        return nodes

    except Exception as e:
        print(f"Error parsing JSON file: {e}")
        return []


def print_node_details(node):
    """Hiển thị thông tin của một CodeNode theo cấu trúc thụt đầu dòng.

    Args:
        node (CodeNode): Node cần hiển thị thông tin
    """

    # Hàm để rút gọn chuỗi dài và loại bỏ linebreak
    def truncate_string(s, max_length=100):
        """Rút gọn chuỗi nếu dài hơn độ dài tối đa và loại bỏ linebreak"""
        if not isinstance(s, str):
            return s

        # Thay thế tất cả các linebreak bằng khoảng trắng
        s = s.replace('\n', ' ').replace('\r', ' ')

        # Thay thế nhiều khoảng trắng liên tiếp bằng một khoảng trắng
        s = ' '.join(s.split())

        # Cắt bớt nếu quá dài, hiển thị 100 ký tự đầu và 5 ký tự cuối
        if len(s) > max_length:
            return s[:100] + "..." + s[-5:]
        return s

    def print_complex_object(obj, indent=0):
        """In đệ quy một đối tượng phức tạp với thụt lề.

        Args:
            obj: Đối tượng cần in
            indent: Số lượng tab thụt vào
        """
        prefix = '\t' * indent

        if isinstance(obj, dict):
            print(f"{prefix}[Dict] - {len(obj)} keys")
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    # Hiển thị key và loại của value
                    print(f"{prefix}\t[{key}] - {type(value).__name__} với {len(value)} phần tử")
                    print_complex_object(value, indent + 1)
                else:
                    # Hiển thị key và value trên cùng một dòng
                    value_str = truncate_string(str(value))
                    print(f"{prefix}\t[{key}] - {value_str}")

        elif isinstance(obj, list):
            print(f"{prefix}[List] - {len(obj)} items")
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    print(f"{prefix}\t[{i}] - {type(item).__name__} với {len(item)} phần tử")
                    print_complex_object(item, indent + 1)
                else:
                    # Hiển thị index và value trên cùng một dòng
                    item_str = truncate_string(str(item))
                    print(f"{prefix}\t[{i}] - {item_str}")

        elif isinstance(obj, str):
            print(f"{prefix}{truncate_string(obj)}")

        elif isinstance(obj, (int, float, bool)) or obj is None:
            print(f"{prefix}{obj}")

        else:
            print(f"{prefix}{truncate_string(str(obj))}")

    print("\n========== NODE DETAILS ==========")
    print(f"[fm] - {truncate_string(node.fm)}")
    print(f"[fc] - {truncate_string(node.fc)}")
    print(f"[path_fm] - {truncate_string(node.path_fm)}")

    # Hiển thị thông tin về methods
    print(f"[m] - {len(node.m)}")
    for i, method in enumerate(node.m):
        print(f"\t[{i}] - {truncate_string(method)}")

    # Hiển thị thông tin về constructors
    print(f"[c] - {len(node.c)}")
    for i, constructor in enumerate(node.c):
        print(f"\t[{i}] - {truncate_string(constructor)}")

    # Hiển thị thông tin về fields
    print(f"[f] - {len(node.f)}")
    for i, field in enumerate(node.f):
        print(f"\t[{i}] - {truncate_string(field)}")

    # Hiển thị thông tin về test cases
    print(f"[datatest] - {len(node.datatest)}")
    for i, test_case in enumerate(node.datatest):
        print(f"\t[{i}] - ID: {test_case.id}")
        print(f"\t\t[isAutomated] - {test_case.isAutomated}")
        print(f"\t\t[td] - {truncate_string(test_case.td)}")
        print(f"\t\t[simplified_t] - {truncate_string(test_case.simplified_t)}")

        # Hiển thị chi tiết về dt
        if test_case.dt:
            print(f"\t\t[dt] - Dict với {len(test_case.dt)} keys")
            for key, value in test_case.dt.items():
                if isinstance(value, (dict, list)):
                    print(f"\t\t\t[{key}] - {type(value).__name__} với {len(value)} phần tử")
                    print_complex_object(value, indent=4)
                else:
                    value_str = truncate_string(str(value))
                    print(f"\t\t\t[{key}] - {value_str}")
if __name__ == "__main__":
    nodes = parse_json_to_nodes(
        "/Users/ducanhnguyen/Documents/aka-llm/data/tmp.json")
    # all_nodes = process_all_json_files("path/to/your/directory")
    if nodes:
        print_node_details(nodes[0])
