

import os
import json
import glob

# Directory path containing JSON files to convert
path = "/path/to/your/json/folder"  # REPLACE THIS with your actual path
outpath = "/path/to/output/folder"  # REPLACE THIS with your desired output path


def convert_json_structure(data_item):
    """Convert a single item from old structure to new structure.
    Only includes fields specified in the new structure."""
    new_item = {
        "fm": data_item.get("fm", ""),
        "f": data_item.get("f", []),
        "path_fm": data_item.get("path_fm", ""),
        "c": data_item.get("c", [""]),
        "fc": data_item.get("fc", ""),
        "m": data_item.get("m", []),
        "datatest": []
    }

    # Create the datatest entry
    datatest_entry = {
        "id": 0,
        "dt": {},  # Empty by default
        "td": "",
        "simplified_t": data_item.get("simplified_t", []),
        "isAutomated": True
    }

    # Add t as td if it exists
    if "t" in data_item:
        datatest_entry["td"] = data_item.get("t", [])

    # Always add the datatest entry, even if simplified_t or t doesn't exist
    new_item["datatest"].append(datatest_entry)

    return new_item


def process_json_file(input_file_path, output_file_path):
    """Process a single JSON file, converting from old to new structure."""
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert each item in the JSON array
        new_data = [convert_json_structure(item) for item in data]

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Write the new structure to the output file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"Converted: {input_file_path} -> {output_file_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_file_path}: {str(e)}")
        return False


def process_directory(input_directory, output_directory):
    """Recursively process all JSON files in directory and subdirectories."""
    # Get all JSON files in current directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))

    # Process each JSON file
    success_count = 0
    for input_file_path in json_files:
        # Create parallel path in output directory
        rel_path = os.path.relpath(input_file_path, input_directory)
        output_file_path = os.path.join(output_directory, rel_path)

        if process_json_file(input_file_path, output_file_path):
            success_count += 1

    # Recursively process subdirectories
    subdirectories = [d for d in os.listdir(input_directory)
                      if os.path.isdir(os.path.join(input_directory, d))]

    for subdir in subdirectories:
        input_subdir_path = os.path.join(input_directory, subdir)
        output_subdir_path = os.path.join(output_directory, subdir)
        success_count += process_directory(input_subdir_path, output_subdir_path)

    return success_count


def main():
    # Set the input/output directories
    input_directory = "/Users/ducanhnguyen/Documents/aka-llm/data/validation2"
    output_directory = "/Users/ducanhnguyen/Documents/aka-llm/data/oldToNewFormat"

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' is not valid.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Process all files
    print(f"Converting JSON files from {input_directory} to {output_directory}...")
    success_count = process_directory(input_directory, output_directory)
    print(f"Conversion complete. Successfully converted {success_count} files.")


if __name__ == "__main__":
    main()