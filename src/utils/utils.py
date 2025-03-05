import os
def normalize_path(path_str):
    normalized = path_str.replace('\\', '/')
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return os.path.normpath(normalized)