import os
import numpy as np

def find_path_to_folder(subfolder_name: list[str]) -> str:
    """Find the path to a folder from root folder of the repo."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    upper_dir = os.path.dirname(current_dir)    # the upper dir should be the root folder of the repo
    path = os.path.join(upper_dir, *subfolder_name)
    return path

def relative_to_root(path: str) -> str:
    """Return path relative to the repo root, for clean portable logging (never an abs path)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current_dir)    # repo root (parent of common_utils)
    return os.path.relpath(path, root)

def write_to_csv(file_path, first_line, array):
    # should consider call back function to reuse the file existence check
    if not os.path.exists(file_path):
        np.savetxt(file_path, array, delimiter=',', fmt='%.17f', header=first_line, comments='')
        print("csv data is written into:\n" + os.path.relpath(file_path, os.getcwd()))
    else:
        raise ValueError("File already exist:\n" + file_path)

