import os
import numpy as np

def find_path_to_folder(subfolder_name: list[str]) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    upper_dir = os.path.dirname(current_dir)
    path = os.path.join(upper_dir, *subfolder_name)
    return path

def write_to_csv(file_path, first_line, array):
    # should consider call back function to reuse the file existence check
    if not os.path.exists(file_path):
        np.savetxt(file_path, array, delimiter=',', fmt='%.17f', header=first_line, comments='')
        print("csv data is written into:\n" + os.path.relpath(file_path, os.getcwd()))
    else:
        raise ValueError("File already exist:\n" + file_path)

