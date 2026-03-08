# The orignal neurobem data does not have residual force and wind veloctity data columns, so we need to add them in the csv file before feeding it to the model

import pandas as pd
import common_utils.file_manager


def preprocess_neurobem_data(file_path: str):
    df = pd.read_csv(common_utils.file_manager.find_path_to_folder(["data", "training", file_path]))

    # 2. create a new column from existing columns
    df["new_column"] = df["col_a"] + df["col_b"]

    # 3. optionally save back
    df.to_csv("data_processed.csv", index=False)


