import os
import warnings
import yaml
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import common_utils


class Logger:
    """Collects per-step signal dicts into lists, then converts to numpy arrays.

    Buffer keys are created on first append (no pre-registration needed).
    logger_config.yaml is a flat list of signal names to include in CSV/pkl output.
    """
    def __init__(self) -> None:
        self.csv_keys = self._load_csv_keys("logger_config.yaml")
        self.buffer = {}
        self.output = {}

    @staticmethod
    def _load_csv_keys(filename: str) -> list:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, filename)
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def convert_buffer_to_output(self):
        for key in self.buffer:
            self.output[key] = np.array(self.buffer[key])

    def get_names_of_items_to_csv(self) -> list:
        return self.csv_keys

    def make_data_frame(self) -> pd.DataFrame:
        df = pd.DataFrame()
        missing_keys = []
        for key in self.csv_keys:
            if key in self.output:
                df[key] = self.output[key].tolist()
            else:
                missing_keys.append(key)
        if missing_keys:
            warnings.warn(f"logger_config.yaml lists keys never logged, absent from CSV: {missing_keys}")
        return df

    @staticmethod
    def result_file_path(file_name: str, type: str) -> str:
        """Build the path a result would be written to, without writing anything."""
        suffix = ".csv" if type == 'csv' else ".pkl"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(current_dir)
        return os.path.join(upper_dir, "data", "training", file_name + suffix)

    @staticmethod
    def result_exists(file_name: str, type: str) -> bool:
        """Whether a result file already exists (for pre-sim skipping)."""
        return os.path.exists(Logger.result_file_path(file_name, type))

    def log_sim_result(self, file_name: str, type: str) -> None:
        file_path = Logger.result_file_path(file_name, type)
        if not os.path.exists(file_path):
            if type == 'csv':
                df = self.make_data_frame()
                df.to_csv(file_path, index=False, float_format='%.17f')
            elif type == 'pkl':
                with open(file_path, "wb") as f:
                    pickle.dump(self.buffer, f)
            print("Sim data is written into:\n" + os.path.relpath(file_path, os.getcwd()))
        else:
            raise ValueError("File already exist:\n" + file_path)

    def dump_full_log(self, tag: str = None, folder: str = "runs") -> str:
        """Auto-save EVERY logged signal (not just logger_config.yaml) to a timestamped,
        git-ignored CSV under data/<folder>/. This is a scratch/inspection dump; the curated
        CSV (csv_keys) remains the ML-pipeline input. Timestamp is first for chronological sort
        and uses no colons (Windows/WSL-safe). Returns the written path."""
        if not self.output:
            self.convert_buffer_to_output()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(os.path.dirname(current_dir), "data", folder)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{tag}.csv" if tag else f"{stamp}.csv"
        file_path = os.path.join(out_dir, name)

        df = pd.DataFrame()
        expected = None
        for key, arr in self.output.items():
            vals = arr.tolist()
            if expected is None:
                expected = len(vals)
            if len(vals) != expected:   # guard against ragged (conditionally-logged) fields
                warnings.warn(f"dump_full_log: skipping ragged field '{key}' "
                              f"({len(vals)} rows vs {expected})")
                continue
            df[key] = vals
        df.to_csv(file_path, index=False, float_format='%.17f')
        print("Full log dumped to:\n" + common_utils.file_manager.relative_to_root(file_path))
        return file_path

    def generate_column_map(self) -> dict:
        """Generate a mapping of CSV column names to their index positions."""
        headers = [k for k in self.csv_keys if k in self.output]
        header_map = {header: idx for idx, header in enumerate(headers)}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        map_file_path = os.path.join(current_dir, "column_map.yaml")
        with open(map_file_path, "w") as f:
            for key, value in header_map.items():
                yaml.dump({key: value}, f)
