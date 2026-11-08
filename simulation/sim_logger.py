import os
import warnings
import yaml
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval
from datetime import datetime

import common_utils


# list/matrix cells are written as "[...]"; literal_eval rejects inf/nan, so fall back to a
# restricted eval (no builtins) that understands those tokens.
_PARSE_NS = {"inf": float("inf"), "nan": float("nan"), "__builtins__": {}}


def _parse_cell(x):
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError):
        return eval(x, dict(_PARSE_NS))


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
        out_dir = Logger.dumps_dir(folder)
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

    @staticmethod
    def dumps_dir(folder: str = "runs") -> str:
        """Absolute path to the full-log dump folder (data/<folder>/)."""
        return common_utils.file_manager.find_path_to_folder(["data", folder])

    @staticmethod
    def latest_dump(folder: str = "runs") -> str:
        """Path to the most recent full-log dump. Filenames are timestamp-first, so lexical
        sort is chronological."""
        out_dir = Logger.dumps_dir(folder)
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(f"no dump folder yet: {out_dir}")
        files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"no full-log dumps in {out_dir}")
        return os.path.join(out_dir, sorted(files)[-1])

    @staticmethod
    def load_full_log(file_path: str = None, folder: str = "runs") -> dict:
        """Recover a full-log dump into an `output`-style dict ({field: np.ndarray}) with native
        shapes restored (scalars [N], vectors [N,3], matrices [N,3,3]). If file_path is None,
        loads the LATEST dump in data/<folder>/. Plot it with Plotter.from_output:

            output = sim_logger.Logger.load_full_log()          # latest dump
            plotter.Plotter.from_output(output).plot_disturbance_force(output)
        """
        if file_path is None:
            file_path = Logger.latest_dump(folder)
        df = pd.read_csv(file_path)
        output = {}
        for col in df.columns:
            series = df[col]
            if series.size and isinstance(series.iloc[0], str):   # vector/matrix cell
                series = series.map(_parse_cell)
            output[col] = np.array(series.to_list())
        return output

    def generate_column_map(self) -> dict:
        """Generate a mapping of CSV column names to their index positions."""
        headers = [k for k in self.csv_keys if k in self.output]
        header_map = {header: idx for idx, header in enumerate(headers)}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        map_file_path = os.path.join(current_dir, "column_map.yaml")
        with open(map_file_path, "w") as f:
            for key, value in header_map.items():
                yaml.dump({key: value}, f)
