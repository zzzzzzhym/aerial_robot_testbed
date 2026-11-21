"""Signal post-processing data layer (production).

The extended post-processing layer for the learning pipeline. It subsumes the traditional
BEMT-fit residual generation (reusing `learning.bemt_traditional_fit.bemt_model.BemtModel` and the
p600 lookup table) and additionally generates the physically-grounded, OOD-robust signals the
phi/a disturbance model consumes. It writes them ADDITIVELY and IDEMPOTENTLY onto the dataset CSVs,
for both training and test data.

Columns written (existing columns preserved; these are overwritten if present):
  - f_residual          : BEMT residual force in BODY frame (reuses BemtModel.get_residual_force).
  - wind_magnitude      : |background wind| (scalar, m/s; frame-invariant; the single OOD dof).
  - wind_body_direction : unit background-wind vector in the body (shared_r_disk) frame [3].
  - rotor_{r}_v_i       : per-rotor induced velocity (signed downwash scalar, m/s) from the lookup
                          table -- the physical inflow, in-domain even at an OOD background wind.

Usage (from repo root):
    python -m learning.signal_post_processing \
        wind_free_space_training_cfd_our_proposal wind_free_space_test_cfd_our_proposal
or programmatically:
    SignalPostProcessor().augment_folder("wind_free_space_test_cfd_our_proposal")

Moved out of learning/bemt_traditional_fit/ (where it began as general_post_processing.py): once the
post-processing is extended beyond the BEMT residual it belongs in the production pipeline, not the
traditional-fitting folder. It still REUSES the fit code via imports.
"""
import os
import sys

import numpy as np
import pandas as pd

import data_factory
import learning.bemt_traditional_fit as bemt_fit
from inflow_model.blade_params import P600_Blade
import inflow_model.propeller_lookup_table as propeller_lookup_table
from drone import parameters
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.fitting_config import FittingConfig

# Column names this layer owns (idempotent overwrite set).
WIND_MAGNITUDE = "wind_magnitude"
WIND_BODY_DIRECTION = "wind_body_direction"
ROTOR_V_I = [f"rotor_{r}_v_i" for r in range(4)]
RESIDUAL = "f_residual"
GENERATED_COLUMNS = [RESIDUAL, WIND_MAGNITUDE, WIND_BODY_DIRECTION, *ROTOR_V_I]

# The reused BEMT-fit resolution config lives with the fit code it configures.
_BEMT_FIT_DIR = os.path.dirname(bemt_fit.__file__)


class SignalPostProcessor:
    def __init__(self, lookup_table_name: str = "p600_single_rotor_after_bemt_fit",
                 config_name: str = "config_full_vehicle.yaml"):
        self.blade = P600_Blade()
        self.drone = parameters.P600()
        config_path = config_name if os.path.isabs(config_name) else \
            os.path.join(_BEMT_FIT_DIR, config_name)
        model_config = FittingConfig.from_yaml(config_path).model
        self.model = BemtModel(self.blade, self.drone, model_config)
        self.lookup_table = propeller_lookup_table.PropellerLookupTable.Reader(lookup_table_name)

    # -- per-frame signal computation -----------------------------------------------------------
    def _rotor_v_i_scalar(self, dataset, i: int, r: int) -> float:
        """Signed induced velocity (positive = downwash) for rotor r, from the lookup table.

        The table returns v_i as an inertial vector (-v_i_scalar * disk_z); project back onto the
        disk normal to recover the scalar (same convention as BemtModel.compute_average_v_i)."""
        v_i_inertial = BemtModel.compute_v_i_with_lookup_table(
            getattr(dataset, f"u_free_{r}")[i], getattr(dataset, f"v_forward_{r}")[i],
            dataset.shared_r_disk[i], getattr(dataset, f"omega_{r}")[i],
            self.model.params.is_ccw_blade[r], self.lookup_table)
        return float(-(dataset.shared_r_disk[i].T @ v_i_inertial)[2])

    def compute_columns(self, dataset: data_factory.FittingDataset) -> dict:
        n = len(dataset)
        residual = np.empty((n, 3))
        wind_mag = np.empty(n)
        wind_dir = np.empty((n, 3))
        v_i = np.empty((n, 4))
        for i in range(n):
            # residual force in body frame (reuse BEMT-fit path, lookup table = True, body = True)
            residual[i] = self.model.get_residual_force(dataset, i, self.lookup_table, True, True)
            # background wind in body frame (average the 4 rotor local winds; they share background)
            u_free_avg = 0.25 * (dataset.u_free_0[i] + dataset.u_free_1[i]
                                 + dataset.u_free_2[i] + dataset.u_free_3[i])
            wind_body = dataset.shared_r_disk[i].T @ u_free_avg
            mag = float(np.linalg.norm(wind_body))
            wind_mag[i] = mag
            wind_dir[i] = wind_body / mag if mag > 1e-9 else np.zeros(3)
            for r in range(4):
                v_i[i, r] = self._rotor_v_i_scalar(dataset, i, r)
        cols = {
            RESIDUAL: residual.tolist(),
            WIND_MAGNITUDE: wind_mag.tolist(),
            WIND_BODY_DIRECTION: wind_dir.tolist(),
        }
        for r in range(4):
            cols[ROTOR_V_I[r]] = v_i[:, r].tolist()
        return cols

    # -- CSV augmentation -----------------------------------------------------------------------
    def augment_csv(self, dataset: data_factory.FittingDataset) -> dict:
        """Compute and append the generated columns to the dataset's CSV file (additive)."""
        cols = self.compute_columns(dataset)
        df = pd.read_csv(dataset.path_to_data_file)
        for name, values in cols.items():
            df[name] = values
        df.to_csv(dataset.path_to_data_file, index=False, float_format="%.17f")
        return cols

    def augment_folder(self, subfolder: str) -> None:
        """Augment every CSV under data/training/<subfolder> with the generated signal columns."""
        factory = data_factory.FittingFactory()
        data_list = data_factory.generate_data_list(subfolder, ".csv")
        datasets = factory.prepare_datasets(data_list)
        for dataset in datasets:
            rel = os.path.relpath(dataset.path_to_data_file)
            print(f"[signal_post_processing] augmenting {rel} ...", flush=True)
            self.augment_csv(dataset)
        print(f"[signal_post_processing] done: {len(datasets)} file(s) in {subfolder}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        argv = ["wind_free_space_training_cfd_our_proposal",
                "wind_free_space_test_cfd_our_proposal"]
    proc = SignalPostProcessor()
    for subfolder in argv:
        proc.augment_folder(subfolder)


if __name__ == "__main__":
    main()
