"""Per-condition fit results and the wind-condition extraction that labels them.

The quadratic pipeline fits one profile per wind condition.  Each fit produces a
PerConditionResult; the engine returns them as a plain list[PerConditionResult] so
that downstream code (lookup-table maker, plotter) can simply traverse it without
knowing how many conditions there are.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

import data_factory


@dataclass
class PerConditionResult:
    """Fitted parameters, the wind condition, and the dataset they were fitted on."""
    params: np.ndarray   # shape (9,): [a_fx, b_fx, c_fx, ..., a_fz, b_fz, c_fz]
    u_free_x: float      # total background wind speed (m/s)
    pitch: float         # wind pitch angle relative to disk plane (rad)
    # Source dataset this fit came from — kept so the fit can be plotted against
    # the very data it was fitted on, without re-pairing results to datasets.
    dataset: Optional[data_factory.FittingDataset] = None


def extract_wind_condition(dataset: data_factory.FittingDataset) -> tuple[float, float]:
    """Return (u_free_x, pitch) from the mean background wind of a dataset."""
    u_free_mean = np.mean(dataset.u_free_0, axis=0)
    r_disk = dataset.shared_r_disk[len(dataset) // 2]
    disk_normal = r_disk[:, 2]
    u_normal = float(u_free_mean @ disk_normal)
    u_plane = float(np.linalg.norm(u_free_mean - u_normal * disk_normal))
    u_free_x = float(np.linalg.norm(u_free_mean))
    pitch = float(np.arctan2(u_normal, u_plane))
    return u_free_x, pitch
