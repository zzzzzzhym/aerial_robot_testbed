"""Per-condition fit results and the wind-condition extraction that labels them.

The quadratic pipeline fits one profile per wind condition.  Each fit produces a
PerConditionResult; the engine returns them as a plain list[PerConditionResult] so
that downstream code (lookup-table maker, plotter) can simply traverse it without
knowing how many conditions there are.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

import data_factory


# Nominal sweep grid for the magnitude/pitch data collection
# (see simulation/training_data_user_guide.ipynb fitting_config).  Pass these to
# extract_wind_condition to snap each dataset's measured wind onto the regular
# (u_free_x, pitch) grid so the lookup table's RegularGridInterpolator sees clean axes.
NOMINAL_U_FREE_X = np.array([3.0, 5.0])                               # m/s
NOMINAL_PITCH = np.radians([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])    # rad


@dataclass
class PerConditionResult:
    """Fitted parameters, the wind condition, and the dataset they were fitted on."""
    params: np.ndarray   # shape (9,): [a_fx, b_fx, c_fx, ..., a_fz, b_fz, c_fz]
    u_free_x: float      # total background wind speed (m/s)
    pitch: float         # wind pitch angle relative to disk plane (rad)
    # Source dataset this fit came from — kept so the fit can be plotted against
    # the very data it was fitted on, without re-pairing results to datasets.
    dataset: Optional[data_factory.FittingDataset] = None


def _snap_to_nominal(value: float, nominal: np.ndarray, tol: float, name: str) -> float:
    """Snap a measured value to the nearest nominal grid value.

    Warns (but still snaps) when the nearest nominal is farther than tol, which
    flags a mislabeled or off-grid dataset instead of silently placing it in the
    wrong grid cell.
    """
    nominal = np.asarray(nominal, dtype=float)
    nearest = float(nominal[np.argmin(np.abs(nominal - value))])
    if abs(nearest - value) > tol:
        warnings.warn(
            f"{name}={value:.4f} is {abs(nearest - value):.4f} from the nearest "
            f"nominal {nearest:.4f}; snapping anyway (possible mislabel / off-grid data)."
        )
    return nearest


def extract_wind_condition(
    dataset: data_factory.FittingDataset,
    nominal_u_free_x: Optional[np.ndarray] = None,
    nominal_pitch: Optional[np.ndarray] = None,
    snap_tol_u: float = 1.0,
    snap_tol_pitch: float = np.radians(15.0),
) -> tuple[float, float]:
    """Return (u_free_x, pitch) from the mean background wind of a dataset.

    Both are derived from the inertial-frame background wind (dataset.u_free_0 is
    in the FLU inertial frame), NOT from the instantaneous disk attitude, so a
    maneuvering drone's tilt does not leak into the pitch label:
        u_free_x = |mean(u_free)|                  (frame-invariant magnitude)
        pitch    = atan2(u_z, hypot(u_x, u_y))     (angle above horizontal, [-90, 90] deg)

    If nominal_u_free_x / nominal_pitch are provided, the result is snapped to the
    nearest nominal grid value (with a tolerance warning) so repeated conditions
    collapse onto a clean regular grid for lookup-table interpolation.
    """
    u_free_mean = np.mean(dataset.u_free_0, axis=0)
    u_free_x = float(np.linalg.norm(u_free_mean))
    pitch = float(np.arctan2(u_free_mean[2], np.hypot(u_free_mean[0], u_free_mean[1])))

    if nominal_u_free_x is not None:
        u_free_x = _snap_to_nominal(u_free_x, nominal_u_free_x, snap_tol_u, "u_free_x")
    if nominal_pitch is not None:
        pitch = _snap_to_nominal(pitch, nominal_pitch, snap_tol_pitch, "pitch")
    return u_free_x, pitch
