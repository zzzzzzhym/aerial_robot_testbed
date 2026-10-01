"""Quadratic propeller model: F(omega) = a*omega^2 + b*omega + c per force component."""

import numpy as np

import data_factory


class QuadraticRotorModel:
    """Quadratic force-vs-omega model for one wind condition.

    Parameter vector (9): [a_fx, b_fx, c_fx,  a_fy, b_fy, c_fy,  a_fz, b_fz, c_fz]
    where F_k(omega) = a_k * omega^2 + b_k * omega + c_k,  forces in disk frame.
    """

    PARAMETER_NAMES = (
        "a_fx", "b_fx", "c_fx",
        "a_fy", "b_fy", "c_fy",
        "a_fz", "b_fz", "c_fz",
    )

    def __init__(self):
        self._coeffs = np.zeros((3, 3))   # shape (n_components=3, n_coeffs=3)

    def apply_params(self, x: np.ndarray) -> None:
        self._coeffs = np.asarray(x, dtype=float).reshape(3, 3)

    def predict_forces_disk(self, omega: float) -> np.ndarray:
        """Return [f_x, f_y, f_z] in disk frame at given omega (uses current params)."""
        v = np.array([omega ** 2, omega, 1.0])
        return self._coeffs @ v

    def get_residual_force(self, dataset: data_factory.FittingDataset, i: int) -> np.ndarray:
        """Residual = f_predicted_disk - f_measured_disk at sample i."""
        r_disk = dataset.shared_r_disk[i]
        f_measured_disk = r_disk.T @ dataset.rotor_0_f_rotor_inertial_frame[i]
        return self.predict_forces_disk(dataset.omega_0[i]) - f_measured_disk
