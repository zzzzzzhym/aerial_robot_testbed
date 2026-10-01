"""Closed-form least-squares solver for QuadraticRotorModel."""

import numpy as np

import data_factory
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel


class QuadraticSolver:
    """For each force component k in {fx, fy, fz}, solves the linear system
        [omega_i^2  omega_i  1] @ [a_k, b_k, c_k]^T = F_k_i
    via np.linalg.lstsq.  One call gives the exact global minimum — no
    iterations, no bounds, no objective function needed.
    """

    def run(self, model: QuadraticRotorModel,
            dataset: data_factory.FittingDataset) -> np.ndarray:
        """Fit model to dataset analytically and store the result in model.

        Returns:
            np.ndarray of shape (9,): fitted [a_fx, b_fx, c_fx, ..., a_fz, b_fz, c_fz].
        """
        omegas = dataset.omega_0
        forces_disk = np.array([
            dataset.shared_r_disk[i].T @ dataset.rotor_0_f_rotor_inertial_frame[i]
            for i in range(len(dataset))
        ])

        A = np.column_stack([omegas ** 2, omegas, np.ones_like(omegas)])

        params = np.zeros(9)
        for k in range(3):
            coeffs, _, _, _ = np.linalg.lstsq(A, forces_disk[:, k], rcond=None)
            params[3 * k: 3 * k + 3] = coeffs

        model.apply_params(params)
        print(f"  a_fz={params[6]:.4e}  b_fz={params[7]:.4e}  c_fz={params[8]:.4e}")
        return params
