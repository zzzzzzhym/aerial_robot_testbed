"""Plotting utilities for the quadratic per-condition fit. Run after fitting."""

import numpy as np
import matplotlib.pyplot as plt

from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.per_condition_result import PerConditionResult


class QuadraticFitPlotter:
    """Plots the quadratic fit against the measured data, one figure per condition."""

    # component index into predict_forces_disk: [f_x, f_y, f_z]
    _COMPONENT_LABELS = ("fx", "fy", "fz")

    @staticmethod
    def _measured_disk_force(dataset, component: int, sample_step: int):
        """Measured disk-frame force component and omega for a dataset.

        Matches exactly what QuadraticSolver fits against: r_disk.T @ f_inertial.
        """
        idx = range(0, len(dataset), sample_step)
        omega = np.array([dataset.omega_0[i] for i in idx])
        force = np.array([
            (dataset.shared_r_disk[i].T @ dataset.rotor_0_f_rotor_inertial_frame[i])[component]
            for i in idx
        ])
        return omega, force

    @staticmethod
    def plot_fit_comparison(results: list[PerConditionResult],
                            component: int = 2,
                            n_curve: int = 300,
                            sample_step: int = 1):
        """Plot each fitted quadratic against the measured data it was fitted on.

        Generates one figure per wind condition (not all overlaid), traversing
        `results` without assuming how many conditions there are. Each figure also
        overlays the zero-wind condition's fit as a dashed reference line, so the
        effect of wind on the thrust curve is visible against the no-wind baseline.

        Args:
            results: list of PerConditionResult from FittingManager.run_quadratic_fit();
                     each must carry its source `dataset` (the engine sets this).
            component: force component to plot (0=fx, 1=fy, 2=fz thrust).
            n_curve: number of points used to draw each fitted curve.
            sample_step: stride for thinning measured samples.

        Returns:
            list of matplotlib Figures, one per condition (same order as results).
        """
        label = QuadraticFitPlotter._COMPONENT_LABELS[component]
        model = QuadraticRotorModel()
        figs = []

        # Zero-wind condition used as the dashed reference in every figure.
        zero_result = min(results, key=lambda r: abs(r.u_free_x)) if results else None
        has_zero = zero_result is not None and abs(zero_result.u_free_x) < 0.5
        zero_model = QuadraticRotorModel()
        if has_zero:
            zero_model.apply_params(zero_result.params)

        for result in results:
            if result.dataset is None:
                raise ValueError(
                    "plot_fit_comparison needs result.dataset; use results from "
                    "FittingManager.run_quadratic_fit() (the engine attaches it)."
                )
            omega_meas, f_meas = QuadraticFitPlotter._measured_disk_force(
                result.dataset, component, sample_step)

            model.apply_params(result.params)
            omega_curve = np.linspace(float(omega_meas.min()), float(omega_meas.max()), n_curve)
            f_curve = [model.predict_forces_disk(w)[component] for w in omega_curve]

            fig, ax = plt.subplots()
            ax.plot(omega_meas, f_meas, linestyle="None", marker=".", label="measured")
            ax.plot(omega_curve, f_curve, label="quadratic fit")
            # Dashed reference: the zero-wind fit evaluated over the same omega range.
            if has_zero and result is not zero_result:
                f_curve_zero = [zero_model.predict_forces_disk(w)[component] for w in omega_curve]
                ax.plot(omega_curve, f_curve_zero, linestyle="--", label="fit @ zero wind")
            ax.set_xlabel("omega (rad/s)")
            ax.set_ylabel(f"Force {label} (N)")
            ax.set_title(f"u_free_x={result.u_free_x:.3f} m/s  "
                         f"pitch={np.degrees(result.pitch):.1f} deg")
            ax.legend()
            fig.tight_layout()
            figs.append(fig)

        return figs
