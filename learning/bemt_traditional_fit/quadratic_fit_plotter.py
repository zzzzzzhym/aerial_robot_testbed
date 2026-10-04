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
        `results` without assuming how many conditions there are.

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
            ax.set_xlabel("omega (rad/s)")
            ax.set_ylabel(f"Force {label} (N)")
            ax.set_title(f"u_free_x={result.u_free_x:.3f} m/s  "
                         f"pitch={np.degrees(result.pitch):.1f} deg")
            ax.legend()
            fig.tight_layout()
            figs.append(fig)

        return figs

    @staticmethod
    def plot_fit_comparison_overlay(results: list[PerConditionResult],
                                    component: int = 2,
                                    n_curve: int = 300,
                                    sample_step: int = 1,
                                    show_measured: bool = True):
        """Overlay every condition's fitted quadratic on a single figure.

        Same data as plot_fit_comparison, but stacked into one axes so the
        conditions can be compared directly. Each condition gets its own color;
        the measured samples (if shown) share the color of their fit curve.

        Args:
            results: list of PerConditionResult from FittingManager.run_quadratic_fit().
            component: force component to plot (0=fx, 1=fy, 2=fz thrust).
            n_curve: number of points used to draw each fitted curve.
            sample_step: stride for thinning measured samples.
            show_measured: also scatter the measured samples behind each fit curve.

        Returns:
            a single matplotlib Figure with all conditions overlaid.
        """
        label = QuadraticFitPlotter._COMPONENT_LABELS[component]
        model = QuadraticRotorModel()
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        fig, ax = plt.subplots()
        for result, color in zip(results, colors):
            if result.dataset is None:
                raise ValueError(
                    "plot_fit_comparison_overlay needs result.dataset; use results from "
                    "FittingManager.run_quadratic_fit() (the engine attaches it)."
                )
            omega_meas, f_meas = QuadraticFitPlotter._measured_disk_force(
                result.dataset, component, sample_step)

            model.apply_params(result.params)
            omega_curve = np.linspace(float(omega_meas.min()), float(omega_meas.max()), n_curve)
            f_curve = [model.predict_forces_disk(w)[component] for w in omega_curve]

            condition_label = (f"u={result.u_free_x:.1f} m/s, "
                               f"pitch={np.degrees(result.pitch):.0f} deg")
            if show_measured:
                ax.plot(omega_meas, f_meas, linestyle="None", marker=".",
                        color=color, alpha=0.4)
            ax.plot(omega_curve, f_curve, color=color, label=condition_label)

        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel(f"Force {label} (N)")
        ax.set_title(f"Quadratic fit {label} — all conditions")
        ax.legend(fontsize="small")
        fig.tight_layout()
        return fig
