"""Plotting utilities for the quadratic per-condition fit. Run after fitting."""

import numpy as np
import matplotlib.pyplot as plt

from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.per_condition_result import PerConditionResult


class QuadraticFitPlotter:
    """Plots fitted quadratic profiles, one line per wind condition."""

    # component index into predict_forces_disk: [f_x, f_y, f_z]
    _COMPONENT_LABELS = ("fx", "fy", "fz")

    @staticmethod
    def plot_force_profiles(results: list[PerConditionResult],
                            omega_range: np.ndarray = None,
                            component: int = 2,
                            ax=None):
        """Plot fitted force-vs-omega profiles, one line per wind condition.

        Traverses `results` without assuming how many conditions there are.

        Args:
            results: list of PerConditionResult from FittingManager.run_quadratic_fit();
                     each carries its fitted params and wind condition.
            omega_range: rotor speeds (rad/s) to evaluate; default linspace(0, 1500, 300).
            component: force component to plot (0=fx, 1=fy, 2=fz thrust).
            ax: optional matplotlib Axes to draw on; one is created if None.

        Returns:
            The matplotlib Axes.
        """
        if omega_range is None:
            omega_range = np.linspace(0, 1500, 300)
        if ax is None:
            _, ax = plt.subplots()

        model = QuadraticRotorModel()
        for result in results:
            model.apply_params(result.params)
            force = [model.predict_forces_disk(w)[component] for w in omega_range]
            ax.plot(omega_range, force,
                    label=f"u={result.u_free_x:.1f} m/s, pitch={np.degrees(result.pitch):.0f}°")

        label = QuadraticFitPlotter._COMPONENT_LABELS[component]
        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel(f"Force {label} (N)")
        ax.set_title("Fitted quadratic force profiles per wind condition")
        if len(results) > 0:
            ax.legend()
        return ax

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
                            sample_step: int = 1,
                            ax=None):
        """Overlay each fitted quadratic against the measured data it was fitted on.

        For every condition: measured samples are drawn as markers and the fitted
        quadratic as a line in the same colour.  Traverses `results` without
        assuming how many conditions there are.

        Args:
            results: list of PerConditionResult from FittingManager.run_quadratic_fit();
                     each must carry its source `dataset` (the engine sets this).
            component: force component to plot (0=fx, 1=fy, 2=fz thrust).
            n_curve: number of points used to draw each fitted curve.
            sample_step: stride for thinning measured samples.
            ax: optional matplotlib Axes to draw on; one is created if None.

        Returns:
            The matplotlib Axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        model = QuadraticRotorModel()
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

            cond_label = f"u={result.u_free_x:.1f} m/s, pitch={np.degrees(result.pitch):.0f}°"
            line, = ax.plot(omega_curve, f_curve, label=f"{cond_label} (fit)")
            ax.plot(omega_meas, f_meas, linestyle="None", marker=".",
                    color=line.get_color(), label=f"{cond_label} (measured)")

        label = QuadraticFitPlotter._COMPONENT_LABELS[component]
        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel(f"Force {label} (N)")
        ax.set_title("Quadratic fit vs measured data per wind condition")
        if len(results) > 0:
            ax.legend()
        return ax
