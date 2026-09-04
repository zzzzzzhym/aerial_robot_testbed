import matplotlib.pyplot as plt

import data_factory
from learning.bemt_traditional_fit.bemt_model import BemtModel


class FitPlotter:
    """Analysis and plotting utilities. Run separately after fitting is complete."""

    @staticmethod
    def plot_the_fit(model: BemtModel, dataset: data_factory.FittingDataset,
                     lookup_table=None, is_using_lookup_table: bool = False,
                     sample_step: int = 1):
        """Plot model fit vs ground truth forces.

        Args:
            model: fitted BemtModel (blade params already set to fitted values)
            dataset: FittingDataset to evaluate
            lookup_table: optional lookup table for force computation
            is_using_lookup_table: use lookup table path if True, BET path if False
            sample_step: stride for selecting samples to plot
        """
        data_len = len(dataset.u_free_0)
        sample_indices = list(range(0, data_len, sample_step))

        fitted_total = []
        fitted_f0, fitted_f1, fitted_f2, fitted_f3 = [], [], [], []

        for i in sample_indices:
            r_disk = dataset.shared_r_disk[i]
            if is_using_lookup_table:
                f_total_inertial = model.compute_total_force_inertial_frame_with_lookup_table(dataset, i, lookup_table)
                # Per-rotor forces in body frame (convert from inertial)
                f0 = r_disk.T @ BemtModel.compute_thrust_with_lookup_table(dataset.u_free_0[i], dataset.v_forward_0[i], r_disk, dataset.omega_0[i], model.params.is_ccw_blade[0], lookup_table)
                f1 = r_disk.T @ BemtModel.compute_thrust_with_lookup_table(dataset.u_free_1[i], dataset.v_forward_1[i], r_disk, dataset.omega_1[i], model.params.is_ccw_blade[1], lookup_table)
                f2 = r_disk.T @ BemtModel.compute_thrust_with_lookup_table(dataset.u_free_2[i], dataset.v_forward_2[i], r_disk, dataset.omega_2[i], model.params.is_ccw_blade[2], lookup_table)
                f3 = r_disk.T @ BemtModel.compute_thrust_with_lookup_table(dataset.u_free_3[i], dataset.v_forward_3[i], r_disk, dataset.omega_3[i], model.params.is_ccw_blade[3], lookup_table)
            else:
                f_total_inertial, _ = model.compute_total_force_inertial_frame(dataset, i)
                # Per-rotor forces in body frame (directly from BET)
                f0, _ = model.compute_model_thrust(dataset.u_free_0[i], dataset.v_forward_0[i], r_disk, dataset.omega_0[i], model.params.is_ccw_blade[0])
                f1, _ = model.compute_model_thrust(dataset.u_free_1[i], dataset.v_forward_1[i], r_disk, dataset.omega_1[i], model.params.is_ccw_blade[1])
                f2, _ = model.compute_model_thrust(dataset.u_free_2[i], dataset.v_forward_2[i], r_disk, dataset.omega_2[i], model.params.is_ccw_blade[2])
                f3, _ = model.compute_model_thrust(dataset.u_free_3[i], dataset.v_forward_3[i], r_disk, dataset.omega_3[i], model.params.is_ccw_blade[3])

            fitted_total.append(f_total_inertial)
            fitted_f0.append(f0)
            fitted_f1.append(f1)
            fitted_f2.append(f2)
            fitted_f3.append(f3)

        (f_total_gt, f_0_gt, f_1_gt, f_2_gt, f_3_gt,
         f_total_rotor_gt) = model.compute_ground_truth(dataset)

        fig0, axs0 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        force_labels = ["Force X", "Force Y", "Force Z"]
        for j in range(3):
            axs0[j].plot(sample_indices, [f[j] for f in fitted_total],
                         label=f'Fitted F{["x","y","z"][j]}', linestyle='None', marker='.')
            axs0[j].plot([f[j] for f in f_total_gt],
                         label=f'GT F{["x","y","z"][j]}', linestyle='-', marker='.')
            axs0[j].plot([f[j] for f in f_total_rotor_gt],
                         label=f'GT from Rotor F{["x","y","z"][j]}', linestyle='-', marker='.')
            axs0[j].set_ylabel(force_labels[j])
            axs0[j].legend()
        axs0[2].set_xlabel("Sample Index")
        fig0.tight_layout()

        fig1, axs1 = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
        rotor_data = [
            ('F0', fitted_f0, f_0_gt),
            ('F1', fitted_f1, f_1_gt),
            ('F2', fitted_f2, f_2_gt),
            ('F3', fitted_f3, f_3_gt),
        ]
        for row, (label, fitted, gt) in enumerate(rotor_data):
            for j in range(3):
                axs1[row, j].plot(sample_indices, [f[j] for f in fitted],
                                  label=f'Fitted {label}', linestyle='None', marker='.')
                axs1[row, j].plot([f[j] for f in gt], label=f'GT {label}', linestyle='-')
                axs1[row, j].set_ylabel(f"{label} {['X', 'Y', 'Z'][j]}")
        for ax in axs1.flat:
            ax.legend()

        return fig0, fig1

    @staticmethod
    def plot_single_rotor_fit(model, dataset: data_factory.FittingDataset, sample_step: int = 1):
        """Plot BET-predicted vs measured force for rotor 0 in disk frame."""
        data_len = len(dataset.rotor_0_sensed_wind_velocity)
        sample_indices = list(range(0, data_len, sample_step))

        predicted = []
        measured = []
        for i in sample_indices:
            r_disk = dataset.shared_r_disk[i]
            f_pred = model.compute_rotor0_thrust(
                dataset.rotor_0_sensed_wind_velocity[i],
                r_disk,
                dataset.omega_0[i],
            )
            f_meas = r_disk.T @ dataset.rotor_0_f_rotor_inertial_frame[i]
            predicted.append(f_pred)
            measured.append(f_meas)

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ["Force X (disk)", "Force Y (disk)", "Force Z (disk)"]
        for j in range(3):
            axs[j].plot(sample_indices, [f[j] for f in predicted],
                        label="Predicted", linestyle="None", marker=".")
            axs[j].plot(sample_indices, [f[j] for f in measured],
                        label="Measured", linestyle="-", marker=".")
            axs[j].set_ylabel(labels[j])
            axs[j].legend()
        axs[2].set_xlabel("Sample Index")
        fig.tight_layout()
        return fig
