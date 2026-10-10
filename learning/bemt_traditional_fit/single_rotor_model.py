import numpy as np

from inflow_model.bet import BladeElementTheory
import data_factory
from learning.bemt_traditional_fit.fitting_config import ModelConfig


class SingleRotorBemtModel:
    """BET model for single-rotor (test stand) fitting.
    Fits cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min on rotor 0 only.

    Uses rotor_0_sensed_wind_velocity (background wind + induced velocity,
    in inertial frame) directly as the flow at the rotor disk.  No separate
    v_i computation is needed: projecting the sensed wind to disk frame gives
    v_flow_disk_frame exactly.
    """

    def __init__(self, blade, is_ccw_rotor0: bool, model_config: ModelConfig):
        self.blade = blade
        self.is_ccw_rotor0 = is_ccw_rotor0
        self.bet_instance = BladeElementTheory(self.blade)
        self.model_config = model_config
        self.use_bemt = False
        self.sample_distance = None
        self.adjust_resolution(is_fine_tune=False)

    def apply_params(self, x):
        """Apply physical coefficients (cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min)."""
        (self.blade.cl_1, self.blade.cl_2, self.blade.cd, self.blade.alpha_0,
         self.blade.alpha_zero_lift, self.blade.alpha_d_min) = x
        self.bet_instance.refresh_blade()

    def adjust_resolution(self, is_fine_tune: bool):
        if is_fine_tune:
            self.sample_distance = self.model_config.fine_sample_distance
            num_of_elements = self.model_config.fine_n_elements
            num_of_rotation_segments = self.model_config.fine_n_rotation_segments
        else:
            self.sample_distance = self.model_config.coarse_sample_distance
            num_of_elements = self.model_config.coarse_n_elements
            num_of_rotation_segments = self.model_config.coarse_n_rotation_segments
        print(
            f"Sample distance: {self.sample_distance}, "
            f"Num of elements: {num_of_elements}, "
            f"Num of rotation segments: {num_of_rotation_segments}"
        )
        self.bet_instance.set_integration_resolution(num_of_elements, num_of_rotation_segments)

    def compute_rotor0_thrust(self, u_sensed: np.ndarray, r_disk: np.ndarray,
                               omega: float) -> np.ndarray:
        """BET-predicted force for rotor 0 in disk frame.

        The sensed wind (u_sensed) is the total local wind at the rotor in the
        inertial frame, including induced velocity.  Passing it as u_free with
        v_i=0 and v_forward=zeros to integrate_element_force yields:
            v_flow_disk_frame = r_disk.T @ u_sensed
        which is the correct total flow in disk coordinates.

        Args:
            u_sensed: rotor_0_sensed_wind_velocity — total wind at rotor 0 (inertial frame, m/s)
            r_disk: disk-to-inertial rotation matrix
            omega: absolute rotor speed (rad/s, always positive)

        Returns:
            force in disk frame (N)
        """
        omega_signed = omega if self.is_ccw_rotor0 else -omega
        return self.bet_instance.integrate_element_force(
            u_sensed, 0.0, np.zeros(3), r_disk, omega_signed,
            is_ccw_blade=self.is_ccw_rotor0,
        )

    def compute_rotor0_thrust_bemt(self, u_free: np.ndarray, v_forward: np.ndarray,
                                   r_disk: np.ndarray, omega: float) -> np.ndarray:
        """BEMT-predicted force for rotor 0 in disk frame.

        Uses momentum-theory root-finding per blade element to solve v_i from the
        background free-stream wind u_free (induced velocity not pre-embedded).

        Args:
            u_free: rotor_0_local_wind_velocity — background wind (inertial frame, m/s)
            v_forward: rotor_0_velocity — disk velocity in inertial frame (m/s)
            r_disk: disk-to-inertial rotation matrix
            omega: absolute rotor speed (rad/s, always positive)

        Returns:
            force in disk frame (N)
        """
        omega_signed = omega if self.is_ccw_rotor0 else -omega
        f_x, f_y, f_z, _ = self.bet_instance.get_rotor_forces(
            u_free, v_forward, r_disk, omega_signed, is_ccw_blade=self.is_ccw_rotor0,
        )
        return np.array([f_x, f_y, f_z])

    def get_residual_force(self, dataset: data_factory.FittingDataset, i: int) -> np.ndarray:
        """Residual = f_predicted_disk - f_sensor_disk for rotor 0 at sample i.

        When use_bemt=False (default): BET with rotor_0_sensed_wind_velocity.
        When use_bemt=True: BEMT with root-found v_i from rotor_0_local_wind_velocity.
        """
        r_disk = dataset.shared_r_disk[i]
        omega = dataset.omega_0[i]
        if self.use_bemt:
            f_predicted_disk = self.compute_rotor0_thrust_bemt(
                dataset.u_free_0[i],
                dataset.v_forward_0[i],
                r_disk,
                omega,
            )
        else:
            f_predicted_disk = self.compute_rotor0_thrust(
                dataset.rotor_0_sensed_wind_velocity[i],
                r_disk,
                omega,
            )
        f_measured_disk = r_disk.T @ dataset.rotor_0_f_rotor_inertial_frame[i]
        return f_predicted_disk - f_measured_disk
