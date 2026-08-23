import numpy as np

from inflow_model.bet import BladeElementTheory
import inflow_model.propeller_lookup_table as propeller_lookup_table
from drone import parameters
import data_factory


class BemtModel:
    """Physics model: encapsulates blade geometry, BEMT solver, and drone parameters."""

    PARAMETER_NAMES = ("cl_1", "cl_2", "cd", "alpha_0", "k_body_drag")

    BOUNDS = [
        (2.0, 50.0),                           # cl_1
        (0.0, 50.0),                           # cl_2
        (0.0, 5.0),                            # cd
        (np.radians(10), np.radians(40)),      # alpha_0
        (0.0, 10.0),                           # k_body_drag
    ]

    def __init__(self, blade, params):
        self.blade = blade
        self.bet_instance = BladeElementTheory(self.blade)
        self.params = params
        self.k_body_drag = 0.0  # kg/m, body aero drag coefficient along body-z from downwash
        self.sample_distance = None
        self.horizontal_weight = None
        self.adjust_resolution(False)

    def adjust_resolution(self, is_fine_tune):
        """When doing coarse search for optimization, sparse sample can save computation time. In other cases, dense sample is preferred."""
        if is_fine_tune:
            self.sample_distance = 20
            self.horizontal_weight = 1.0
            num_of_elements = 20
            num_of_rotation_segments = 18
        else:
            self.sample_distance = 100
            self.horizontal_weight = 100
            num_of_elements = 2
            num_of_rotation_segments = 6
        print(
            f"Sample distance: {self.sample_distance}, "
            f"Horizontal weight: {self.horizontal_weight:.3f}, "
            f"Num of elements: {num_of_elements}, "
            f"Num of rotation segments: {num_of_rotation_segments}"
        )
        self.bet_instance.set_integration_resolution(num_of_elements, num_of_rotation_segments)

    def compute_total_force_inertial_frame(self, dataset: data_factory.FittingDataset, i: int):
        f_0, v_i_0 = self.compute_model_thrust(dataset.u_free_0[i],
                                dataset.v_forward_0[i],
                                dataset.shared_r_disk[i],
                                dataset.omega_0[i],
                                self.params.is_ccw_blade[0])
        # torque_0 = np.cross(dataset.relative_position_inertial_frame_0[i], f_0)

        f_1, v_i_1 = self.compute_model_thrust(dataset.u_free_1[i],
                                dataset.v_forward_1[i],
                                dataset.shared_r_disk[i],
                                dataset.omega_1[i],
                                self.params.is_ccw_blade[1])
        # torque_1 = np.cross(dataset.relative_position_inertial_frame_1[i], f_1)

        f_2, v_i_2 = self.compute_model_thrust(dataset.u_free_2[i],
                                dataset.v_forward_2[i],
                                dataset.shared_r_disk[i],
                                dataset.omega_2[i],
                                self.params.is_ccw_blade[2])
        # torque_2 = np.cross(dataset.relative_position_inertial_frame_2[i], f_2)

        f_3, v_i_3 = self.compute_model_thrust(dataset.u_free_3[i],
                                dataset.v_forward_3[i],
                                dataset.shared_r_disk[i],
                                dataset.omega_3[i],
                                self.params.is_ccw_blade[3])
        # torque_3 = np.cross(dataset.relative_position_inertial_frame_3[i], f_3)

        f = f_0 + f_1 + f_2 + f_3
        f_inertial_frame = dataset.shared_r_disk[i]@f
        v_i_avg = (v_i_0 + v_i_1 + v_i_2 + v_i_3) / 4
        return f_inertial_frame, v_i_avg

    def compute_average_v_i(self, dataset: data_factory.FittingDataset, i: int, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        v_i_0 = BemtModel.compute_v_i_with_lookup_table(
                    dataset.u_free_0[i],
                    dataset.v_forward_0[i],
                    dataset.shared_r_disk[i],
                    dataset.omega_0[i],
                    self.params.is_ccw_blade[0],
                    lookup_table
                )

        v_i_1 = BemtModel.compute_v_i_with_lookup_table(
            dataset.u_free_1[i],
            dataset.v_forward_1[i],
            dataset.shared_r_disk[i],
            dataset.omega_1[i],
            self.params.is_ccw_blade[1],
            lookup_table
        )

        v_i_2 = BemtModel.compute_v_i_with_lookup_table(
            dataset.u_free_2[i],
            dataset.v_forward_2[i],
            dataset.shared_r_disk[i],
            dataset.omega_2[i],
            self.params.is_ccw_blade[2],
            lookup_table
        )

        v_i_3 = BemtModel.compute_v_i_with_lookup_table(
            dataset.u_free_3[i],
            dataset.v_forward_3[i],
            dataset.shared_r_disk[i],
            dataset.omega_3[i],
            self.params.is_ccw_blade[3],
            lookup_table
        )
        v_i_inertial_avg = (v_i_0 + v_i_1 + v_i_2 + v_i_3) / 4
        # lookup_table.get_rotor_forces returns v_i as a 3D inertial vector: v_i_inertial = -v_i_scalar * disk_z
        # project back to the scalar used by compute_body_drag_force: v_i_scalar = -(r_disk.T @ v_i_inertial)[2]
        r_disk = dataset.shared_r_disk[i]
        return -(r_disk.T @ v_i_inertial_avg)[2]

    def compute_total_force_inertial_frame_with_lookup_table(self, dataset: data_factory.FittingDataset, i: int, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        f_0 = BemtModel.compute_thrust_with_lookup_table(
            dataset.u_free_0[i],
            dataset.v_forward_0[i],
            dataset.shared_r_disk[i],
            dataset.omega_0[i],
            self.params.is_ccw_blade[0],
            lookup_table
        )

        f_1 = BemtModel.compute_thrust_with_lookup_table(
            dataset.u_free_1[i],
            dataset.v_forward_1[i],
            dataset.shared_r_disk[i],
            dataset.omega_1[i],
            self.params.is_ccw_blade[1],
            lookup_table
        )

        f_2 = BemtModel.compute_thrust_with_lookup_table(
            dataset.u_free_2[i],
            dataset.v_forward_2[i],
            dataset.shared_r_disk[i],
            dataset.omega_2[i],
            self.params.is_ccw_blade[2],
            lookup_table
        )

        f_3 = BemtModel.compute_thrust_with_lookup_table(
            dataset.u_free_3[i],
            dataset.v_forward_3[i],
            dataset.shared_r_disk[i],
            dataset.omega_3[i],
            self.params.is_ccw_blade[3],
            lookup_table
        )

        f_inertial_frame = f_0 + f_1 + f_2 + f_3
        return f_inertial_frame

    def compute_body_drag_force(self, v_i_avg: float, u_free_avg: np.ndarray, r_disk: np.ndarray) -> np.ndarray:
        """Body aero drag along body-z from downwash hitting the drone frame.

        Physical model: F_drag = k_body_drag * (v_i_avg + u_background_axial)^2 in the body-z direction.
        The force acts downward (in -disk_z direction), so the propellers must produce extra thrust to
        compensate. Returns the compensation term to add to f_gt (in inertial frame, along +disk_z).

        Args:
            v_i_avg: disk-area-weighted average induced velocity, positive = downwash (m/s)
            u_free_avg: background free-stream wind averaged over 4 rotors, inertial frame (m/s)
            r_disk: disk-to-inertial rotation matrix (columns = disk axes in inertial frame)
        """
        # axial wind component in downwash direction (-disk_z), positive = augments downwash
        u_background_axial = -(r_disk.T @ u_free_avg)[2]
        v_total = v_i_avg + u_background_axial
        disk_z_inertial = r_disk[:, 2]  # disk z-axis in inertial frame (thrust direction)
        return self.k_body_drag * v_total**2 * disk_z_inertial

    def compute_residual_force(self, f_inertial_frame, a_groundtruth, v_i_avg=0.0, u_free_avg=None, r_disk=None):
        """Compute the residual force between the model thrust and the ground truth thrust.
        Assumes a_groundtruth is in the inertial frame (FLU).
        The residual is in the inertial frame (FLU).
        """
        f_gt_inertial_frame = -self.params.m * parameters.Environment.g*np.array([0.0, 0.0, -1.0]) + self.params.m * a_groundtruth
        if self.k_body_drag != 0.0 and u_free_avg is not None and r_disk is not None:
            f_gt_inertial_frame += self.compute_body_drag_force(v_i_avg, u_free_avg, r_disk)
        f_residual = f_inertial_frame - f_gt_inertial_frame
        return f_residual

    def get_residual_force(self, dataset: data_factory.FittingDataset, i: int, lookup_table=None, is_using_lookup_table=False, is_in_body_frame=False):
        """The residual is in the inertial frame (FLU)."""
        if is_using_lookup_table:
            f_inertial_frame = self.compute_total_force_inertial_frame_with_lookup_table(dataset, i, lookup_table)
            v_i_avg = self.compute_average_v_i(dataset, i, lookup_table)
            u_free_avg = (dataset.u_free_0[i] + dataset.u_free_1[i] + dataset.u_free_2[i] + dataset.u_free_3[i]) / 4
            r_disk = dataset.shared_r_disk[i]
        else:
            f_inertial_frame, v_i_avg = self.compute_total_force_inertial_frame(dataset, i)
            u_free_avg = (dataset.u_free_0[i] + dataset.u_free_1[i] + dataset.u_free_2[i] + dataset.u_free_3[i]) / 4
            r_disk = dataset.shared_r_disk[i]
        f_residual = self.compute_residual_force(f_inertial_frame, dataset.dv[i], v_i_avg, u_free_avg, r_disk)
        if is_in_body_frame:
            f_residual = dataset.shared_r_disk[i].T @ f_residual
        return f_residual

    def compute_model_thrust(self, u_free, v_forward, r_disk, omega, is_ccw_blade):
        if is_ccw_blade:
            f_x, f_y, f_z, v_i = self.bet_instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        else:
            f_x, f_y, f_z, v_i = self.bet_instance.get_rotor_forces(u_free, v_forward, r_disk, -omega, is_ccw_blade) # negative omega for CW rotation, this is an interface mismatch
        return np.array([f_x, f_y, f_z]), v_i

    @staticmethod
    def compute_thrust_with_lookup_table(u_free, v_forward, r_disk, omega, is_ccw_blade, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        # The output is in inertial frame
        thrust, _ = lookup_table.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        return thrust

    @staticmethod
    def compute_v_i_with_lookup_table(u_free, v_forward, r_disk, omega, is_ccw_blade, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        # The output is in inertial frame
        _, v_i = lookup_table.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        return v_i

    def compute_ground_truth(self, dataset):
        f_total_inertial_frame_gt = [
            (-self.params.m * parameters.Environment.g * np.array([0.0, 0.0, -1.0]) + self.params.m * dataset.dv[i])
            for i in range(len(dataset.dv))
        ]
        f_0_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_0_f_rotor_inertial_frame[i]
            for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
        ]
        f_1_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_1_f_rotor_inertial_frame[i]
            for i in range(len(dataset.rotor_1_f_rotor_inertial_frame))
        ]
        f_2_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_2_f_rotor_inertial_frame[i]
            for i in range(len(dataset.rotor_2_f_rotor_inertial_frame))
        ]
        f_3_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_3_f_rotor_inertial_frame[i]
            for i in range(len(dataset.rotor_3_f_rotor_inertial_frame))
        ]
        f_total_from_rotor_inertial_frame_gt = [
            (dataset.rotor_0_f_rotor_inertial_frame[i] + dataset.rotor_1_f_rotor_inertial_frame[i] +
             dataset.rotor_2_f_rotor_inertial_frame[i] + dataset.rotor_3_f_rotor_inertial_frame[i])
            for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
        ]
        return f_total_inertial_frame_gt, f_0_body_frame_gt, f_1_body_frame_gt, f_2_body_frame_gt, f_3_body_frame_gt, f_total_from_rotor_inertial_frame_gt
