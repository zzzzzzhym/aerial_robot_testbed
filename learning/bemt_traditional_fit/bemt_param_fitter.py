import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pandas as pd


from inflow_model.bet import BladeElementTheory
from inflow_model.blade_params import Blade
from inflow_model.propeller_lookup_table import PropellerLookupTable
import inflow_model.blade_params
import inflow_model.propeller_lookup_table as propeller_lookup_table
from drone import utils
from drone import parameters
import data_factory

class FittedBlade(Blade):
    """
        Gill, Rajan, and Raffaello D'andrea. "Propeller thrust and drag in forward flight." 
        2017 IEEE Conference on control technology and applications (CCTA). IEEE, 2017.
    """  
    def __init__(self):
        super().__init__(num_of_blades=2, y_max=0.1, cl_1=5.3, cl_2=1.7, alpha_0=np.radians(20.6), cd=1.8, cd_0=0.01)
        x_table = [0.0, 0.3, 1.0]
        y_table = [0.8, 1.0, 0.0]
        self.interp_func = interp1d(x_table, y_table, kind="linear")
    
    def get_chord(self, y: float):
        """Get the chord length at a given y position along the blade."""
        min_length = 0.005
        max_length = 0.024
        weight = self.interp_func(y/self.y_max)
        length = (1 - weight)*min_length + weight*max_length
        return length

    def get_blade_pitch(self, y: float):
        min_pitch = np.radians(17)
        max_pitch = np.radians(45)
        pitch = self.linear_interpolate(max_pitch, min_pitch, y)
        return pitch
    
    def linear_interpolate(self, val_root, val_tip, y):
        weight = y/self.y_max
        result = (1 - weight)*val_root + weight*val_tip
        return result

class BemtParamFitter:
    def __init__(self):
        self.blade = FittedBlade()
        self.bet_instance = BladeElementTheory(self.blade)
        self.params = parameters.PennStateARILab550()
        self.logger = {}
        self.sample_distance = None
        self.horizontal_weight = None
        self.fitted_params = None
        self.can_log_and_display_data = False
        self.adjust_resolution(False)

    def initialize_logger(self):
        self.logger = {
            "sample_index": [], 
            "f_0": [], 
            "f_1": [], 
            "f_2": [], 
            "f_3": [], 
            "f_total": [], 
            "f_inertial_frame": []
        }

    def make_lookup_table(self, fitted_params, table_name: str, is_hover_only: bool = False):
        blade = inflow_model.blade_params.APC_8x6()
        blade.cl_1, blade.cl_2, blade.cd, blade.alpha_0 = fitted_params  # fitted parameters
        print(
            "Making lookup table with parameters:\n"
            f"cl_1 = {blade.cl_1}\n"
            f"cl_2 = {blade.cl_2}\n"
            f"cd = {blade.cd}\n"
            f"alpha_0 = {blade.alpha_0}"
        )
        if is_hover_only:
            PropellerLookupTable.Maker.make_propeller_lookup_table(table_name, blade, u_free_x_range=np.array([0.0]), pitch_range=np.array([0.0]))
        else:
            PropellerLookupTable.Maker.make_propeller_lookup_table(table_name, blade)

    def adjust_resolution(self, is_fine_tune):
        """When doing coarse search for optimization, sparse sample can save computation time. In other cases, dense sample is preferred."""
        if is_fine_tune:
            self.sample_distance = 20
            self.horizontal_weight = 1.0
            num_of_elements = 20
            num_of_rotation_segments = 18
            # self.sample_distance = 500
            # self.horizontal_weight = 1.0
            # num_of_elements = 100
            # num_of_rotation_segments = 90
        else:
            self.sample_distance = 100
            self.horizontal_weight = 100        
            num_of_elements = 5
            num_of_rotation_segments = 6
        print(
            f"Sample distance: {self.sample_distance}, "
            f"Horizontal weight: {self.horizontal_weight:.3f}, "
            f"Num of elements: {num_of_elements}, "
            f"Num of rotation segments: {num_of_rotation_segments}"
        )
        self.bet_instance.set_integration_resolution(num_of_elements, num_of_rotation_segments)

    def compute_total_force_inertial_frame(self, dataset: data_factory.FittingDataset, i: int):
        f_0 = self.compute_model_thrust(dataset.u_free_0[i], 
                                dataset.v_forward_0[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_0[i], 
                                self.params.is_ccw_blade[0])
        # torque_0 = np.cross(dataset.relative_position_inertial_frame_0[i], f_0)

        f_1 = self.compute_model_thrust(dataset.u_free_1[i], 
                                dataset.v_forward_1[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_1[i], 
                                self.params.is_ccw_blade[1])
        # torque_1 = np.cross(dataset.relative_position_inertial_frame_1[i], f_1)

        f_2 = self.compute_model_thrust(dataset.u_free_2[i], 
                                dataset.v_forward_2[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_2[i], 
                                self.params.is_ccw_blade[2])
        # torque_2 = np.cross(dataset.relative_position_inertial_frame_2[i], f_2)

        f_3 = self.compute_model_thrust(dataset.u_free_3[i], 
                                dataset.v_forward_3[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_3[i], 
                                self.params.is_ccw_blade[3])
        # torque_3 = np.cross(dataset.relative_position_inertial_frame_3[i], f_3)

        f = f_0 + f_1 + f_2 + f_3
        f_inertial_frame = dataset.shared_r_disk[i]@f

        if self.can_log_and_display_data:
            self.logger["sample_index"].append(i)
            self.logger["f_0"].append(f_0)
            self.logger["f_1"].append(f_1)
            self.logger["f_2"].append(f_2)
            self.logger["f_3"].append(f_3)
            self.logger["f_total"].append(f)
            self.logger["f_inertial_frame"].append(f_inertial_frame)
        return f_inertial_frame

    def compute_total_force_inertial_frame_with_lookup_table(self, dataset: data_factory.FittingDataset, i: int, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        f_0 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_0[i],
            dataset.v_forward_0[i],
            dataset.shared_r_disk[i],
            dataset.omega_0[i],
            self.params.is_ccw_blade[0],
            lookup_table
        )

        f_1 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_1[i],
            dataset.v_forward_1[i],
            dataset.shared_r_disk[i],
            dataset.omega_1[i],
            self.params.is_ccw_blade[1],
            lookup_table
        )

        f_2 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_2[i],
            dataset.v_forward_2[i],
            dataset.shared_r_disk[i],
            dataset.omega_2[i],
            self.params.is_ccw_blade[2],
            lookup_table
        )

        f_3 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_3[i],
            dataset.v_forward_3[i],
            dataset.shared_r_disk[i],
            dataset.omega_3[i],
            self.params.is_ccw_blade[3],
            lookup_table
        )

        f_inertial_frame = f_0 + f_1 + f_2 + f_3


        if self.can_log_and_display_data:
            # convert to body frame for comparison
            f_0 = dataset.shared_r_disk[i].T@f_0
            f_1 = dataset.shared_r_disk[i].T@f_1
            f_2 = dataset.shared_r_disk[i].T@f_2
            f_3 = dataset.shared_r_disk[i].T@f_3
            f = f_0 + f_1 + f_2 + f_3

            self.logger["sample_index"].append(i)
            self.logger["f_0"].append(f_0)
            self.logger["f_1"].append(f_1)
            self.logger["f_2"].append(f_2)
            self.logger["f_3"].append(f_3)
            self.logger["f_total"].append(f)
            self.logger["f_inertial_frame"].append(f_inertial_frame)

        return f_inertial_frame

    def compute_residual_force(self, f_inertial_frame, a_groundtruth):
        """Compute the residual force between the model thrust and the ground truth thrust.
        Assumes a_groundtruth is in the inertial frame (FLU).
        The residual is in the inertial frame (FLU).
        """
        f_gt_inertial_frame = -self.params.m * parameters.Environment.g*np.array([0.0, 0.0, -1.0]) + self.params.m * a_groundtruth
        f_residual = f_inertial_frame - f_gt_inertial_frame
        return f_residual

    def get_residual_force(self, dataset: data_factory.FittingDataset, i: int, lookup_table=None, is_using_lookup_table=False, is_in_body_frame=False):
        """The residual is in the inertial frame (FLU)."""
        if is_using_lookup_table:
            f_inertial_frame = self.compute_total_force_inertial_frame_with_lookup_table(dataset, i, lookup_table)
        else:
            f_inertial_frame = self.compute_total_force_inertial_frame(dataset, i)
        f_residual = self.compute_residual_force(f_inertial_frame, dataset.dv[i])
        if is_in_body_frame:
            f_residual = dataset.shared_r_disk[i].T @ f_residual
        return f_residual

    def compute_model_thrust(self, u_free, v_forward, r_disk, omega, is_ccw_blade):
        if is_ccw_blade:
            f_x, f_y, f_z, v_i = self.bet_instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        else:
            f_x, f_y, f_z, v_i = self.bet_instance.get_rotor_forces(u_free, v_forward, r_disk, -omega, is_ccw_blade) # negative omega for CW rotation, this is an interface mismatch
        return np.array([f_x, f_y, f_z])

    @staticmethod
    def compute_thrust_with_lookup_table(u_free, v_forward, r_disk, omega, is_ccw_blade, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        # The output is in inertial frame
        thrust, _ = lookup_table.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        return thrust
    
    def get_loss(self, x, datasets: list[data_factory.FittingDataset], lookup_table=None, is_using_lookup_table=False):
        
        self.blade.cl_1, self.blade.cl_2, self.blade.cd, self.blade.alpha_0 = x 
        self.bet_instance.refresh_blade()

        loss = 0.0
        for dataset in datasets:
            self.initialize_logger()
            data_len = len(dataset.u_free_0)
            loss_per_data_set = 0.0
            num_of_samples_per_dataset = data_len // self.sample_distance
            for i in range(0, data_len, self.sample_distance):
                f_residual = self.get_residual_force(dataset, i, lookup_table, is_using_lookup_table)
                loss_f = self.horizontal_weight*(f_residual[0]**2 + f_residual[1]**2) + f_residual[2]**2
                loss_per_data_set += loss_f
            loss_per_data_set = loss_per_data_set / num_of_samples_per_dataset
            loss += loss_per_data_set
            if self.can_log_and_display_data:
                self.plot_the_fit(dataset)
                self.initialize_logger()
        loss = loss / len(datasets)
        print(f"Current loss: {loss}")
        return loss

    def fit_params(self, datasets: list[data_factory.FittingDataset], initial_guess=None, is_fine_tune=False):
        cl_1_range = [3.0, 4.0, 5.0, 6.0]
        cl_2_range = [1.0, 1.5, 2.0]
        cd_range = [1.0, 1.5, 2.0]
        alpha_0_range = [np.radians(15), np.radians(20), np.radians(25)]
        if initial_guess is None:
            # Initial guess for cl_1, cl_2, cd, alpha_0
            initial_guess = np.array([5.0, 1.5, 1.5, 0.3])*1.5
        print("Initial guess:", initial_guess)
        # Define the bounds for each parameter
        bounds = [(2.0, 50.0), (0.0, 50.0), (0.0, 5.0), (np.radians(10), np.radians(40))]
        if is_fine_tune:
            maxiter = 200
        else:
            maxiter = 50
        trace = []
        step_counter = {"count": 0}  # mutable so it can persist across callback calls

        def record_and_print(xk):
            trace.append(xk.copy())
            step_counter["count"] += 1
            if step_counter["count"] % 1 == 0:
                print(f"Step {step_counter['count']}: x = {xk}")

        # x_fixed = initial_guess.copy()

        # def loss_first2(z, datasets):
        #     x = x_fixed.copy()
        #     x[:2] = z          # only first 2 dims are optimized
        #     return self.get_loss(x, datasets)

        # # Optimize the parameters using a minimization algorithm
        # result = minimize(
        #     loss_first2,
        #     x_fixed[:2],
        #     args=(datasets,),  # args=(datasets, None, is_fine_tune),
        #     bounds=bounds[:2],
        #     callback=record_and_print,
        #     method='Nelder-Mead',
        #     options={'disp': True, 
        #             'maxiter': maxiter,
        #             'fatol': 1e-1}
        # )

        # Optimize the parameters using a minimization algorithm
        result = minimize(
            self.get_loss,
            initial_guess.copy(),
            args=(datasets,),  # args=(datasets, None, is_fine_tune),
            bounds=bounds,
            callback=record_and_print,
            method='Nelder-Mead',
            options={'disp': True, 
                    'maxiter': maxiter,
                    'fatol': 1e-1}
        )

        if result.success:
            fitted_params = result.x
            print("Fitted parameters:", fitted_params)
            return fitted_params
        else:
            print("Optimization failed:", result.message)
            return None

    def compute_ground_truth(self, dataset):
        f_total_inertial_frame_gt = [
            (-self.params.m * parameters.Environment.g * np.array([0.0, 0.0, -1.0]) + self.params.m * dataset.dv[i])
            for i in range(len(dataset.dv))
        ]
        f_0_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_0_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
            ]
        f_1_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_1_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_1_f_rotor_inertial_frame))
            ]
        f_2_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_2_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_2_f_rotor_inertial_frame))
            ]
        f_3_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_3_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_3_f_rotor_inertial_frame))
            ]
        f_total_from_rotor_inertial_frame_gt = [
            (dataset.rotor_0_f_rotor_inertial_frame[i] + dataset.rotor_1_f_rotor_inertial_frame[i] + \
            dataset.rotor_2_f_rotor_inertial_frame[i] + dataset.rotor_3_f_rotor_inertial_frame[i])
            for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
            ]
        return f_total_inertial_frame_gt, f_0_body_frame_gt, f_1_body_frame_gt, f_2_body_frame_gt, f_3_body_frame_gt, f_total_from_rotor_inertial_frame_gt

    def plot_the_fit(self, dataset):
        f_total_inertial_frame_gt, f_0_body_frame_gt, f_1_body_frame_gt, f_2_body_frame_gt, f_3_body_frame_gt, f_total_from_rotor_inertial_frame_gt = self.compute_ground_truth(dataset)

        fig0, axs0 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        force_labels = ["Force X", "Force Y", "Force Z"]
        for i in range(3):
            axs0[i].plot(self.logger["sample_index"], [f[i] for f in self.logger["f_inertial_frame"]], label=f'Fitted F{["x","y","z"][i]}', linestyle='None', marker='.')
            axs0[i].plot([f[i] for f in f_total_inertial_frame_gt], label=f'GT F{["x","y","z"][i]}', linestyle='-', marker='.')
            axs0[i].plot([f[i] for f in f_total_from_rotor_inertial_frame_gt], label=f'GT from Rotor F{["x","y","z"][i]}', linestyle='-', marker='.')
            axs0[i].set_ylabel(force_labels[i])
            axs0[i].legend()
        axs0[2].set_xlabel("Sample Index")

        fig0.tight_layout()

        fig1, axs1 = plt.subplots(4, 3, figsize=(12, 8), sharex=True)

        for i in range(3):
            axs1[0, i].plot(self.logger["sample_index"], [f[i] for f in self.logger["f_0"]], label='Fitted F0', linestyle='None', marker='.')
            axs1[0, i].plot([f[i] for f in f_0_body_frame_gt], label='GT F0', linestyle='-')
            axs1[1, i].plot(self.logger["sample_index"], [f[i] for f in self.logger["f_1"]], label='Fitted F1', linestyle='None', marker='.')
            axs1[1, i].plot([f[i] for f in f_1_body_frame_gt], label='GT F1', linestyle='-')
            axs1[2, i].plot(self.logger["sample_index"], [f[i] for f in self.logger["f_2"]], label='Fitted F2', linestyle='None', marker='.')
            axs1[2, i].plot([f[i] for f in f_2_body_frame_gt], label='GT F2', linestyle='-')
            axs1[3, i].plot(self.logger["sample_index"], [f[i] for f in self.logger["f_3"]], label='Fitted F3', linestyle='None', marker='.')
            axs1[3, i].plot([f[i] for f in f_3_body_frame_gt], label='GT F3', linestyle='-')
            axs1[0, i].set_ylabel(f"F0 {['X', 'Y', 'Z'][i]}")
            axs1[1, i].set_ylabel(f"F1 {['X', 'Y', 'Z'][i]}")
            axs1[2, i].set_ylabel(f"F2 {['X', 'Y', 'Z'][i]}")
            axs1[3, i].set_ylabel(f"F3 {['X', 'Y', 'Z'][i]}")
        for ax in axs1.flat:
            ax.legend()
        return fig0, fig1

    def make_residual_force_columns(self, dataset: data_factory.FittingDataset, lookup_table):
        f_residual = []
        for i in range(len(dataset)):
            f_residual.append(self.get_residual_force(dataset, i, lookup_table, True, True))
        f_residual = np.array(f_residual)
        df = pd.read_csv(dataset.path_to_data_file)
        df["f_residual"] = f_residual.tolist()
        df.to_csv(dataset.path_to_data_file, index=False, float_format='%.17f')

    