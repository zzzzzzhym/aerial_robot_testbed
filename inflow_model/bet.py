import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, least_squares
import warnings

from blade_params import Blade, APC_8x6, APC_8x6_OfficialData
import airfoil.aero_coeff as aero_coeff 
from airfoil.air import Air

class BladeElementTheory:
    """a hybrid model that combines the BEMT model and the BET model to simulate the propeller loading condition in forward flight.
    Reference:
        Gill, Rajan, and Raffaello D'andrea. "Propeller thrust and drag in forward flight." 
        2017 IEEE Conference on control technology and applications (CCTA). IEEE, 2017.
    """    
    def __init__(self, blade: Blade, num_of_elements=100, num_of_rotation_segments=90):
        self.blade = blade
        self.num_of_elements = num_of_elements
        self.num_of_rotation_segments = num_of_rotation_segments
        self.dy = None
        self.refresh_blade()

    def refresh_blade(self):
        self.coeff = aero_coeff.Coeffecients(cl_1=self.blade.cl_1, cl_2=self.blade.cl_2, alpha_0=self.blade.alpha_0, cd=self.blade.cd, cd_0=self.blade.cd_0)
        self.disk_area = np.pi*(self.blade.y_max**2 - self.blade.y_min**2)  # rotor disk area    
        self.set_integration_resolution(self.num_of_elements, self.num_of_rotation_segments)

    def set_integration_resolution(self, num_of_elements: int, num_of_rotation_segments: int):
        """Set the resolution of the integration for the blade element theory.
        
        Args:
            num_of_elements (int): Number of elements along the blade length.
            num_of_rotation_segments (int): Number of segments for one revolution of the blade.
        """
        self.num_of_elements = num_of_elements
        self.num_of_rotation_segments = num_of_rotation_segments
        self.dy = self.blade.y_max/self.num_of_elements

    def get_v_flow_disk_frame(self, u_free: np.ndarray, v_i: float, v_forward: np.ndarray, r_disk: np.ndarray):
        """get the relative wind speed in the disk frame. It accounts for the forward speed of the disk.

        Args:
            u_free (np.ndarray): free stream velocity in the inertial frame
            v_i (float): inflow velocity in rotation disk frame, perpendicular to the disk, positive when flow in -z direction
            v_forward (np.ndarray): disk velocity in the inertial frame
            r_disk (np.ndarray): disk rotation matrix, should have a fixed relative pose to drone body, while the drone is using SAE coordinate system, the disk is using ISO coordinate system

        Returns:
            np.ndarray: 1D array of the relative wind speed in the disk frame
        """
        v_relative_inertial_frame = u_free - v_forward
        v_flow_disk_frame = r_disk.T@v_relative_inertial_frame + np.array([0, 0, -v_i])
        return v_flow_disk_frame

    def get_relative_v_to_blade_section(self, v_flow_disk_frame: np.ndarray, psi_blade_angle: float, y: float, omega_blade: float):
        """convention in the blade rotation frame:
        the x axis is along the blade, the y axis is perpendicular to the blade, the z axis is along the blade rotation axis clockwise rotation is positive
        when psi_blade_angle is 0, the blade rotation frame overlaps with the disk frame
        
        Args:
            v_flow_disk_frame (np.ndarray): wind blowing speed to the disk
            psi_blade_angle (float): rotation angle of blade
            y (float): _description_
            omega_blade (float): rotation speed of the blade, counter clockwise is positive

        Returns:
            u_t: relative wind speed perpendicular to the blade (in y direction), when blade rotates in positive direction without wind, it generates positive u_t
            u_p: relative wind speed in z direction
        """
        # v_x_blade_frame = v_flow_disk_frame[0:2]@np.array([np.cos(psi_blade_angle), np.sin(psi_blade_angle)])
        v_y_blade_frame = v_flow_disk_frame[0:2]@np.array([-np.sin(psi_blade_angle), np.cos(psi_blade_angle)])
        u_p = v_flow_disk_frame[2]
        u_t = -v_y_blade_frame + omega_blade*y
        alpha_flow = np.arctan2(u_p, u_t)
        return u_t, u_p, alpha_flow

    def get_attack_angle_to_blade(self, alpha_flow, y):
        attack_angle = alpha_flow + self.blade.get_blade_pitch(y)
        return attack_angle

    def get_blade_element_force_in_airfoil_frame(self, u_t: float, u_p: float, attack_angle: float, y: float):
        if attack_angle > np.pi/3 or attack_angle < -np.pi/3:   # protection from backward flow
            # warnings.warn(f"attack angle is out of range, attack_angle: {np.rad2deg(attack_angle)} [deg]")
            lift, drag = 0, 0
        else:
            u = np.sqrt(u_t**2 + u_p**2)
            chord = self.blade.get_chord(y)
            cl = self.coeff.get_cl(attack_angle)
            cd = self.coeff.get_cd(attack_angle, u, chord)
            lift = 0.5*Air.rho*u**2*cl*chord*self.dy 
            drag = 0.5*Air.rho*u**2*cd*chord*self.dy 
        return lift, drag
    
    def convert_force_from_airfoil_to_blade_frame(self, lift: float, drag: float, alpha_flow: float):
        f_z = lift*np.cos(alpha_flow) + drag*np.sin(alpha_flow)
        f_y = lift*np.sin(alpha_flow) - drag*np.cos(alpha_flow)
        return f_z, f_y
    
    def get_force_in_blade_frame_ccw_blade(self, v_flow_disk_frame: np.ndarray, psi_blade_angle: float, y: float, omega_blade: float):
        """assumes ccw blade"""
        u_t, u_p, alpha_flow = self.get_relative_v_to_blade_section(v_flow_disk_frame, psi_blade_angle, y, omega_blade)
        attack_angle = self.get_attack_angle_to_blade(alpha_flow, y)
        lift, drag = self.get_blade_element_force_in_airfoil_frame(u_t, u_p, attack_angle, y)
        f_z_blade, f_y_blade = self.convert_force_from_airfoil_to_blade_frame(lift, drag, alpha_flow)
        return f_z_blade, f_y_blade
    
    def get_force_in_blade_frame(self, v_flow_disk_frame: np.ndarray, psi_blade_angle: float, y: float, omega_blade: float, is_ccw_blade=True):
        if is_ccw_blade:
            f_z_blade, f_y_blade = self.get_force_in_blade_frame_ccw_blade(v_flow_disk_frame, psi_blade_angle, y, omega_blade)
        else:
            # mirror the flow, rotation and the blade direction along x axis to create an equivalent ccw blade problem
            v_flow_mirrored_disk_frame = v_flow_disk_frame.copy()
            v_flow_mirrored_disk_frame[1] = -v_flow_mirrored_disk_frame[1]
            f_z_blade, f_y_blade = self.get_force_in_blade_frame_ccw_blade(v_flow_mirrored_disk_frame, -psi_blade_angle, y, -omega_blade)
            f_y_blade = -f_y_blade  
        return f_z_blade, f_y_blade
    
    def convert_force_from_blade_to_disk_frame(self, f_z_blade: float, f_y_blade: float, psi_blade_angle: float):
        f_x = -f_y_blade*np.sin(psi_blade_angle)
        f_y = f_y_blade*np.cos(psi_blade_angle)
        f_z = f_z_blade
        return f_x, f_y, f_z
    
    def integrate_element_force_over_one_revolution_per_blade(self, y: float, v_flow_disk_frame: np.ndarray, omega_blade: float, is_ccw_blade=True):
        f_z_per_rev = 0
        f_y_per_rev = 0
        f_x_per_rev = 0
        for j in range(self.num_of_rotation_segments):
            psi_blade_angle = j*2*np.pi/self.num_of_rotation_segments - np.pi
            f_z_blade, f_y_blade = self.get_force_in_blade_frame(v_flow_disk_frame, psi_blade_angle, y, omega_blade, is_ccw_blade)
            f_x_disk_frame, f_y_disk_frame, f_z_disk_frame = self.convert_force_from_blade_to_disk_frame(f_z_blade, f_y_blade, psi_blade_angle)
            f_z_per_rev += f_z_disk_frame
            f_y_per_rev += f_y_disk_frame
            f_x_per_rev += f_x_disk_frame
        f_z_per_rev = f_z_per_rev/self.num_of_rotation_segments
        f_y_per_rev = f_y_per_rev/self.num_of_rotation_segments
        f_x_per_rev = f_x_per_rev/self.num_of_rotation_segments
        return f_x_per_rev, f_y_per_rev, f_z_per_rev
    
    def integrate_element_force(self, u_free: np.ndarray, v_i: float, v_forward: np.ndarray, r_disk: np.ndarray, omega_blade: float, is_ccw_blade=True, can_plot=False):
        f_z_total = 0
        f_y_total = 0
        f_x_total = 0

        if can_plot:
            # data[:, 0] = f_x_disk_frame
            # data[:, 1] = f_y_disk_frame
            # data[:, 2] = f_z_disk_frame
            # data[:, 3] = psi_blade_angle
            # data[:, 4] = y
            # data[:, 5] = x_disk_frame
            # data[:, 6] = y_disk_frame
            data = np.zeros((self.num_of_elements*self.num_of_rotation_segments, 7)) 
            i_log = 0

        v_flow_disk_frame = self.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
        for i in range(self.num_of_elements):
            y = i*self.dy
            f_z_per_rev = 0
            f_y_per_rev = 0
            f_x_per_rev = 0
            for j in range(self.num_of_rotation_segments):
                psi_blade_angle = j*2*np.pi/self.num_of_rotation_segments - np.pi
                f_z_blade, f_y_blade = self.get_force_in_blade_frame(v_flow_disk_frame, psi_blade_angle, y, omega_blade, is_ccw_blade)
                f_x_disk_frame, f_y_disk_frame, f_z_disk_frame = self.convert_force_from_blade_to_disk_frame(f_z_blade, f_y_blade, psi_blade_angle)
                f_z_per_rev += f_z_disk_frame
                f_y_per_rev += f_y_disk_frame
                f_x_per_rev += f_x_disk_frame
                if can_plot:
                    data[i_log, 0] = f_x_disk_frame
                    data[i_log, 1] = f_y_disk_frame
                    data[i_log, 2] = f_z_disk_frame
                    data[i_log, 3] = psi_blade_angle
                    data[i_log, 4] = y
                    data[i_log, 5] = y*np.cos(psi_blade_angle)
                    data[i_log, 6] = y*np.sin(psi_blade_angle)
                    i_log += 1
            f_z_per_rev = f_z_per_rev/self.num_of_rotation_segments
            f_y_per_rev = f_y_per_rev/self.num_of_rotation_segments
            f_x_per_rev = f_x_per_rev/self.num_of_rotation_segments
            f_z_total += f_z_per_rev
            f_y_total += f_y_per_rev
            f_x_total += f_x_per_rev 
        f_z_total = f_z_total*self.blade.num_of_blades
        f_y_total = f_y_total*self.blade.num_of_blades
        f_x_total = f_x_total*self.blade.num_of_blades
        if can_plot:
            self.plot_force_distribution(data)
        return np.array([f_x_total, f_y_total, f_z_total])

    def plot_force_distribution(self, data: np.ndarray):
        interval = 5
        fig0 = plt.figure()
        fig1 = plt.figure()
        fig2 = plt.figure()
        ax0 = fig0.add_subplot(111, projection='3d')
        ax1 = fig1.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')
        ax0.set_title('f_x')
        ax1.set_title('f_y')
        ax2.set_title('f_z')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax0.scatter(data[::interval, 5], data[::interval, 6], data[::interval, 0])
        ax1.scatter(data[::interval, 5], data[::interval, 6], data[::interval, 1])
        ax2.scatter(data[::interval, 5], data[::interval, 6], data[::interval, 2])
        fig3 = plt.figure()
        interval = 10
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.quiver(data[::interval, 5], data[::interval, 6], 0, data[::interval, 0], data[::interval, 1], data[::interval, 2], length=0.01, normalize=True)

    def get_dm_dot(self, v_flow_disk_frame: np.ndarray, y: float):
        # the differential mass flow rate
        # v_flow_disk_frame already considers inflow velocity, thus the equation appears to be different from the paper
        dm_dot = Air.rho*2*np.pi*y*self.dy*np.sqrt(v_flow_disk_frame[0]**2 + v_flow_disk_frame[1]**2 + v_flow_disk_frame[2]**2)
        return dm_dot

    def get_differential_thrust_from_momentum(self, v_flow_disk_frame: np.ndarray, v_i: float, y: float):
        dm_dot = self.get_dm_dot(v_flow_disk_frame, y)
        d_thrust = 2*v_i*dm_dot
        return d_thrust
    
    def get_thrust_difference(self, v_i_wrapped: np.array, y: float, u_free: np.ndarray, v_forward: np.ndarray, r_disk: np.ndarray, omega_blade: float, is_ccw_blade=True):
        v_i = v_i_wrapped[0]    # compatible with scipy.optimize.root
        v_flow_disk_frame = self.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
        df_bet = self.integrate_element_force_over_one_revolution_per_blade(y, v_flow_disk_frame, omega_blade, is_ccw_blade)
        d_thrust_bet = self.blade.num_of_blades * df_bet[2] 
        d_thrust_bemt = self.get_differential_thrust_from_momentum(v_flow_disk_frame, v_i, y)
        thrust_diff = d_thrust_bet - d_thrust_bemt
        return thrust_diff
    
    def guess_initial_v_i(self, y: float, omega_blade: float, is_ccw_blade=True):
        """Make initial guess based on static thrust model."""
        rpm_to_rad_per_sec = 2*np.pi/60
        if is_ccw_blade:
            omega_blade_in_rpm = omega_blade/rpm_to_rad_per_sec
        else:
            omega_blade_in_rpm = -omega_blade/rpm_to_rad_per_sec
        thrust_lbf = np.interp(
            omega_blade_in_rpm,
            APC_8x6_OfficialData.OMEGA_APC8X6_OFFICIAL_DATA_RPM,
            APC_8x6_OfficialData.THRUST_APC8X6_OFFICIAL_DATA_LBF
        )
        pound_force_to_newton = 4.44822
        thrust = thrust_lbf *pound_force_to_newton

        inch_to_m = 0.0254
        radius = 8*inch_to_m/2  # 8 inch diameter propeller
        area = np.pi*radius**2  # rotor disk area
        v_i = np.sqrt(thrust/(2*Air.rho*area))  # static thrust model
        return  np.array([v_i])

    def solve_v_i(self, y: float, u_free: np.ndarray, v_forward: np.ndarray, r_disk: np.ndarray, omega_blade: float, is_ccw_blade=True):
        v_i_0 = self.guess_initial_v_i(y, omega_blade, is_ccw_blade)
        result = least_squares(
            fun=self.get_thrust_difference,
            x0=v_i_0*0.5,   # in some conditions the gradient is too small to converge, 0.5 is a magic number
            args=(y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade),
            method='trf',  # or 'lm' for small problems
            max_nfev=2000,
            xtol=1e-6,
            ftol=1e-6,
            verbose=0
        )
        if not result.success:
            print(f"failed to solve v_i in inflow model, message: {result.message}")
            print(f"inputs: y: {y}, u_free: {u_free}, v_forward: {v_forward}, r_disk: {r_disk}, omega_blade: {omega_blade}, is_ccw_blade: {is_ccw_blade}")
            print(f"v_i_0: {v_i_0}, v_i_solved: {result.x[0]}")
            pass
        return result.x[0]

    def get_rotor_forces(self, u_free: np.ndarray, v_forward: np.ndarray, r_disk: np.ndarray, omega_blade: float, is_ccw_blade=True):
        """Calculate rotor forces in the disk frame.

        Args:
            u_free (np.ndarray): Free stream velocity in the inertial frame.
            v_forward (np.ndarray): Disk velocity in the inertial frame.
            r_disk (np.ndarray): Disk rotation matrix.
            omega_blade (float): Rotation speed of the blade in rad/s. In nominal conditions, ccw blade is positive, and cw blade is negative.
            is_ccw_blade (bool, optional): Indicates if the blade is counter-clockwise. Defaults to True.

        Returns:
            tuple: Forces in x, y, and z directions in the disk frame.
        """
        f_x = 0
        f_y = 0
        f_z = 0
        v_i_disk = 0
        for i in range(self.num_of_elements):
            y = i*self.dy + self.blade.y_min
            v_i = self.solve_v_i(y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade)
            v_flow_disk_frame = self.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
            df_bet = self.integrate_element_force_over_one_revolution_per_blade(y, v_flow_disk_frame, omega_blade, is_ccw_blade)
            f_x += df_bet[0]
            f_y += df_bet[1]
            f_z += df_bet[2]
            v_i_disk += v_i*self.dy*y*np.pi*2
        v_i_disk = v_i_disk/self.disk_area
        f_z = f_z*self.blade.num_of_blades
        f_y = f_y*self.blade.num_of_blades
        f_x = f_x*self.blade.num_of_blades
        return f_x, f_y, f_z, v_i_disk

    @staticmethod
    def pitch_rotor_disk_along_y_axis(pitch_angle: float):
        r_disk = np.array([[np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                           [0, 1, 0],
                           [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]])
        return r_disk

