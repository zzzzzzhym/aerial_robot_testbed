import numpy as np

class Environment:
    """Environment parameters"""
    rho_air = 1.225 # kg/m^3 air density
    g = 9.81 # gravity [m/s^2]

class Rotor:
    """Rotor parameters as a subset of drone parameters"""
    def __init__(self):
        self.num_of_rotors = 0
        self.rotor_position = []
        self.is_ccw_blade = []


class Multicopter:
    """Drone parameters base class """
    def __init__(self, m, inertia, c_tau_f, 
                 rotor_position, is_ccw_blade, rotor_direction=None):
        """Initialize base parameters that must be set in the subclass
        Args:
            m: mass
            inertia: 3x3 inertia matrix
            c_tau_f: yaw torque / thrust coefficient
            rotor_position: list of rotor position vectors in body frame FLU (different from SE(3) paper)
            is_ccw_blade: list[bool], True if blade rotates counter-clockwise from bird view (opposite to z axis body frame)
            rotor_direction: list of rotor direction vectors (optional)
            Note: c_tau_f is not the c_m in M_z = c_m*omega_z**2. If M_z = c_m*omega_z**2, F = c_t*omega**2, then c_tau_f = c_m/c_t. 
        """
        self.m = m
        self.inertia = inertia
        self.c_tau_f = c_tau_f 
        self.rotor_position = rotor_position
        self.is_ccw_blade = is_ccw_blade 
        self.num_of_rotors = len(self.rotor_position)

        if len(self.is_ccw_blade) != self.num_of_rotors:
            raise ValueError("rotor_position and is_ccw_blade must have the same length")

        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 matrix")
        
        # The following attributes are calculated based on the initialized attributes
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.m_thrust_to_wrench, self.m_wrench_to_thrust = self.get_thrust_wrench_matrix()
    
    @staticmethod
    def flip_between_flu_frd(vector_flu: np.ndarray):
        """cannot use utils because of circular import"""
        # conversion matrix between different coordinate systems
        m_frd_flu = np.array([[1, 0, 0], 
                              [0, -1, 0], 
                              [0, 0, -1]])
        return m_frd_flu@vector_flu

    def get_rotor_position(self):
        return self.rotor_position

    def get_thrust_wrench_matrix(self):
        """The convention follows Geometric Tracking Control of a Quadrotor UAV on SE(3) paper, which is front right down positive
        Note that in the paper, thrust of a rotor is positive (f_i > 0) when it is in the negative z axis direction.
        Assume thrust is always in the negative z axis direction of the body frame (does not apply to ARI Tarot 960)
        Returns:
            tuple: (m_thrust_to_wrench, m_wrench_to_thrust)
        """
        m_0 = np.ones(self.num_of_rotors)
        thrust_moment = Multicopter.construct_thrust_induced_moment(self.rotor_position)  # [3, N] moment contributions from thrust forces only

        m_3 = np.array([self.c_tau_f if ccw else -self.c_tau_f for ccw in self.is_ccw_blade])   # ccw blade provides positive z axis torque to drone
        thrust_moment[2, :] += m_3
        m_thrust_to_wrench = np.vstack((m_0, thrust_moment))
        m_wrench_to_thrust = np.linalg.inv(m_thrust_to_wrench)
        return m_thrust_to_wrench, m_wrench_to_thrust
    
    @staticmethod
    def construct_thrust_induced_moment(rotor_position):
        """
        Construct moment contributions from thrust forces only.

        Each column i corresponds to:
            m_i = r_i x F_i

        where:
            r_i: rotor position (converted to FRD)
            F_i: thrust direction [0, 0, -1]

        Returns:
            np.ndarray: shape (3, N)
        """
        unit_thrust = np.array([0.0, 0.0, -1.0])    # thrust is in negative z axis

        moment_list = []
        for p in rotor_position:
            p_frd = Multicopter.flip_between_flu_frd(p)
            moment = np.cross(p_frd, unit_thrust)
            moment_list.append(moment)

        return np.array(moment_list).T   # shape (3, N)

    def get_rotor_data(self):
        """Package rotor data to send to other classes. 
        The Rotor class here is params.Rotor, not the execution class that accepts the Package
        """
        rotor = Rotor()
        rotor.num_of_rotors = self.num_of_rotors
        rotor.rotor_position = self.rotor_position
        rotor.is_ccw_blade = self.is_ccw_blade
        return rotor

class Quadcopter(Multicopter):
    """Concrete quadcopter class."""

    def __init__(self, m, inertia, c_tau_f, p_0, p_1, p_2, p_3, is_ccw_blade):
        rotor_position = [p_0, p_1, p_2, p_3]

        if len(is_ccw_blade) != 4:
            raise ValueError("Quadcopter must have exactly 4 blade direction entries")

        super().__init__(
            m=m,
            inertia=inertia,
            c_tau_f=c_tau_f,
            rotor_position=rotor_position,
            is_ccw_blade=is_ccw_blade,
        )


class PennStateARILab550(Quadcopter):
    """ARI lab 550 drone (4 rotors) parameters (12inch-2blade propeller)"""
    def __init__(self):
        m = 1.6315+0.508    # drone + battery [kg]
        d = 0.28   # from drone center to motor center [m]
        h = 0.095  # height rotor to center of gravity [m]
        inertia = np.diag([0.0820, 0.0845, 0.1377])  # [kgm2]  this is temporary value, copy from elsewhere
        num_of_rotors = 4 
        c_tau_f = 8.004e-3  # this is temporary value, copy from SE3 paper and intendedly increased because the weak yaw torque will hit rotor limit easily, increasing this will make a stronger yaw torque control
        # rotors are 90 degree apart
        # rotor labels start from front left in a counter-clockwise order
        p_0 = np.array([d*np.cos(np.pi/4), d*np.sin(np.pi/4), h])   # front left
        p_1 = np.array([-d*np.cos(np.pi/4), d*np.sin(np.pi/4), h])  # rear left
        p_2 = np.array([-d*np.cos(np.pi/4), -d*np.sin(np.pi/4), h]) # rear right
        p_3 = np.array([d*np.cos(np.pi/4), -d*np.sin(np.pi/4), h])  # front right
        is_ccw_blade = [True, False, True, False]
        super().__init__(m=m, inertia=inertia, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3, 
                         is_ccw_blade=is_ccw_blade)  
        self.f_motor_max = 50.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]      

class PennStateARILabTarot960(Multicopter):
    """ARI lab Tarot 960 drone (8 rotors) parameters (15inch-2blade propeller)"""
    def __init__(self):
        m = 3.5    # kg
        inertia = np.diag([0.0820, 0.0845, 0.1377])  # [kgm2] this is temporary value, copy from elsewhere
        num_of_rotors = 6
        c_tau_f = 8.004e-4  # convert thrust to torque in z axis [m]

        # rotor position vectors in body frame (note that in this paper, 2 rotors are in x axis and 2 rotors are in y axis, unlike a regular drone setup)
        p_0 = np.array([ 0.41796455,  0.28346076,  0.05590194])
        p_1 = np.array([ 0.03227500,  0.48316780,  0.05590194])
        p_2 = np.array([-0.45023955,  0.22755882,  0.05590194])
        p_3 = np.array([-0.45023955, -0.22755882,  0.05590194])
        p_4 = np.array([ 0.03227500, -0.48316780,  0.05590194])
        p_5 = np.array([ 0.41796455, -0.28346076,  0.05590194])
        is_ccw_blade = [False, True, False, True, False, True]  
        super().__init__(m=m, inertia=inertia, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3,
                         p_4=p_4, p_5=p_5, is_ccw_blade=is_ccw_blade)  
        self.f_motor_max = 50.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]   

class P600(Quadcopter):
    """Drone of "A Highly-Efficient Hybrid Simulation System for Flight Controller Design and Evaluation of Unmanned Aerial Vehicles"
    """
    def __init__(self):
        m = 3.0    # kg
        inertia = np.diag([0.05487, 0.05487, 0.1027])  # [kgm2] 
        num_of_rotors = 4
        c_tau_f = 9.004e-7/4.848e-5  # convert thrust to torque in z axis [m]; this is not from their repo
        # rotor position vectors in body frame (note that in this paper, 2 rotors are in x axis and 2 rotors are in y axis, unlike a regular drone setup)
        p_0 = np.array([ 0.21213,  0.21213, 0.243])  # front left
        p_1 = np.array([-0.21213,  0.21213, 0.243])  # rear left
        p_2 = np.array([-0.21213, -0.21213, 0.243])  # rear right
        p_3 = np.array([ 0.21213, -0.21213, 0.243])  # front right

        is_ccw_blade = [False, True, False, True]  
        super().__init__(m=m, inertia=inertia, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3, 
                         is_ccw_blade=is_ccw_blade)
        self.f_motor_max = 20.0  # maximum possible thrust per motor [N] 
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]

class TrackingOnSE3(Quadcopter):
    """
    parameters come from 
    Geometric Tracking Control of a Quadrotor UAV on SE(3)
    """
    def __init__(self):
        m = 5    # kg
        d = 0.315   # distance from drone center to motor center [m]
        inertia = np.diag([0.0820, 0.0845, 0.1377])  # [kgm2]
        num_of_rotors = 4
        c_tau_f = 8.004e-4  # convert thrust to torque in z axis [m]
        # rotor position vectors in body frame (note that in this paper, 2 rotors are in x axis and 2 rotors are in y axis, unlike a regular drone setup)
        p_0 = self.flip_between_flu_frd(np.array([d, 0, 0]))     # positive x
        p_1 = self.flip_between_flu_frd(np.array([0, d, 0]))     # positive y
        p_2 = self.flip_between_flu_frd(np.array([-d, 0, 0]))    # negative x
        p_3 = self.flip_between_flu_frd(np.array([0, -d, 0]))    # negative y
        is_ccw_blade = [False, True, False, True]  
        super().__init__(m=m, inertia=inertia, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3, 
                         is_ccw_blade=is_ccw_blade)
        self.f_motor_max = 50.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]

class Neurobem(Quadcopter):
    """
    From https://download.ifi.uzh.ch/rpg/NeuroBEM/
    code/simulator/include/params.h
    """
    def __init__(self):
        m = 0.772    # kg   probably included other payload, since it's not the same as 0.752 in the params.h
        d = 0.1   # distance from drone center to motor center [m]
        inertia = np.diag([0.0025, 0.0021, 0.0043])  # [kgm2] 
        num_of_rotors = 4
        c_tau_f = 8.004e-4  # convert thrust to torque in z axis [m]; this is not from their repo
        # rotor position vectors in body frame (note that in this paper, 2 rotors are in x axis and 2 rotors are in y axis, unlike a regular drone setup)
        p_0 = self.flip_between_flu_frd(np.array([d, 0, 0]))     # positive x
        p_1 = self.flip_between_flu_frd(np.array([0, d, 0]))     # positive y
        p_2 = self.flip_between_flu_frd(np.array([-d, 0, 0]))    # negative x
        p_3 = self.flip_between_flu_frd(np.array([0, -d, 0]))    # negative y
        is_ccw_blade = [False, True, False, True]  
        super().__init__(m=m, inertia=inertia, num_of_rotors=num_of_rotors, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3, 
                         is_ccw_blade=is_ccw_blade)
        self.f_motor_max = 10.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]

class EndEffector:
    """End effector parameters
    Assume a fixed rod with a spherical sponge at the tip, only consider 1D linear deformation without damping effect
    Center-tip length refer to Guo, Xiaofeng, et al. "Flying calligrapher: Contact-aware motion and force planning and control for aerial manipulation." IEEE Robotics and Automation Letters (2024).
    """
    def __init__(self):
        self.tip_position = np.array([-0.45, 0.0, 0.0])  # position of end effector in body frame FLU[m]
        self.sponge_radius = 0.1  # radius of the spherical sponge [m]
        self.k_sponge = 100*0.1  # spring constant of the sponge [N/m]
        self.miu_friction = 0.4  # friction coefficient between sponge and wall [unitless]


class Control:
    """Control parameters"""
    k_x = 16
    k_v = 5.6
    k_r = 8.81
    k_omega = 2.54

# miscellaneous parameters (used in disturbance model)
rotor_radius = 0.2 # [m] 15inch diameter rotor
c_d = 1.2   # unit free [0.5-1.5]   "An Experimental Study of Drag Coefficients of a Quadrotor Airframe." Table 2
area_frontal = 0.03  # m^2 [0.01-0.1]   "An Experimental Study of Drag Coefficients of a Quadrotor Airframe." Table 2

