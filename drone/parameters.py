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


class Drone:
    """Drone parameters base class """
    def __init__(self, m, inertia, num_of_rotors, c_tau_f, 
                 p_0, p_1, p_2, p_3, is_ccw_blade):
        """Initialize base parameters that must be set in the subclass"""
        self.m = m
        self.inertia = inertia
        self.num_of_rotors = num_of_rotors
        self.c_tau_f = c_tau_f
        # rotor position vectors in body frame front left up (different from SE(3) paper)
        self.p_0 = p_0  
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3        
        self.is_ccw_blade = is_ccw_blade    # ccw blade rotates counter-clockwise from bird view (opposite to z axis body frame)

        # conversion matrix between different coordinate systems
        self.m_frd_flu = np.array([[1, 0, 0], 
                                   [0, -1, 0], 
                                   [0, 0, -1]])  # front right down to front left up

        # The following attributes are calculated based on the initialized attributes
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.m_thrust_to_wrench, self.m_wrench_to_thrust = self.get_thrust_wrench_matrix()
        self.rotor_position = self.get_rotor_position()
    
    def flip_between_flu_frd(self, vector_flu: np.ndarray):
        """cannot use utils because of circular import"""
        # conversion matrix between different coordinate systems
        m_frd_flu = np.array([[1, 0, 0], 
                              [0, -1, 0], 
                              [0, 0, -1]])
        return m_frd_flu@vector_flu

    def get_rotor_position(self):
        return [self.p_0, self.p_1, self.p_2, self.p_3]

    def get_thrust_wrench_matrix(self):
        """The convention follows Geometric Tracking Control of a Quadrotor UAV on SE(3) paper, which is front right down positive
        Note that in the paper, thrust of a rotor is positive (f_i > 0) when it is in the negative z axis direction.

        Returns:
            tuple: (m_thrust_to_wrench, m_wrench_to_thrust)
        """
        m_0 = np.array([1.0, 1.0, 1.0, 1.0])
        unit_thrust = np.array([0, 0, -1.0])    # thrust is in negative z axis
        v0 = np.cross(self.flip_between_flu_frd(self.p_0), unit_thrust)
        v1 = np.cross(self.flip_between_flu_frd(self.p_1), unit_thrust)
        v2 = np.cross(self.flip_between_flu_frd(self.p_2), unit_thrust)
        v3 = np.cross(self.flip_between_flu_frd(self.p_3), unit_thrust)
        m_1_2 = np.vstack((v0[:-1], v1[:-1], v2[:-1], v3[:-1])).T
        m_3 = np.array([self.c_tau_f if ccw else -self.c_tau_f for ccw in self.is_ccw_blade])   # ccw blade provides positive z axis torque to drone
        m_thrust_to_wrench = np.vstack((m_0, m_1_2, m_3))
        m_wrench_to_thrust = np.linalg.inv(m_thrust_to_wrench)
        return m_thrust_to_wrench, m_wrench_to_thrust
    
    def get_rotor_data(self):
        """Package rotor data to send to other classes. 
        The Rotor class here is params.Rotor, not the execution class that accepts the Package
        """
        rotor = Rotor()
        rotor.num_of_rotors = self.num_of_rotors
        rotor.rotor_position = self.rotor_position
        rotor.is_ccw_blade = self.is_ccw_blade
        return rotor

class PennStateARILab550(Drone):
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
        super().__init__(m=m, inertia=inertia, num_of_rotors=num_of_rotors, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3, 
                         is_ccw_blade=is_ccw_blade)  
        self.f_motor_max = 50.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]      

class TrackingOnSE3(Drone):
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
        super().__init__(m=m, inertia=inertia, num_of_rotors=num_of_rotors, 
                         c_tau_f=c_tau_f, p_0=p_0, p_1=p_1, p_2=p_2, p_3=p_3, 
                         is_ccw_blade=is_ccw_blade)
        self.f_motor_max = 50.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
        self.f_motor_min = 0.1   # minimum possible thrust per motor [N]

class Neurobem(Drone):
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

if __name__ == "__main__":
    test_instance = TrackingOnSE3()
    print(test_instance.m_frd_to_flu)
