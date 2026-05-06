import numpy as np
import quaternion

from drone import utils

class QuaternionOfRotation:
    """this class handles change of rotation matrix, given a rotation operation represented by a quaternion 
    """
    def __init__(self, 
                 q: quaternion.quaternion = np.quaternion(1.0, 0.0, 0.0, 0.0)) -> None:
        self.q = q
        self.q_dot = np.quaternion(1.0, 0.0, 0.0, 0.0)

    def step_derivative(self, omega: np.ndarray) -> None:
        '''
        omega need to be in inertial frame
        q_dot = 0.5*omega_quaternion*q
        '''        
        omega_q = utils.convert_vector_to_quaternion(omega)
        self.q_dot = 0.5*omega_q*self.q

    def step_rotation_matrix(self, r_0: np.ndarray) -> np.ndarray:
        """given the quaternion, rotate the rotation matrix
        treat rotation matrix basis individually and apply e_t = q*e_0*q^-1 

        Args:
            r_0 (np.ndarray): rotation matrix before rotation

        Returns:
            np.ndarray: rotation matrix after rotattion
        """
        r_t = np.eye(3)
        q_normalized = normalize_quaternion(self.q)
        q_normalized_conj = q_normalized.conjugate()
        for i in range(r_0.shape[1]):
            e_q = utils.convert_vector_to_quaternion(r_0[:, i])
            e_rotated = q_normalized*e_q*q_normalized_conj
            r_t[:, i] = utils.convert_quaternion_to_vector(e_rotated)
        return r_t


def get_norm_of_quaternion(q: quaternion.quaternion) -> float:
    norm = np.sqrt(q.real**2 + q.imag[0]**2 + q.imag[1]**2 + q.imag[2]**2)
    return norm


def normalize_quaternion(q: quaternion.quaternion) -> quaternion.quaternion:
    norm = get_norm_of_quaternion(q)
    q_normalized = q/norm
    return q_normalized

if __name__ == "__main__":
    q_test = np.quaternion(1, 1, 1, 1)
    q_result = normalize_quaternion(q_test)
    print(q_result)
    q_test = np.quaternion(1, 1, 0, 0)
    q_result = normalize_quaternion(q_test)
    print(q_result)

    omega = np.array([np.pi/6, 0, 0])
    t = 1
    q_test = utils.get_quaternion_from_angular_displacement(omega*t)
    test_instance = QuaternionOfRotation(q_test)
    r_0_test = np.eye(3)
    r_t_test = test_instance.step_rotation_matrix(r_0_test)
    print(r_t_test)

    omega = np.array([np.pi/6, 0, 0])
    dt = 0.01
    t = 1
    q_test = utils.get_quaternion_from_angular_displacement(omega*t)
    test_instance2 = QuaternionOfRotation()
    for _ in range(100):
        test_instance2.step_derivative(omega)   # omega in inertial frame
        test_instance2.q = test_instance2.q_dot*dt + test_instance2.q
    
    print(normalize_quaternion(test_instance2.q) - q_test)
    print(test_instance2.step_rotation_matrix(r_0_test))

