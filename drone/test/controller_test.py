import unittest
import numpy as np

import drone.parameters as params
from drone.controller import DroneController


class DummySensorData:
    def __init__(self):
        self.pose = np.eye(3)                 # body -> world, FLU
        self.position = np.zeros(3)
        self.v = np.zeros(3)
        self.v_dot = np.zeros(3)
        self.omega = np.zeros(3)              # body frame
        self.omega_dot = np.zeros(3)


class DummyReference:
    def __init__(self):
        self.x_d = np.zeros(3)
        self.v_d = np.zeros(3)
        self.x_d_dot2 = np.zeros(3)
        self.x_d_dot3 = np.zeros(3)
        self.x_d_dot4 = np.zeros(3)
        self.b_1d = np.array([1.0, 0.0, 0.0])
        self.b_1d_dot = np.zeros(3)
        self.b_1d_dot2 = np.zeros(3)


class TestDroneControllerMath(unittest.TestCase):

    @staticmethod
    def compute_equivalent_body_thrust_derivative(
        f_d,
        f_d_dot,
        pose,
        omega_body,
    ):
        b3 = pose[:, 2]
        omega_world = pose @ omega_body
        b3_dot = np.cross(omega_world, b3)

        f_world_dot = (
            (f_d_dot @ b3) * b3
            + (f_d @ b3_dot) * b3
            + (f_d @ b3) * b3_dot
        )

        return pose.T @ f_world_dot

    def test_project_desired_force_to_body_thrust_equivalent_world_derivative(self):
        pose = np.eye(3)
        omega_body = np.array([0.2, -0.3, 0.4])

        f_d = np.array([1.0, 2.0, 10.0])
        f_d_dot = np.array([0.5, -0.2, 1.5])

        f, f_dot = DroneController.project_desired_force_to_body_thrust(
            f_d=f_d,
            f_d_dot=f_d_dot,
            pose=pose,
            omega_body=omega_body,
        )

        f_dot_equivalent = self.compute_equivalent_body_thrust_derivative(
            f_d=f_d,
            f_d_dot=f_d_dot,
            pose=pose,
            omega_body=omega_body,
        )

        np.testing.assert_allclose(f_dot, f_dot_equivalent, atol=1e-12)
        np.testing.assert_allclose(f, np.array([0.0, 0.0, f_d @ pose[:, 2]]), atol=1e-12)



if __name__ == '__main__':
    unittest.main()