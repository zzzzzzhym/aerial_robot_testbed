import unittest
from unittest.mock import MagicMock
import numpy as np

import drone.parameters as params
from drone.controller import DroneController, DisturbanceEstimatorChoice


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


class TestTorqueDisturbanceCompensation(unittest.TestCase):
    """Controller built without __init__ (which loads the learned models); only the needed fields are set."""

    def make_controller(self, choice):
        controller = DroneController.__new__(DroneController)
        controller.disturbance_estimator_choice = choice
        controller.is_compensating_torque_disturbance = True
        controller.max_torque_disturb_compensation = 3.0
        controller.torque_disturb = np.array([0.1, -2.0, 0.0])
        controller.torque_disturb_base = np.array([0.0, 0.5, 0.0])
        controller.torque_disturb_bemt = np.zeros(3)
        controller.torque_disturb_seq = np.array([0.0, 0.0, 1.0])
        return controller

    def test_compensation_cancels_selected_estimate(self):
        controller = self.make_controller(DisturbanceEstimatorChoice.PURE_DAIML)
        np.testing.assert_allclose(controller.get_torque_disturb_compensation(), [-0.1, 2.0, 0.0])
        controller.disturbance_estimator_choice = DisturbanceEstimatorChoice.SEQUENCE
        np.testing.assert_allclose(controller.get_torque_disturb_compensation(), [0.0, 0.0, -1.0])

    def test_compensation_saturates(self):
        controller = self.make_controller(DisturbanceEstimatorChoice.PURE_DAIML)
        controller.torque_disturb = np.array([0.0, -8.0, 0.0])
        np.testing.assert_allclose(controller.get_torque_disturb_compensation(), [0.0, 3.0, 0.0])

    def test_compensation_disabled(self):
        controller = self.make_controller(DisturbanceEstimatorChoice.PURE_DAIML)
        controller.is_compensating_torque_disturbance = False
        np.testing.assert_array_equal(controller.get_torque_disturb_compensation(), np.zeros(3))

    def test_compensation_low_pass(self):
        controller = self.make_controller(DisturbanceEstimatorChoice.PURE_DAIML)
        controller.torque_disturb_compensation_time_constant = 0.3
        controller.torque_disturb_compensation_dt = 0.01
        controller.torque_disturb_compensation = np.zeros(3)
        first = controller.step_torque_disturb_compensation().copy()
        np.testing.assert_allclose(first, np.array([-0.1, 2.0, 0.0])*0.01/0.31)
        for _ in range(30 - 1):     # one time constant -> ~63 % of the target
            controller.step_torque_disturb_compensation()
        np.testing.assert_allclose(controller.torque_disturb_compensation[1], 2.0*(1 - (0.3/0.31)**30), rtol=1e-12)
        for _ in range(1000):
            controller.step_torque_disturb_compensation()
        np.testing.assert_allclose(controller.torque_disturb_compensation, [-0.1, 2.0, 0.0], atol=1e-9)

    def test_daiml_torque_axes_get_omega_tracking_error(self):
        controller = self.make_controller(DisturbanceEstimatorChoice.PURE_DAIML)
        controller._log_data = {}
        controller.e_v = np.array([1.0, 2.0, 3.0])
        controller.e_omega = np.array([0.4, 0.5, 0.6])
        controller.f = np.zeros(3)
        controller.torque = np.zeros(3)
        controller.get_sensed_disturbance = lambda sensor_data: np.zeros(6)
        controller.get_predicted_air_force = lambda sensor_data: (np.zeros(3), np.zeros(3))
        for name in ["disturbance_estimator", "baseline_disturbance_estimator",
                     "bemt_disturbance_estimator", "sequence_disturbance_estimator"]:
            estimator = MagicMock()
            estimator.get_disturbance_force.return_value = np.zeros(3)
            estimator.get_disturbance_torque.return_value = np.zeros(3)
            setattr(controller, name, estimator)
        sensor_data = MagicMock()
        sensor_data.rotors.rotors = [MagicMock(rotation_speed=0.0) for _ in range(4)]

        controller.step_disturbance_estimator(sensor_data)

        daiml_tracking_error = controller.disturbance_estimator.step_disturbance.call_args.args[-1]
        np.testing.assert_array_equal(daiml_tracking_error, [1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
        baseline_tracking_error = controller.baseline_disturbance_estimator.step_disturbance.call_args.args[-1]
        np.testing.assert_array_equal(baseline_tracking_error, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

        controller.is_compensating_torque_disturbance = False   # not fed back -> no torque tracking term
        controller.step_disturbance_estimator(sensor_data)
        daiml_tracking_error = controller.disturbance_estimator.step_disturbance.call_args.args[-1]
        np.testing.assert_array_equal(daiml_tracking_error, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])



if __name__ == '__main__':
    unittest.main()