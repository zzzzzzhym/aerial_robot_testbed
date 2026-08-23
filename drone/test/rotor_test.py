import unittest
import numpy as np

from drone.rotor import Rotor
from drone import propeller


class TestRotorSetForceFromBodyFrame(unittest.TestCase):

    def _make_rotor(self):
        return Rotor(propeller.apc_8x6, relative_position_body_frame=np.array([0.1, 0.0, 0.0]), is_ccw_blade=True)

    def test_identity_pose_passthrough(self):
        rotor = self._make_rotor()
        rotor.pose = np.eye(3)
        f_body = np.array([1.0, 2.0, 5.0])
        rotor.set_force_from_body_frame(f_body)
        np.testing.assert_array_almost_equal(rotor.f_rotor_inertial_frame, f_body)
        self.assertAlmostEqual(rotor.thrust, 5.0)

    def test_rotated_pose_transforms_correctly(self):
        rotor = self._make_rotor()
        # 90-degree rotation around z: x->y, y->-x
        rotor.pose = np.array([[0, -1, 0],
                                [1,  0, 0],
                                [0,  0, 1]], dtype=float)
        f_body = np.array([1.0, 0.0, 3.0])
        rotor.set_force_from_body_frame(f_body)
        expected_inertial = np.array([0.0, 1.0, 3.0])
        np.testing.assert_array_almost_equal(rotor.f_rotor_inertial_frame, expected_inertial)
        self.assertAlmostEqual(rotor.thrust, 3.0)

    def test_thrust_is_body_z_component(self):
        rotor = self._make_rotor()
        rotor.pose = np.eye(3)
        f_body = np.array([0.5, -0.3, 7.2])
        rotor.set_force_from_body_frame(f_body)
        self.assertAlmostEqual(rotor.thrust, 7.2)

    def test_zero_force(self):
        rotor = self._make_rotor()
        rotor.pose = np.eye(3)
        rotor.set_force_from_body_frame(np.zeros(3))
        np.testing.assert_array_equal(rotor.f_rotor_inertial_frame, np.zeros(3))
        self.assertAlmostEqual(rotor.thrust, 0.0)


if __name__ == "__main__":
    unittest.main()
