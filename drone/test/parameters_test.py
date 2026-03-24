import unittest
import numpy as np

import parameters 

class TestGeometry(unittest.TestCase):
    def test_get_thrust_wrench_matrix(self):
        test_instance = parameters.TrackingOnSE3()
        d = test_instance.rotor_position[0][0]
        c_tau_f = test_instance.c_tau_f
        expected_matrix = np.array([[ 1.0,     1.0,     1.0,      1.0],         # thrust
                                    [ 0.0,     -d,      0.0,      d],           # torque in x axis
                                    [ d,       0.0,     -d,       0.0],         # torque in y axis
                                    [-c_tau_f, c_tau_f, -c_tau_f, c_tau_f]])  # torque in z axis
        np.testing.assert_array_almost_equal(test_instance.m_thrust_to_wrench, expected_matrix)
        np.testing.assert_array_almost_equal(test_instance.m_wrench_to_thrust, np.linalg.inv(expected_matrix))

if __name__ == '__main__':
    unittest.main()