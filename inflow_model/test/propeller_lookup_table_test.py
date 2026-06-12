import unittest
import numpy as np

from propeller_lookup_table import PropellerLookupTable

class TestPropellerLookupTableReader(unittest.TestCase):
    def test_read_data(self):
        # check if the parameters are set correctly
        instance = PropellerLookupTable.Reader("apc_8x6_with_trail")
        np.testing.assert_array_equal(instance.omega_range, PropellerLookupTable.Maker._DEFAULT_OMEGA_RANGE)
        np.testing.assert_array_equal(instance.u_free_x_range, PropellerLookupTable.Maker._DEFAULT_U_FREE_X_RANGE)
        np.testing.assert_array_equal(instance.pitch_range, PropellerLookupTable.Maker._DEFAULT_PITCH_RANGE)
        # check if the table has the correct shape
        expected_shape = (len(instance.u_free_x_range), len(instance.pitch_range), len(instance.omega_range), 4)
        self.assertEqual(instance.table.shape, expected_shape)

    def test_get_rotation_matrix_between_inertial_and_lookup_table_frame(self):
        x_axis = np.array([1, 0, 0])
        r_disk = np.eye(3)
        matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        np.testing.assert_array_almost_equal(matrix_from_inertial_to_lookup_table, np.eye(3))
        np.testing.assert_array_almost_equal(matrix_from_lookup_table_to_inertial, np.eye(3))

        x_axis = np.array([0, 1, 0])
        matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        expected_lookup_table_rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T
        np.testing.assert_array_almost_equal(matrix_from_lookup_table_to_inertial, expected_lookup_table_rotation_matrix)
        np.testing.assert_array_almost_equal(matrix_from_inertial_to_lookup_table, expected_lookup_table_rotation_matrix.T)

        x_axis = np.array([0, 0, 1])
        matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        expected_lookup_table_rotation_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
        np.testing.assert_array_almost_equal(matrix_from_lookup_table_to_inertial, expected_lookup_table_rotation_matrix)
        np.testing.assert_array_almost_equal(matrix_from_inertial_to_lookup_table, expected_lookup_table_rotation_matrix.T)

        x_axis = np.array([1, 0, 0])
        r_disk = np.array([[0, 1, 0],   # x axis after transpose
                           [0, 0, 1],   # y axis after transpose
                           [1, 0, 0]]).T    # z axis after transpose
        matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        expected_lookup_table_rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        np.testing.assert_array_almost_equal(matrix_from_lookup_table_to_inertial, expected_lookup_table_rotation_matrix)
        np.testing.assert_array_almost_equal(matrix_from_inertial_to_lookup_table, expected_lookup_table_rotation_matrix.T)
        
        theta = np.pi / 3
        r_disk = np.array([[np.cos(theta), np.sin(theta), 0],   # x axis after transpose
                           [-np.sin(theta), np.cos(theta), 0],  # y axis after transpose
                           [0, 0, 1]]).T    # z axis after transpose
        matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        expected_lookup_table_rotation_matrix = np.eye(3)
        np.testing.assert_array_almost_equal(matrix_from_lookup_table_to_inertial, expected_lookup_table_rotation_matrix)
        np.testing.assert_array_almost_equal(matrix_from_inertial_to_lookup_table, expected_lookup_table_rotation_matrix.T)

    def test_get_pitch_angle(self):
        x_axis = np.array([1, 0, 0])
        r_disk = np.eye(3)
        matrix_from_inertial_to_lookup_table, _ = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        pitch = PropellerLookupTable.Reader.get_pitch_angle(r_disk, matrix_from_inertial_to_lookup_table)
        self.assertAlmostEqual(pitch, 0.0)

        r_disk = np.array([[0, 1, 0],   # x axis after transpose
                           [0, 0, 1],   # y axis after transpose
                           [1, 0, 0]]).T    # z axis after transpose
        matrix_from_inertial_to_lookup_table, _ = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        pitch = PropellerLookupTable.Reader.get_pitch_angle(r_disk, matrix_from_inertial_to_lookup_table)
        self.assertAlmostEqual(pitch, np.pi / 2)

        r_disk = np.array([[0, 0, 1],   # x axis after transpose
                           [1, 0, 0],   # y axis after transpose
                           [0, 1, 0]]).T    # z axis after transpose
        matrix_from_inertial_to_lookup_table, _ = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
        pitch = PropellerLookupTable.Reader.get_pitch_angle(r_disk, matrix_from_inertial_to_lookup_table)
        self.assertAlmostEqual(pitch, 0.0)

    def test_get_rotor_forces(self):
        instance = PropellerLookupTable.Reader("apc_8x6_with_trail")
        u_free = 0.0
        v_forward = 0.0
        r_disk = np.eye(3)
        omega = 0.0
        is_ccw_blade = True
        forces_in_inertial_frame, _ = instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        np.testing.assert_array_almost_equal(forces_in_inertial_frame, np.array([0.0, 0.0, 0.0]))

    def test_get_rotation_speed(self):
        omega_range = np.array([1000, 2000, 3000, 4000, 5000])
        thrust_profile = np.array([0.1, 0.5, 0.3, 0.6, 0.9])  # non-monotonic

        # read any file and overwrite the members
        reader = PropellerLookupTable.Reader("apc_8x6_with_trail")
        reader.omega_range = omega_range
        reader.u_free_x_range = np.array([0.0])
        reader.pitch_range = np.array([0.0])
        reader.table = np.zeros((1, 1, len(omega_range), 4))
        reader.table[0, 0, :, 2] = thrust_profile
        reader.get_interpolator()

        # zero wind + horizontal disk → u_norm=0, pitch=0, same as querying at (0.0, 0.0, omega)
        u_free = np.zeros(3)
        v_forward = np.zeros(3)
        r_disk = np.eye(3)

        # Case 1: within range, close to 2000 rpm
        omega = reader.get_rotation_speed(u_free, v_forward, r_disk, omega_current=2000, thrust_desired=0.49)
        self.assertAlmostEqual(omega, 1975.0, places=1)

        # Case 2: clipped below min thrust
        omega = reader.get_rotation_speed(u_free, v_forward, r_disk, omega_current=3000, thrust_desired=-1.0)
        self.assertEqual(omega, 1000)

        # Case 3: clipped above max thrust
        omega = reader.get_rotation_speed(u_free, v_forward, r_disk, omega_current=1000, thrust_desired=5.0)
        self.assertEqual(omega, 5000)

        # Case 4: on flat segment (simulate thrust_left[i] ~= thrust_right[i+1])
        reader.table[0, 0, :, 2] = np.array([0.1, 0.5, 0.5, 0.6, 0.9])  # flat between 2000–3000
        reader.get_interpolator()
        omega = reader.get_rotation_speed(u_free, v_forward, r_disk, omega_current=2400, thrust_desired=0.5)
        self.assertEqual(omega, 2400)

        # Case 5: no valid brackets found
        reader.table[0, 0, :, 2] = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # constant thrust
        reader.get_interpolator()
        omega = reader.get_rotation_speed(u_free, v_forward, r_disk, omega_current=2500, thrust_desired=0.3)
        self.assertEqual(omega, 2500)


if __name__ == '__main__':
    unittest.main()