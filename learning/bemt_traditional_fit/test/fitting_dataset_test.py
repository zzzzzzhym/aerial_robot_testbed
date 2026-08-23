import unittest
import numpy as np

import data_factory
from learning.bemt_traditional_fit.test.common import _make_mock_df


class TestFittingDataset(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.df = _make_mock_df(self.n)
        self.dataset = data_factory.FittingDataset(self.df, "mock.csv")

    def test_len_returns_n(self):
        self.assertEqual(len(self.dataset), self.n)

    def test_u_free_shape_all_rotors(self):
        for attr in ("u_free_0", "u_free_1", "u_free_2", "u_free_3"):
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.dataset, attr).shape, (self.n, 3))

    def test_v_forward_shape_all_rotors(self):
        for attr in ("v_forward_0", "v_forward_1", "v_forward_2", "v_forward_3"):
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.dataset, attr).shape, (self.n, 3))

    def test_omega_shape_all_rotors(self):
        for attr in ("omega_0", "omega_1", "omega_2", "omega_3"):
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.dataset, attr).shape, (self.n,))

    def test_shared_r_disk_shape(self):
        self.assertEqual(self.dataset.shared_r_disk.shape, (self.n, 3, 3))

    def test_dv_shape(self):
        self.assertEqual(self.dataset.dv.shape, (self.n, 3))

    def test_rotor_forces_shape_all_rotors(self):
        for attr in ("rotor_0_f_rotor_inertial_frame", "rotor_1_f_rotor_inertial_frame",
                     "rotor_2_f_rotor_inertial_frame", "rotor_3_f_rotor_inertial_frame"):
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.dataset, attr).shape, (self.n, 3))

    def test_f_residual_initially_none(self):
        self.assertIsNone(self.dataset.f_residual)

    def test_is_ready_for_second_training_initially_false(self):
        self.assertFalse(self.dataset.is_ready_for_second_training())

    def test_data_values_match_source_dataframe(self):
        # Spot-check that u_free_0[0] matches the first row of the source column
        expected = np.array(self.df["rotor_0_local_wind_velocity"].iloc[0])
        np.testing.assert_array_almost_equal(self.dataset.u_free_0[0], expected)


if __name__ == "__main__":
    unittest.main()
