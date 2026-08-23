import unittest
import numpy as np
import pandas as pd

import data_factory
from learning.bemt_traditional_fit.bemt_param_fitter import generate_seeds, BemtParamFitter, FittedBlade


def _make_mock_df(n: int) -> pd.DataFrame:
    """Build a minimal DataFrame that satisfies FittingDataset.__init__."""
    rng = np.random.default_rng(0)
    data = {}
    for rotor in range(4):
        data[f"rotor_{rotor}_local_wind_velocity"] = [rng.standard_normal(3).tolist() for _ in range(n)]
        data[f"rotor_{rotor}_velocity"] = [rng.standard_normal(3).tolist() for _ in range(n)]
        data[f"rotor_{rotor}_rotation_spd"] = rng.uniform(100, 600, n).tolist()
        data[f"rotor_{rotor}_f_rotor_inertial_frame"] = [rng.standard_normal(3).tolist() for _ in range(n)]
    data["shared_r_disk"] = [np.eye(3).tolist() for _ in range(n)]
    data["sensed_dv"] = [rng.standard_normal(3).tolist() for _ in range(n)]
    data["sensed_omega"] = [rng.standard_normal(3).tolist() for _ in range(n)]
    return pd.DataFrame(data)


class TestGenerateSeeds(unittest.TestCase):

    def test_lhs_only_shape(self):
        bounds = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
        seeds = generate_seeds(bounds, n_lhs=10)
        self.assertEqual(seeds.shape, (10, 3))

    def test_bounds_respected(self):
        bounds = [(2.0, 50.0), (0.0, 50.0), (0.0, 5.0), (np.radians(10), np.radians(40)), (0.0, 10.0)]
        seeds = generate_seeds(bounds, n_lhs=30)
        for col, (lo, hi) in enumerate(bounds):
            self.assertTrue(np.all(seeds[:, col] >= lo), f"col {col} below lower bound")
            self.assertTrue(np.all(seeds[:, col] <= hi), f"col {col} above upper bound")

    def test_physical_seed_prepended_as_first_row(self):
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        physical_seed = np.array([0.3, 0.7])
        seeds = generate_seeds(bounds, n_lhs=5, physical_seed=physical_seed)
        np.testing.assert_array_equal(seeds[0], physical_seed)
        self.assertEqual(seeds.shape, (6, 2))  # 1 physical + 5 LHS

    def test_reproducible_with_same_random_seed(self):
        bounds = [(0.0, 1.0), (0.0, 2.0)]
        seeds1 = generate_seeds(bounds, n_lhs=10, random_seed=42)
        seeds2 = generate_seeds(bounds, n_lhs=10, random_seed=42)
        np.testing.assert_array_equal(seeds1, seeds2)

    def test_different_random_seeds_give_different_results(self):
        bounds = [(0.0, 1.0), (0.0, 2.0)]
        seeds1 = generate_seeds(bounds, n_lhs=10, random_seed=1)
        seeds2 = generate_seeds(bounds, n_lhs=10, random_seed=99)
        self.assertFalse(np.allclose(seeds1, seeds2))


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


class TestFittedBlade(unittest.TestCase):

    def setUp(self):
        self.blade = FittedBlade()

    def test_get_chord_at_root(self):
        # interp at y/y_max=0 → weight=0.8; chord = 0.2*0.005 + 0.8*0.024
        chord = self.blade.get_chord(0.0)
        expected = 0.2 * 0.005 + 0.8 * 0.024
        self.assertAlmostEqual(chord, expected)

    def test_get_chord_at_tip_is_min(self):
        # interp at y/y_max=1 → weight=0 → chord = min_length
        chord = self.blade.get_chord(self.blade.y_max)
        self.assertAlmostEqual(chord, 0.005)

    def test_get_chord_values_in_valid_range(self):
        for y in np.linspace(0, self.blade.y_max, 20):
            with self.subTest(y=y):
                chord = self.blade.get_chord(y)
                self.assertGreaterEqual(chord, 0.005)
                self.assertLessEqual(chord, 0.024)

    def test_get_blade_pitch_at_root_is_max(self):
        pitch = self.blade.get_blade_pitch(0.0)
        self.assertAlmostEqual(pitch, np.radians(45))

    def test_get_blade_pitch_at_tip_is_min(self):
        pitch = self.blade.get_blade_pitch(self.blade.y_max)
        self.assertAlmostEqual(pitch, np.radians(17))

    def test_get_blade_pitch_monotone_decreasing(self):
        y_values = np.linspace(0, self.blade.y_max, 10)
        pitches = [self.blade.get_blade_pitch(y) for y in y_values]
        for i in range(len(pitches) - 1):
            self.assertGreaterEqual(pitches[i], pitches[i + 1])


class TestBemtParamFitterMath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fitter = BemtParamFitter()

    def test_compute_body_drag_force_zero_k(self):
        self.fitter.k_body_drag = 0.0
        force = self.fitter.compute_body_drag_force(
            v_i_avg=3.0,
            u_free_avg=np.array([5.0, 0.0, 0.0]),
            r_disk=np.eye(3),
        )
        np.testing.assert_array_equal(force, np.zeros(3))

    def test_compute_body_drag_force_horizontal_disk_no_wind(self):
        # r_disk=I → disk_z = world z; no wind → v_total = v_i_avg; force = k * v_i^2 * z_hat
        self.fitter.k_body_drag = 1.0
        force = self.fitter.compute_body_drag_force(
            v_i_avg=2.0,
            u_free_avg=np.zeros(3),
            r_disk=np.eye(3),
        )
        np.testing.assert_array_almost_equal(force, np.array([0.0, 0.0, 4.0]))

    def test_compute_body_drag_force_downward_wind_increases_drag(self):
        # Wind pointing downward (-z) augments downwash; drag magnitude should be larger
        self.fitter.k_body_drag = 1.0
        r_disk = np.eye(3)
        v_i = 2.0
        force_no_wind = self.fitter.compute_body_drag_force(v_i, np.zeros(3), r_disk)
        force_with_wind = self.fitter.compute_body_drag_force(v_i, np.array([0.0, 0.0, -1.0]), r_disk)
        self.assertGreater(force_with_wind[2], force_no_wind[2])

    def test_compute_residual_force_hover_equilibrium(self):
        # f_inertial exactly balances gravity → residual = 0
        self.fitter.k_body_drag = 0.0
        m = self.fitter.params.m
        g = 9.81
        f_inertial = np.array([0.0, 0.0, m * g])
        residual = self.fitter.compute_residual_force(f_inertial, a_groundtruth=np.zeros(3))
        np.testing.assert_array_almost_equal(residual, np.zeros(3))

    def test_compute_residual_force_upward_acceleration(self):
        # f_inertial = m*(g + a_z) in z → residual = 0
        self.fitter.k_body_drag = 0.0
        m = self.fitter.params.m
        g = 9.81
        a_z = 2.0
        f_inertial = np.array([0.0, 0.0, m * (g + a_z)])
        residual = self.fitter.compute_residual_force(
            f_inertial, a_groundtruth=np.array([0.0, 0.0, a_z])
        )
        np.testing.assert_array_almost_equal(residual, np.zeros(3))

    def test_compute_residual_force_nonzero_means_underthrust(self):
        # Provide less thrust than needed → residual is negative in z
        self.fitter.k_body_drag = 0.0
        m = self.fitter.params.m
        g = 9.81
        f_inertial = np.array([0.0, 0.0, m * g * 0.9])  # 10% short
        residual = self.fitter.compute_residual_force(f_inertial, a_groundtruth=np.zeros(3))
        self.assertLess(residual[2], 0.0)


if __name__ == "__main__":
    unittest.main()
