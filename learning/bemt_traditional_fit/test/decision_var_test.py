import unittest

import numpy as np

from learning.bemt_traditional_fit import decision_var as dv


class TestDecisionVarBase(unittest.TestCase):

    def test_identity_conversions(self):
        d = dv.DecisionVar(search_names=("a", "b"), physical_names=("a", "b"),
                           bounds=[(0.0, 1.0), (0.0, 1.0)])
        x = np.array([0.3, 0.7])
        np.testing.assert_array_almost_equal(d.to_physical(x), x)
        np.testing.assert_array_almost_equal(d.to_search(x), x)

    def test_clip(self):
        d = dv.DecisionVar(search_names=("a",), physical_names=("a",), bounds=[(0.0, 1.0)])
        np.testing.assert_array_almost_equal(d.clip([5.0]), [1.0])
        np.testing.assert_array_almost_equal(d.clip([-5.0]), [0.0])

    def test_format_marks_angle_params_in_degrees(self):
        d = dv.DecisionVar(search_names=("cl_1", "alpha_0"), physical_names=("cl_1", "alpha_0"),
                           bounds=[(0.0, 10.0), (0.0, 1.0)], angle_params=("alpha_0",))
        s = d.format([3.0, np.radians(12.0)])
        self.assertIn("cl_1=3.000", s)
        self.assertIn("alpha_0=12.000deg", s)


class TestFactories(unittest.TestCase):

    def test_full_vehicle_is_physical_identity(self):
        d = dv.full_vehicle()
        self.assertEqual(d.search_names, d.physical_names)
        self.assertEqual(len(d.bounds), 5)
        x = np.array([5.0, 1.7, 1.8, np.radians(20.0), 0.5])
        np.testing.assert_array_almost_equal(d.to_physical(x), x)

    def test_body_drag_single_parameter(self):
        d = dv.body_drag()
        self.assertEqual(d.search_names, ("k_body_drag",))
        self.assertEqual(d.bounds, [(0.0, 10.0)])


class TestOffsetAugmentedAeroCoeffDecisionVar(unittest.TestCase):

    def setUp(self):
        self.dv = dv.OffsetAugmentedAeroCoeffDecisionVar()

    def test_names_and_bounds_lengths(self):
        self.assertEqual(len(self.dv.search_names), 6)
        self.assertEqual(len(self.dv.physical_names), 6)
        self.assertEqual(len(self.dv.bounds), 6)
        self.assertEqual(self.dv.search_names,
                         ("cl_1", "k_cl_2", "cd", "delta_alpha", "alpha_zero_lift", "alpha_d_min"))

    def test_format_shows_both_physical_and_decision(self):
        # search vector: cl_1, k_cl_2, cd, delta_alpha, alpha_zero_lift, alpha_d_min
        s = self.dv.format([7.0, 0.4, 1.5, np.radians(12.0), np.radians(-3.0), np.radians(4.0)])
        # physical coefficients (alpha_0 = alpha_zero_lift + delta_alpha = -3 + 12 = 9 deg)
        self.assertIn("cl_2=", s)
        self.assertIn("alpha_0=9.000deg", s)
        # decision variables alongside
        self.assertIn("[decision:", s)
        self.assertIn("k_cl_2=0.400", s)
        self.assertIn("delta_alpha=12.000deg", s)

    def test_k_cl_2_bounds_are_zero_to_099(self):
        self.assertEqual(self.dv.bounds[1], (0.0, 0.99))

    def test_delta_alpha_lower_bound_strictly_positive(self):
        self.assertGreater(self.dv.bounds[3][0], 0.0)

    def test_to_physical_formula(self):
        cl_1, k_cl_2, cd = 5.0, 0.5, 1.2
        delta_alpha, azl, admin = np.radians(10.0), np.radians(-2.0), np.radians(3.0)
        phys = self.dv.to_physical((cl_1, k_cl_2, cd, delta_alpha, azl, admin))
        self.assertAlmostEqual(phys[0], cl_1)        # cl_1 passthrough
        self.assertAlmostEqual(phys[2], cd)          # cd passthrough
        self.assertAlmostEqual(phys[4], azl)         # alpha_zero_lift passthrough
        self.assertAlmostEqual(phys[5], admin)       # alpha_d_min passthrough
        self.assertAlmostEqual(phys[1], 2.0 * k_cl_2 * cl_1 * delta_alpha)  # cl_2
        self.assertAlmostEqual(phys[3], azl + delta_alpha)                  # alpha_0

    def test_constraint_structurally_enforced(self):
        # For any decision in bounds, cl_2 < 2*cl_1*(alpha_0 - alpha_zero_lift).
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in self.dv.bounds])
        hi = np.array([b[1] for b in self.dv.bounds])
        for _ in range(200):
            decision = lo + rng.random(6) * (hi - lo)
            cl_1, cl_2, _, alpha_0, azl, _ = self.dv.to_physical(decision)
            self.assertLess(cl_2, 2.0 * cl_1 * (alpha_0 - azl) + 1e-12)

    def test_round_trip_to_search_to_physical(self):
        physical = np.array([7.0, 2.1, 1.5, np.radians(12.0), np.radians(-3.0), np.radians(4.0)])
        # feasible physical -> search (no clipping) -> physical recovers it
        np.testing.assert_array_almost_equal(
            self.dv.to_physical(self.dv.to_search(physical)), physical)

    def test_to_search_clips_nonpositive_delta_alpha(self):
        # alpha_0 <= alpha_zero_lift (non-positive attached width) -> clipped, not rejected.
        physical = np.array([5.0, 1.0, 1.0, np.radians(-5.0), np.radians(-2.0), 0.0])
        search = self.dv.to_search(physical)
        delta_lo, delta_hi = self.dv.bounds[3]
        self.assertGreaterEqual(search[3], delta_lo)  # delta_alpha clipped up to its bound
        for value, (lo, hi) in zip(search, self.dv.bounds):
            self.assertGreaterEqual(value, lo)
            self.assertLessEqual(value, hi)

    def test_to_search_clips_infeasible_physical(self):
        # cl_2 large enough to violate the shape constraint -> k_cl_2 clipped to 0.99.
        physical = np.array([5.0, 50.0, 1.0, np.radians(10.0), 0.0, 0.0])
        search = self.dv.to_search(physical)
        for value, (lo, hi) in zip(search, self.dv.bounds):
            self.assertGreaterEqual(value, lo)
            self.assertLessEqual(value, hi)


if __name__ == "__main__":
    unittest.main()
