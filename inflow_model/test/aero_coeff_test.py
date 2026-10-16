import unittest
import numpy as np
import yaml

from airfoil.aero_coeff import (
    AeroCoefficients, ClassicalAeroCoefficients, BandCorrectedAeroCoefficients,
    NeuroBEMCoefficients,
)
from blade_params import APC_8x6


class TestClassicalAeroCoefficients(unittest.TestCase):
    def setUp(self):
        self.cl_1 = 5.3
        self.cl_2 = 1.7
        self.cd = 1.8

    def _sigma(self, coeff, alpha):
        return coeff.sigma.compute(alpha)

    def test_zero_lift_angle_gives_zero_attached_lift(self):
        """At the attached-flow limit (sigma -> 0), lift vanishes at alpha_zero_lift."""
        alpha_zero_lift = np.radians(-3.0)
        coeff = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             alpha_zero_lift=alpha_zero_lift)
        alpha = alpha_zero_lift
        # geometric sin(alpha)*cos(alpha) term is tiny but nonzero; subtract it to
        # isolate the attached linear term, which must be exactly zero here.
        sigma = self._sigma(coeff, alpha)
        stalled_term = sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha)
        self.assertAlmostEqual(coeff.get_cl(alpha) - stalled_term, 0.0)

    def test_cl_matches_reference_formula(self):
        alpha_zero_lift = np.radians(2.0)
        coeff = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             alpha_zero_lift=alpha_zero_lift)
        alpha = np.radians(10.0)
        sigma = self._sigma(coeff, alpha)
        expected = ((1 - sigma) * self.cl_1 * (alpha - alpha_zero_lift)
                    + sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha))
        self.assertAlmostEqual(coeff.get_cl(alpha), expected)

    def test_sigma_uses_geometric_alpha_not_shifted(self):
        """Transition must depend on geometric alpha, independent of alpha_zero_lift."""
        c0 = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd, alpha_zero_lift=0.0)
        c1 = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                          alpha_zero_lift=np.radians(5.0))
        alpha = np.radians(15.0)
        self.assertAlmostEqual(self._sigma(c0, alpha), self._sigma(c1, alpha))

    def test_min_drag_angle_shifts_drag_minimum(self):
        """Profile-drag (sin^2) term is minimized at alpha = alpha_d_min."""
        alpha_d_min = np.radians(4.0)
        coeff = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             cd_0=0.0, cp=0.0, alpha_d_min=alpha_d_min)
        cd_at_min = coeff.get_cd(alpha_d_min, u=10.0, chord=0.1)
        cd_off_min = coeff.get_cd(alpha_d_min + np.radians(5.0), u=10.0, chord=0.1)
        self.assertAlmostEqual(cd_at_min, 0.0)
        self.assertGreater(cd_off_min, cd_at_min)

    def test_cd_matches_reference_formula(self):
        alpha_d_min = np.radians(1.5)
        cp = 1.328
        cd_0 = 0.01
        coeff = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             cd_0=cd_0, cp=cp, alpha_d_min=alpha_d_min)
        alpha, u, chord = np.radians(12.0), 10.0, 0.1
        rn = np.maximum(ClassicalAeroCoefficients.get_reynolds_number(u, chord), 1)
        expected = (self.cd * np.sin(alpha - alpha_d_min) ** 2
                    + 2 * 1.02 * cp / np.sqrt(rn) + cd_0)
        self.assertAlmostEqual(coeff.get_cd(alpha, u, chord), expected)

    def test_defaults_reduce_to_unshifted_behavior(self):
        """With both new angles at 0, cl/cd reduce to the un-shifted formulation."""
        coeff = ClassicalAeroCoefficients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd)
        alpha = np.radians(8.0)
        sigma = coeff.sigma.compute(alpha)
        expected_cl = ((1 - sigma) * self.cl_1 * alpha
                       + sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha))
        self.assertAlmostEqual(coeff.get_cl(alpha), expected_cl)


class TestAeroCoefficientHierarchy(unittest.TestCase):
    def test_abc_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            AeroCoefficients()

    def test_classical_is_aero_coefficients(self):
        self.assertIsInstance(ClassicalAeroCoefficients(), AeroCoefficients)

    def test_update_fit_params_keeps_cd0_cp_and_refreshes_sigma(self):
        coeff = ClassicalAeroCoefficients(cd_0=0.02, cp=1.5)
        coeff.update_fit_params(7.0, 2.0, 1.0, np.radians(15.0), np.radians(-2.0), np.radians(3.0))
        self.assertAlmostEqual(coeff.cl_1, 7.0)
        self.assertAlmostEqual(coeff.alpha_0, np.radians(15.0))
        self.assertAlmostEqual(coeff.alpha_zero_lift, np.radians(-2.0))
        self.assertAlmostEqual(coeff.cd_0, 0.02)   # fixed params preserved
        self.assertAlmostEqual(coeff.cp, 1.5)
        self.assertAlmostEqual(coeff.sigma.alpha_0, np.radians(15.0))  # sigma rebuilt

    def test_update_fit_params_keeps_angles_when_omitted(self):
        coeff = ClassicalAeroCoefficients(alpha_zero_lift=np.radians(1.0), alpha_d_min=np.radians(2.0))
        coeff.update_fit_params(7.0, 2.0, 1.0, np.radians(15.0))
        self.assertAlmostEqual(coeff.alpha_zero_lift, np.radians(1.0))
        self.assertAlmostEqual(coeff.alpha_d_min, np.radians(2.0))


class TestBandCorrectedAeroCoefficients(unittest.TestCase):
    def test_zero_offsets_reduce_to_classical(self):
        base = ClassicalAeroCoefficients()
        band = BandCorrectedAeroCoefficients()
        for alpha in np.radians([-20.0, 0.0, 8.0, 25.0, 50.0]):
            self.assertAlmostEqual(band.get_cl(alpha), base.get_cl(alpha))
            self.assertAlmostEqual(band.get_cd(alpha, 10.0, 0.1), base.get_cd(alpha, 10.0, 0.1))

    def test_cl_offset_applied_within_band(self):
        knots = np.radians([-180.0, 0.0, 10.0, 90.0])  # no +-180 duplicate (periodic interp)
        band = BandCorrectedAeroCoefficients(cl_knots=knots, cl_offsets=[0.0, 0.0, 0.5, 0.0])
        base = ClassicalAeroCoefficients()
        a = np.radians(10.0)
        self.assertAlmostEqual(band.get_cl(a) - base.get_cl(a), 0.5)  # at the knot
        # linear interp between knots stays within the offset band
        mid = np.radians(5.0)
        self.assertLessEqual(band.get_cl(mid) - base.get_cl(mid), 0.5 + 1e-12)

    def test_correction_is_periodic_pm180(self):
        # +180 deg must equal -180 deg (angle of attack wraps).
        knots = np.radians([-180.0, -45.0, 0.0, 45.0, 90.0])
        band = BandCorrectedAeroCoefficients(
            cl_knots=knots, cl_offsets=[0.3, -0.2, 0.1, 0.25, -0.15],
            cd_knots=knots, cd_offsets=[0.05, -0.1, 0.0, 0.1, -0.05])
        self.assertAlmostEqual(band.get_cl(np.pi), band.get_cl(-np.pi))
        self.assertAlmostEqual(band.get_cd(np.pi, 10.0, 0.1), band.get_cd(-np.pi, 10.0, 0.1))
        # at the -180 knot the CL offset is the stored value (0.3)
        self.assertAlmostEqual(band.get_cl(np.pi) - ClassicalAeroCoefficients().get_cl(np.pi), 0.3)

    def test_cd_clamped_nonnegative(self):
        knots = np.radians([-180.0, 0.0, 90.0])
        # a large negative cd offset would drive cd below zero; must clamp
        band = BandCorrectedAeroCoefficients(cd_knots=knots, cd_offsets=[-5.0, -5.0, -5.0])
        self.assertGreaterEqual(band.get_cd(np.radians(5.0), 10.0, 0.1), 0.0)


class TestToDictAndDescribe(unittest.TestCase):
    def test_classical_to_dict_is_yaml_serializable(self):
        d = ClassicalAeroCoefficients(cl_1=5.3, cl_2=1.7).to_dict()
        self.assertEqual(d["type"], "classical")
        self.assertIn("alpha_zero_lift", d)
        self.assertIsInstance(d["cl_1"], float)
        # round-trips through safe YAML (no numpy types)
        self.assertEqual(yaml.safe_load(yaml.safe_dump(d))["cl_1"], 5.3)

    def test_band_to_dict_includes_knots_and_offsets(self):
        knots = np.radians([-180.0, -45.0, 0.0, 45.0, 90.0])
        band = BandCorrectedAeroCoefficients(
            cl_knots=knots, cl_offsets=[0.1, -0.2, 0.0, 0.3, -0.1],
            cd_knots=knots, cd_offsets=[0.0, 0.05, 0.0, -0.05, 0.0])
        d = band.to_dict()
        self.assertEqual(d["type"], "band_corrected")
        self.assertEqual(len(d["cl_knots"]), 5)
        self.assertEqual(d["cl_offsets"][0], 0.1)
        yaml.safe_dump(d)  # must not raise on numpy-derived lists

    def test_neurobem_to_dict(self):
        d = NeuroBEMCoefficients().to_dict()
        self.assertEqual(d["type"], "neurobem")
        yaml.safe_dump(d)

    def test_describe_lists_params(self):
        text = ClassicalAeroCoefficients().describe()
        self.assertIn("cl_1 = ", text)
        self.assertIn("alpha_0 = ", text)


class TestBladeAeroCoefficient(unittest.TestCase):
    def test_blade_exposes_aero_coeff(self):
        blade = APC_8x6()
        self.assertIsInstance(blade.aero_coeff, ClassicalAeroCoefficients)

    def test_get_aero_coeff_is_spanwise_uniform(self):
        blade = APC_8x6()
        self.assertIs(blade.get_aero_coeff(0.0), blade.aero_coeff)
        self.assertIs(blade.get_aero_coeff(blade.y_max), blade.aero_coeff)

    def test_aero_coeff_can_be_swapped(self):
        blade = APC_8x6()
        new_coeff = BandCorrectedAeroCoefficients()
        blade.aero_coeff = new_coeff
        self.assertIs(blade.get_aero_coeff(0.0), new_coeff)

    def test_blade_has_no_loose_aero_attributes(self):
        blade = APC_8x6()
        self.assertFalse(hasattr(blade, "cl_1"))
        self.assertFalse(hasattr(blade, "alpha_0"))


if __name__ == "__main__":
    unittest.main()
