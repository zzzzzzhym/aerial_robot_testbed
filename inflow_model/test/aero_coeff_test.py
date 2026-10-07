import unittest
import numpy as np

from airfoil.aero_coeff import Coeffecients


class TestCoeffecients(unittest.TestCase):
    def setUp(self):
        self.cl_1 = 5.3
        self.cl_2 = 1.7
        self.cd = 1.8

    def _sigma(self, coeff, alpha):
        return coeff.sigma.compute(alpha)

    def test_zero_lift_angle_gives_zero_attached_lift(self):
        """At the attached-flow limit (sigma -> 0), lift vanishes at alpha_zero_lift."""
        alpha_zero_lift = np.radians(-3.0)
        coeff = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             alpha_zero_lift=alpha_zero_lift)
        alpha = alpha_zero_lift
        # geometric sin(alpha)*cos(alpha) term is tiny but nonzero; subtract it to
        # isolate the attached linear term, which must be exactly zero here.
        sigma = self._sigma(coeff, alpha)
        stalled_term = sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha)
        self.assertAlmostEqual(coeff.get_cl(alpha) - stalled_term, 0.0)

    def test_cl_matches_reference_formula(self):
        alpha_zero_lift = np.radians(2.0)
        coeff = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             alpha_zero_lift=alpha_zero_lift)
        alpha = np.radians(10.0)
        sigma = self._sigma(coeff, alpha)
        expected = ((1 - sigma) * self.cl_1 * (alpha - alpha_zero_lift)
                    + sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha))
        self.assertAlmostEqual(coeff.get_cl(alpha), expected)

    def test_sigma_uses_geometric_alpha_not_shifted(self):
        """Transition must depend on geometric alpha, independent of alpha_zero_lift."""
        c0 = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd, alpha_zero_lift=0.0)
        c1 = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                          alpha_zero_lift=np.radians(5.0))
        alpha = np.radians(15.0)
        self.assertAlmostEqual(self._sigma(c0, alpha), self._sigma(c1, alpha))

    def test_min_drag_angle_shifts_drag_minimum(self):
        """Profile-drag (sin^2) term is minimized at alpha = alpha_d_min."""
        alpha_d_min = np.radians(4.0)
        coeff = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             cd_0=0.0, cp=0.0, alpha_d_min=alpha_d_min)
        cd_at_min = coeff.get_cd(alpha_d_min, u=10.0, chord=0.1)
        cd_off_min = coeff.get_cd(alpha_d_min + np.radians(5.0), u=10.0, chord=0.1)
        self.assertAlmostEqual(cd_at_min, 0.0)
        self.assertGreater(cd_off_min, cd_at_min)

    def test_cd_matches_reference_formula(self):
        alpha_d_min = np.radians(1.5)
        cp = 1.328
        cd_0 = 0.01
        coeff = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd,
                             cd_0=cd_0, cp=cp, alpha_d_min=alpha_d_min)
        alpha, u, chord = np.radians(12.0), 10.0, 0.1
        rn = np.maximum(Coeffecients.get_reynolds_number(u, chord), 1)
        expected = (self.cd * np.sin(alpha - alpha_d_min) ** 2
                    + 2 * 1.02 * cp / np.sqrt(rn) + cd_0)
        self.assertAlmostEqual(coeff.get_cd(alpha, u, chord), expected)

    def test_defaults_reduce_to_unshifted_behavior(self):
        """With both new angles at 0, cl/cd reduce to the un-shifted formulation."""
        coeff = Coeffecients(cl_1=self.cl_1, cl_2=self.cl_2, cd=self.cd)
        alpha = np.radians(8.0)
        sigma = coeff.sigma.compute(alpha)
        expected_cl = ((1 - sigma) * self.cl_1 * alpha
                       + sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha))
        self.assertAlmostEqual(coeff.get_cl(alpha), expected_cl)


if __name__ == "__main__":
    unittest.main()
