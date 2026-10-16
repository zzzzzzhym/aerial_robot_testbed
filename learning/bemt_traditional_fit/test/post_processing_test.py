import unittest
from unittest.mock import patch

import numpy as np

from inflow_model.blade_params import P600_Blade
import learning.bemt_traditional_fit.post_processing as pp
from learning.bemt_traditional_fit import decision_var as dv


class TestMakeLookupTable(unittest.TestCase):
    """make_lookup_table installs the right aero coefficients before building the table."""

    def _make(self, fitted_params):
        blade = P600_Blade()
        with patch.object(pp.PropellerLookupTable.Maker, "make_propeller_lookup_table") as mk:
            pp.make_lookup_table(fitted_params, blade, "unit_tmp", is_hover_only=True)
        return blade, mk

    def test_base_six_vector_sets_classical_coeffs(self):
        blade, mk = self._make([5.3, 1.7, 1.8, np.radians(20.0), np.radians(-2.0), np.radians(3.0)])
        coeff = blade.aero_coeff
        self.assertFalse(hasattr(coeff, "cl_offsets"))  # stayed classical
        self.assertAlmostEqual(coeff.cl_2, 1.7)
        self.assertAlmostEqual(coeff.alpha_0, np.radians(20.0))
        self.assertTrue(mk.called)

    def test_band_vector_installs_correction_offsets(self):
        n = len(dv.BAND_CORRECTION_KNOTS)
        cl_off = np.linspace(0.1, 0.4, n)
        cd_off = np.linspace(-0.1, 0.1, n)
        params = np.concatenate(
            [[6.0, 3.0, 1.5, np.radians(8.0), np.radians(-3.0), np.radians(4.0)], cl_off, cd_off])
        blade, mk = self._make(params)
        coeff = blade.aero_coeff
        self.assertTrue(hasattr(coeff, "cl_offsets"))  # swapped to band-corrected
        np.testing.assert_array_almost_equal(coeff.cl_offsets, cl_off)
        np.testing.assert_array_almost_equal(coeff.cd_offsets, cd_off)
        np.testing.assert_array_almost_equal(coeff.cl_knots, dv.BAND_CORRECTION_KNOTS)
        self.assertAlmostEqual(coeff.alpha_0, np.radians(8.0))
        self.assertTrue(mk.called)

    def test_band_vector_wrong_length_raises(self):
        n = len(dv.BAND_CORRECTION_KNOTS)
        with self.assertRaises(ValueError):
            pp.make_lookup_table(np.zeros(6 + 2 * n - 1), P600_Blade(), "unit_tmp")


if __name__ == "__main__":
    unittest.main()
