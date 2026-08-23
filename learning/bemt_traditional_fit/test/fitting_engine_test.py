import unittest
from unittest.mock import patch
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.objective import FittingObjective
from learning.bemt_traditional_fit.fitting_engine import (
    FittingEngine, format_full_params, format_single_rotor_params
)


class TestFittingEngine(unittest.TestCase):

    def setUp(self):
        blade = APC_8x6()
        params = parameters.PennStateARILab550()
        self.model = BemtModel(blade, params)
        self.objective = FittingObjective(self.model)

    def test_bounds_has_five_params(self):
        self.assertEqual(len(BemtModel.BOUNDS), 5)

    def test_construction_stores_model_and_objective(self):
        engine = FittingEngine(self.model, self.objective)
        self.assertIs(engine.model, self.model)
        self.assertIs(engine.objective, self.objective)

    def test_default_formatter_used_when_none_given(self):
        engine = FittingEngine(self.model, self.objective)
        result = engine.parameter_formatter(np.array([1.0, 2.0]))
        self.assertIn("x0=", result)
        self.assertIn("x1=", result)

    def test_custom_formatter_stored(self):
        fmt = lambda x: "custom"
        engine = FittingEngine(self.model, self.objective, parameter_formatter=fmt)
        self.assertIs(engine.parameter_formatter, fmt)

    def test_fit_custom_init_included_in_seeds(self):
        engine = FittingEngine(self.model, self.objective)
        custom = np.array([5.0, 2.0, 2.0, 0.4, 0.0])
        captured = []

        def fake_screen(seeds, datasets, n_keep=3):
            captured.extend(seeds)
            raise StopIteration

        with patch.object(engine, '_screen_seeds', side_effect=fake_screen):
            try:
                engine.fit([], custom_init=custom)
            except StopIteration:
                pass

        self.assertTrue(any(np.allclose(s, custom) for s in captured))


class TestFormatters(unittest.TestCase):

    def test_format_full_params_contains_k_body_drag(self):
        x = [5.3, 1.7, 1.8, np.radians(20.0), 0.5]
        result = format_full_params(x)
        self.assertIn("k_body_drag", result)
        self.assertIn("alpha_0", result)

    def test_format_single_rotor_params_has_four_fields(self):
        x = [5.3, 1.7, 1.8, np.radians(20.0)]
        result = format_single_rotor_params(x)
        self.assertIn("cl_1", result)
        self.assertIn("alpha_0", result)
        self.assertNotIn("k_body_drag", result)


if __name__ == "__main__":
    unittest.main()
