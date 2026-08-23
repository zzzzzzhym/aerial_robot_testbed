import unittest

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.objective import ObjectiveWeights, FittingObjective


class TestObjectiveWeights(unittest.TestCase):

    def test_horizontal_weight_stored(self):
        w = ObjectiveWeights(horizontal_weight=3.5)
        self.assertAlmostEqual(w.horizontal_weight, 3.5)

    def test_fitting_objective_uses_weights_when_provided(self):
        model = BemtModel(APC_8x6(), parameters.PennStateARILab550())
        w = ObjectiveWeights(horizontal_weight=99.0)
        obj = FittingObjective(model, weights=w)
        self.assertAlmostEqual(obj.weights.horizontal_weight, 99.0)

    def test_fitting_objective_falls_back_to_model_horizontal_weight(self):
        model = BemtModel(APC_8x6(), parameters.PennStateARILab550())
        obj = FittingObjective(model)
        self.assertIsNone(obj.weights)
        # horizontal_weight used at loss-compute time comes from model
        self.assertIsNotNone(model.horizontal_weight)


if __name__ == "__main__":
    unittest.main()
