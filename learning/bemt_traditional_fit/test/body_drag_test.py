import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.body_drag_objective import BodyDragObjective
from learning.bemt_traditional_fit.fitting_config import ModelConfig, FittingConfig
from learning.bemt_traditional_fit.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.manager import FittingManager


def _make_model_config():
    return ModelConfig(
        coarse_n_elements=2, coarse_n_rotation_segments=6, coarse_sample_distance=100,
        fine_n_elements=20, fine_n_rotation_segments=18, fine_sample_distance=20,
    )


def _make_model():
    return BemtModel(APC_8x6(), parameters.PennStateARILab550(), _make_model_config())


def _make_mock_dataset(n=10):
    """Minimal FittingDataset-like mock with n samples."""
    ds = MagicMock()
    ds.u_free_0 = np.zeros((n, 3))
    ds.u_free_1 = np.zeros((n, 3))
    ds.u_free_2 = np.zeros((n, 3))
    ds.u_free_3 = np.zeros((n, 3))
    return ds


class TestBodyDragObjectiveGetLoss(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.model.configure_for_body_drag_fit()
        self.lookup_table = MagicMock()
        self.objective = BodyDragObjective(self.model, self.lookup_table)

    def test_sets_k_body_drag_from_x(self):
        self.model.get_residual_force = MagicMock(return_value=np.zeros(3))
        dataset = _make_mock_dataset(n=10)
        self.objective.get_loss([3.7], [dataset])
        self.assertAlmostEqual(self.model.k_body_drag, 3.7)

    def test_calls_get_residual_force_with_lookup_table(self):
        self.model.get_residual_force = MagicMock(return_value=np.zeros(3))
        dataset = _make_mock_dataset(n=10)
        self.objective.get_loss([1.0], [dataset])
        for call in self.model.get_residual_force.call_args_list:
            _, kwargs = call
            self.assertIs(kwargs.get('lookup_table') or call[0][2], self.lookup_table)
            self.assertTrue(kwargs.get('is_using_lookup_table', False) or call[0][3])

    def test_loss_is_nonnegative(self):
        self.model.get_residual_force = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        dataset = _make_mock_dataset(n=10)
        loss = self.objective.get_loss([1.0], [dataset])
        self.assertGreaterEqual(loss, 0.0)

    def test_zero_residual_gives_zero_loss(self):
        self.model.get_residual_force = MagicMock(return_value=np.zeros(3))
        dataset = _make_mock_dataset(n=10)
        loss = self.objective.get_loss([0.5], [dataset])
        self.assertAlmostEqual(loss, 0.0)

    def test_loss_averages_over_datasets(self):
        residual_a = np.array([0.0, 0.0, 2.0])
        residual_b = np.array([0.0, 0.0, 4.0])
        ds_a = _make_mock_dataset(n=self.model.sample_distance)
        ds_b = _make_mock_dataset(n=self.model.sample_distance)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return residual_a if call_count[0] <= 1 else residual_b

        self.model.get_residual_force = MagicMock(side_effect=side_effect)
        loss = self.objective.get_loss([1.0], [ds_a, ds_b])
        # loss_a = 4.0, loss_b = 16.0, averaged = 10.0
        self.assertAlmostEqual(loss, 10.0)


class TestFittingManagerForBodyDrag(unittest.TestCase):

    def setUp(self):
        self.blade = APC_8x6()
        self.params = parameters.PennStateARILab550()
        self.lookup_table = MagicMock()
        self.fixed_aero_params = (5.0, 1.0, 0.1, np.radians(15))
        self.datasets = []

    def test_creates_bemt_model(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertIsInstance(manager.model, BemtModel)

    def test_model_has_1d_bounds(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertEqual(manager.engine.decision_var.bounds, [(0.0, 10.0)])

    def test_model_has_k_body_drag_parameter_name(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertEqual(manager.engine.decision_var.search_names, ("k_body_drag",))

    def test_fixed_aero_params_applied_to_blade(self):
        cl_1, cl_2, cd, alpha_0 = self.fixed_aero_params
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertAlmostEqual(manager.model.blade.cl_1, cl_1)
        self.assertAlmostEqual(manager.model.blade.cl_2, cl_2)
        self.assertAlmostEqual(manager.model.blade.cd, cd)
        self.assertAlmostEqual(manager.model.blade.alpha_0, alpha_0)

    def test_objective_is_body_drag_objective(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertIsInstance(manager.engine.objective, BodyDragObjective)

    def test_objective_holds_lookup_table(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertIs(manager.engine.objective.lookup_table, self.lookup_table)

    def test_engine_is_fitting_engine(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertIsInstance(manager.engine, FittingEngine)

    def test_init_guess_stored_as_array(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
            init_guess=[2.5],
        )
        np.testing.assert_array_equal(manager.init_guess, [2.5])

    def test_no_init_guess_by_default(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        self.assertIsNone(manager.init_guess)

    def test_run_multiseed_delegates_to_engine_fit(self):
        manager = FittingManager.for_body_drag(
            self.blade, self.params, self.datasets,
            self.lookup_table, self.fixed_aero_params,
        )
        fake_result = np.array([1.5])
        with patch.object(manager.engine, 'fit', return_value=fake_result) as mock_fit:
            result = manager.run(is_multiseed=True)
        mock_fit.assert_called_once_with(self.datasets, custom_init=None)
        self.assertIs(result, fake_result)


if __name__ == "__main__":
    unittest.main()
