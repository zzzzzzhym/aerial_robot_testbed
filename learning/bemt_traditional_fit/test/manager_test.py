import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.manager import FittingManager


class TestFittingManagerForFullVehicle(unittest.TestCase):

    def setUp(self):
        self.blade = APC_8x6()
        self.params = parameters.PennStateARILab550()
        self.datasets = []

    def test_for_full_vehicle_creates_bemt_model(self):
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets)
        self.assertIsInstance(manager.model, BemtModel)

    def test_for_full_vehicle_creates_fitting_engine(self):
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets)
        self.assertIsInstance(manager.engine, FittingEngine)

    def test_datasets_stored(self):
        datasets = [object(), object()]
        manager = FittingManager.for_full_vehicle(self.blade, self.params, datasets)
        self.assertIs(manager.datasets, datasets)

    def test_no_init_guess_by_default(self):
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets)
        self.assertIsNone(manager.init_guess)

    def test_init_guess_stored_as_array(self):
        guess = [5.0, 2.0, 2.0, 0.4, 0.0]
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets, init_guess=guess)
        np.testing.assert_array_equal(manager.init_guess, guess)

    def test_run_multiseed_delegates_to_engine_fit(self):
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets)
        fake_result = np.zeros(5)
        with patch.object(manager.engine, 'fit', return_value=fake_result) as mock_fit:
            result = manager.run(is_multiseed=True)
        mock_fit.assert_called_once_with(self.datasets, custom_init=None)
        self.assertIs(result, fake_result)

    def test_run_multiseed_passes_init_guess(self):
        guess = [5.0, 2.0, 2.0, 0.4, 0.0]
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets, init_guess=guess)
        with patch.object(manager.engine, 'fit', return_value=np.zeros(5)) as mock_fit:
            manager.run(is_multiseed=True)
        _, kwargs = mock_fit.call_args
        np.testing.assert_array_equal(kwargs['custom_init'], guess)

    def test_run_single_delegates_to_engine_fit_single(self):
        guess = [5.0, 2.0, 2.0, 0.4, 0.0]
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets, init_guess=guess)
        fake_result = np.zeros(5)
        with patch.object(manager.engine, 'fit_single', return_value=fake_result) as mock_fit_single:
            result = manager.run(is_multiseed=False, is_fine_tune=True)
        args, kwargs = mock_fit_single.call_args
        self.assertIs(args[0], self.datasets)
        self.assertTrue(kwargs.get('is_fine_tune'))
        self.assertIs(result, fake_result)

    def test_run_single_uses_init_guess_as_seed(self):
        guess = [5.0, 2.0, 2.0, 0.4, 0.0]
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets, init_guess=guess)
        with patch.object(manager.engine, 'fit_single', return_value=np.zeros(5)) as mock_fit_single:
            manager.run(is_multiseed=False)
        seed_gen = mock_fit_single.call_args[1]['seed_generator']
        seeds = seed_gen.get_seeds(None)
        np.testing.assert_array_equal(seeds[0], guess)

    def test_run_single_no_init_guess_raises(self):
        manager = FittingManager.for_full_vehicle(self.blade, self.params, self.datasets)
        with self.assertRaises(ValueError):
            manager.run(is_multiseed=False)

    def test_plot_delegates_to_fit_plotter(self):
        fake_dataset = MagicMock()
        manager = FittingManager.for_full_vehicle(self.blade, self.params, [fake_dataset])
        fake_figs = (MagicMock(), MagicMock())
        with patch('learning.bemt_traditional_fit.fit_plotter.FitPlotter.plot_the_fit',
                   return_value=fake_figs) as mock_plot:
            result = manager.plot(dataset_idx=0, sample_step=2)
        mock_plot.assert_called_once_with(
            manager.model, fake_dataset,
            lookup_table=None, is_using_lookup_table=False, sample_step=2,
        )
        self.assertIs(result, fake_figs)


class TestFittingManagerForSingleRotor(unittest.TestCase):

    def setUp(self):
        self.blade = APC_8x6()
        self.datasets = []

    def test_for_single_rotor_creates_single_rotor_model(self):
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False, datasets=self.datasets)
        self.assertIsInstance(manager.model, SingleRotorBemtModel)

    def test_for_single_rotor_creates_fitting_engine(self):
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False, datasets=self.datasets)
        self.assertIsInstance(manager.engine, FittingEngine)

    def test_for_single_rotor_creates_single_rotor_objective(self):
        from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False, datasets=self.datasets)
        self.assertIsInstance(manager.engine.objective, SingleRotorObjective)

    def test_datasets_stored(self):
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False, datasets=self.datasets)
        self.assertIs(manager.datasets, self.datasets)

    def test_no_init_guess_by_default(self):
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False, datasets=self.datasets)
        self.assertIsNone(manager.init_guess)

    def test_run_multiseed_delegates_to_engine_fit(self):
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False, datasets=self.datasets)
        fake_result = np.zeros(4)
        with patch.object(manager.engine, 'fit', return_value=fake_result) as mock_fit:
            result = manager.run(is_multiseed=True)
        mock_fit.assert_called_once_with(self.datasets, custom_init=None)
        self.assertIs(result, fake_result)


if __name__ == "__main__":
    unittest.main()
