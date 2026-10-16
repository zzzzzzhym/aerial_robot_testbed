import unittest
from unittest.mock import patch
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit import decision_var as dv
from learning.bemt_traditional_fit.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.seed_generator import SingleSeedGenerator
from learning.bemt_traditional_fit.solver import Solver
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.manager import FittingManager


class _FakeModel:
    """Minimal model for engine-level subspace tests (no BET)."""

    def __init__(self):
        self.applied = None

    def adjust_resolution(self, is_fine_tune):
        pass

    def apply_params(self, x):
        self.applied = np.asarray(x, dtype=float)


class _QuadraticObjective:
    """Loss = sum((physical - target)^2); minimized at target."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def get_loss(self, physical, datasets):
        d = np.asarray(physical, dtype=float) - self.target
        return float(d @ d)


def _simple_engine(objective, bounds):
    decision_var = dv.DecisionVar(
        search_names=tuple(f"p{i}" for i in range(len(bounds))),
        physical_names=tuple(f"p{i}" for i in range(len(bounds))),
        bounds=bounds,
    )
    s = lambda: Solver(bounds, options={'disp': False, 'fatol': 1e-6})
    return FittingEngine(_FakeModel(), objective, decision_var, s(), s(), s(), s())


class TestFitSingleSubspace(unittest.TestCase):
    """fit_single restricted to a subset of the decision vector via free_indices."""

    def test_frozen_indices_unchanged_free_index_optimized(self):
        bounds = [(0.0, 10.0)] * 3
        engine = _simple_engine(_QuadraticObjective([2.0, 5.0, 8.0]), bounds)
        seed_gen = SingleSeedGenerator([0.0, 0.0, 0.0], free_indices=[1])
        result = engine.fit_single(seed_gen, [], is_fine_tune=True)
        # Only index 1 was free: 0 and 2 stay at the seed, 1 moves toward 5.
        self.assertAlmostEqual(result[0], 0.0, places=6)
        self.assertAlmostEqual(result[2], 0.0, places=6)
        self.assertAlmostEqual(result[1], 5.0, places=1)

    def test_multiple_free_indices_optimized(self):
        bounds = [(0.0, 10.0)] * 3
        engine = _simple_engine(_QuadraticObjective([2.0, 5.0, 8.0]), bounds)
        seed_gen = SingleSeedGenerator(np.zeros(3), free_indices=[0, 2])
        result = engine.fit_single(seed_gen, [], is_fine_tune=True)
        self.assertAlmostEqual(result[1], 0.0, places=6)   # frozen
        self.assertAlmostEqual(result[0], 2.0, places=1)
        self.assertAlmostEqual(result[2], 8.0, places=1)

    def test_no_free_indices_optimizes_everything(self):
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        result = engine.fit_single(SingleSeedGenerator(np.zeros(2)), [], is_fine_tune=True)
        self.assertAlmostEqual(result[0], 3.0, places=1)
        self.assertAlmostEqual(result[1], 4.0, places=1)

    def test_result_applied_to_model(self):
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        result = engine.fit_single(SingleSeedGenerator(np.zeros(2)), [], is_fine_tune=True)
        np.testing.assert_array_equal(engine.model.applied, result)

    def test_budget_comes_from_config_solver(self):
        # The reduced solver inherits the (config-built) template solver's maxiter.
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        engine.single_fine_solver = Solver(bounds, maxiter=7, options={'disp': False})
        captured = {}
        real_run = Solver.run

        def spy(self, *a, **k):
            captured['maxiter'] = self.options['maxiter']
            return real_run(self, *a, **k)

        with patch.object(Solver, 'run', spy):
            engine.fit_single(SingleSeedGenerator(np.zeros(2), free_indices=[0]), [], is_fine_tune=True)
        self.assertEqual(captured['maxiter'], 7)


class TestRunStaged(unittest.TestCase):

    def setUp(self):
        self.blade = APC_8x6()
        self.datasets = []

    def test_factory_builds_staged_manager(self):
        manager = FittingManager.for_single_rotor_staged(self.blade, is_ccw_rotor0=False,
                                                         datasets=self.datasets)
        self.assertIsInstance(manager.model, SingleRotorBemtModel)
        self.assertIsNone(manager.engine)
        self.assertIsNotNone(manager._config)

    def test_run_staged_without_factory_raises(self):
        manager = FittingManager.for_single_rotor(self.blade, is_ccw_rotor0=False,
                                                  datasets=self.datasets)
        with self.assertRaises(RuntimeError):
            manager.run_staged()

    def test_run_staged_three_stage_sequence(self):
        manager = FittingManager.for_single_rotor_staged(self.blade, is_ccw_rotor0=False,
                                                         datasets=self.datasets)
        physical_base = np.array([5.0, 2.0, 1.0, 0.4, 0.0, 0.0])

        single_calls = []

        def fake_fit(self, seed_generator, datasets):
            return physical_base.copy()

        def fake_fit_single(self, seed_generator, datasets, is_fine_tune=False):
            free = seed_generator.free_indices
            single_calls.append({
                'seed_len': len(seed_generator.initial_guess),
                'free_indices': list(free) if free is not None else None,
                'is_fine_tune': is_fine_tune,
                'use_bemt': manager.model.use_bemt,
            })
            return np.zeros(len(seed_generator.initial_guess))

        with patch.object(FittingEngine, 'fit', fake_fit), \
             patch.object(FittingEngine, 'fit_single', fake_fit_single), \
             patch.object(manager.model, 'configure_for_band_correction_fit') as mock_cfg:
            result = manager.run_staged()

        # Two single-start stages ran (1.5 then 2), band correction configured once between.
        self.assertEqual(len(single_calls), 2)
        mock_cfg.assert_called_once()

        # Stage 1.5: refine all six base coefficients (no restriction), coarse solver, BEMT on.
        self.assertIsNone(single_calls[0]['free_indices'])
        self.assertEqual(single_calls[0]['seed_len'], 6)
        self.assertFalse(single_calls[0]['is_fine_tune'])
        self.assertTrue(single_calls[0]['use_bemt'])

        # Stage 2: only the 14 offsets free, full 20-vector seed, fine solver, BEMT on.
        self.assertEqual(single_calls[1]['free_indices'], list(range(6, 20)))
        self.assertEqual(single_calls[1]['seed_len'], 20)
        self.assertTrue(single_calls[1]['is_fine_tune'])
        self.assertTrue(single_calls[1]['use_bemt'])

        self.assertEqual(len(result), 20)

    def test_run_staged_uses_bet_for_stage_one(self):
        manager = FittingManager.for_single_rotor_staged(self.blade, is_ccw_rotor0=False,
                                                         datasets=self.datasets)
        seen = {}

        def fake_fit(self, seed_generator, datasets):
            seen['use_bemt_stage1'] = manager.model.use_bemt
            return np.array([5.0, 2.0, 1.0, 0.4, 0.0, 0.0])

        def fake_fit_single(self, seed_generator, datasets, is_fine_tune=False):
            return np.zeros(len(seed_generator.initial_guess))

        with patch.object(FittingEngine, 'fit', fake_fit), \
             patch.object(FittingEngine, 'fit_single', fake_fit_single), \
             patch.object(manager.model, 'configure_for_band_correction_fit'):
            manager.run_staged()

        self.assertFalse(seen['use_bemt_stage1'])


if __name__ == "__main__":
    unittest.main()
