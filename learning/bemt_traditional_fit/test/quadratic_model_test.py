"""Unit tests for the quadratic fit pipeline: model, solver, engine, lookup table, plotter."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")  # headless: no display needed for plotter tests

import numpy as np
import pandas as pd

import data_factory
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.quadratic_solver import QuadraticSolver
from learning.bemt_traditional_fit.quadratic_fitting_engine import QuadraticFittingEngine
from learning.bemt_traditional_fit.quadratic_fit_plotter import QuadraticFitPlotter
from learning.bemt_traditional_fit.per_condition_result import (
    PerConditionResult, extract_wind_condition,
    NOMINAL_U_FREE_X, NOMINAL_PITCH,
)
from learning.bemt_traditional_fit.post_processing import make_lookup_table_from_quadratic_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quadratic_dataset(
        n: int,
        omega_values: np.ndarray,
        coeffs_fz=(2e-5, 0.0, 0.0),
        coeffs_fx=(0.0, 0.0, 0.0),
        coeffs_fy=(0.0, 0.0, 0.0),
        wind: np.ndarray = None,
) -> data_factory.FittingDataset:
    """Dataset where rotor_0 forces follow the given quadratic coefficients."""
    if wind is None:
        wind = np.zeros(3)
    omegas = np.tile(omega_values, (n // len(omega_values) + 1))[:n]

    a_fz, b_fz, c_fz = coeffs_fz
    a_fx, b_fx, c_fx = coeffs_fx
    a_fy, b_fy, c_fy = coeffs_fy

    forces = [
        [a_fx * w ** 2 + b_fx * w + c_fx,
         a_fy * w ** 2 + b_fy * w + c_fy,
         a_fz * w ** 2 + b_fz * w + c_fz]
        for w in omegas
    ]

    df_data = {}
    for rotor in range(4):
        df_data[f"rotor_{rotor}_local_wind_velocity"] = [wind.tolist()] * n
        df_data[f"rotor_{rotor}_velocity"] = [np.zeros(3).tolist()] * n
        df_data[f"rotor_{rotor}_rotation_spd"] = omegas.tolist()
        df_data[f"rotor_{rotor}_f_rotor_inertial_frame"] = (
            forces if rotor == 0 else [np.zeros(3).tolist()] * n
        )
    df_data["shared_r_disk"] = [np.eye(3).tolist()] * n
    df_data["sensed_dv"] = [np.zeros(3).tolist()] * n
    df_data["sensed_omega"] = [np.zeros(3).tolist()] * n
    df_data["v"] = [np.zeros(3).tolist()] * n
    return data_factory.FittingDataset(pd.DataFrame(df_data), "mock")


# ---------------------------------------------------------------------------
# QuadraticRotorModel tests
# ---------------------------------------------------------------------------

class TestQuadraticRotorModel(unittest.TestCase):

    def setUp(self):
        self.model = QuadraticRotorModel()

    def test_parameter_names_count(self):
        self.assertEqual(len(QuadraticRotorModel.PARAMETER_NAMES), 9)

    def test_predict_zero_params(self):
        self.model.apply_params(np.zeros(9))
        np.testing.assert_array_equal(self.model.predict_forces_disk(500.0), np.zeros(3))

    def test_predict_fz_quadratic(self):
        a_fz, b_fz, c_fz = 2e-5, 0.0, 0.0
        x = np.array([0, 0, 0,  0, 0, 0,  a_fz, b_fz, c_fz])
        self.model.apply_params(x)
        omega = 600.0
        self.assertAlmostEqual(self.model.predict_forces_disk(omega)[2],
                               a_fz * omega ** 2, places=10)

    def test_predict_fx_linear(self):
        b_fx = 0.02
        x = np.array([0, b_fx, 0,  0, 0, 0,  0, 0, 0])
        self.model.apply_params(x)
        omega = 400.0
        self.assertAlmostEqual(self.model.predict_forces_disk(omega)[0],
                               b_fx * omega, places=10)

    def test_get_residual_force_shape(self):
        dataset = _make_quadratic_dataset(10, np.linspace(100, 1000, 10))
        self.model.apply_params(np.zeros(9))
        self.assertEqual(self.model.get_residual_force(dataset, 0).shape, (3,))

    def test_get_residual_zero_when_params_match(self):
        a_fz = 2e-5
        omega_values = np.linspace(100, 1000, 20)
        dataset = _make_quadratic_dataset(200, omega_values, coeffs_fz=(a_fz, 0.0, 0.0))
        x = np.array([0, 0, 0,  0, 0, 0,  a_fz, 0.0, 0.0])
        self.model.apply_params(x)
        for i in range(0, 200, 20):
            np.testing.assert_array_almost_equal(
                self.model.get_residual_force(dataset, i), np.zeros(3), decimal=10
            )


# ---------------------------------------------------------------------------
# QuadraticSolver tests
# ---------------------------------------------------------------------------

class TestQuadraticSolver(unittest.TestCase):

    def setUp(self):
        self.model = QuadraticRotorModel()
        self.solver = QuadraticSolver()

    def _solve(self, **kwargs):
        omega_values = np.linspace(100, 1000, 30)
        dataset = _make_quadratic_dataset(300, omega_values, **kwargs)
        return self.solver.run(self.model, dataset)

    def test_returns_9_params(self):
        params = self._solve()
        self.assertEqual(params.shape, (9,))

    def test_fz_coefficients_recovered(self):
        a, b, c = 3e-5, 0.005, -0.2
        params = self._solve(coeffs_fz=(a, b, c))
        self.assertAlmostEqual(params[6], a, places=10)
        self.assertAlmostEqual(params[7], b, places=10)
        self.assertAlmostEqual(params[8], c, places=10)

    def test_fx_coefficients_recovered(self):
        a, b, c = -1e-5, 0.003, 0.1
        params = self._solve(coeffs_fx=(a, b, c))
        self.assertAlmostEqual(params[0], a, places=10)
        self.assertAlmostEqual(params[1], b, places=10)
        self.assertAlmostEqual(params[2], c, places=10)

    def test_fy_zero_when_no_lateral_force(self):
        params = self._solve(coeffs_fz=(2e-5, 0.0, 0.0))
        np.testing.assert_array_almost_equal(params[3:6], 0.0, decimal=10)

    def test_model_params_updated_after_solve(self):
        a_fz = 2e-5
        omega_values = np.linspace(100, 1000, 30)
        dataset = _make_quadratic_dataset(300, omega_values, coeffs_fz=(a_fz, 0.0, 0.0))
        self.solver.run(self.model, dataset)
        omega_test = 500.0
        self.assertAlmostEqual(self.model.predict_forces_disk(omega_test)[2],
                               a_fz * omega_test ** 2, places=8)

    def test_exact_fit_zero_residual(self):
        """Solved params should give zero residual on noiseless data."""
        a_fz = 2e-5
        omega_values = np.linspace(100, 1000, 30)
        dataset = _make_quadratic_dataset(300, omega_values, coeffs_fz=(a_fz, 0.0, 0.0))
        self.solver.run(self.model, dataset)
        for i in range(len(dataset)):
            residual = self.model.get_residual_force(dataset, i)
            np.testing.assert_array_almost_equal(residual, np.zeros(3), decimal=8)


# ---------------------------------------------------------------------------
# _extract_wind_condition tests
# ---------------------------------------------------------------------------

class TestExtractWindCondition(unittest.TestCase):

    def test_zero_wind(self):
        dataset = _make_quadratic_dataset(20, np.linspace(100, 1000, 10))
        u_free_x, pitch = extract_wind_condition(dataset)
        self.assertAlmostEqual(u_free_x, 0.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)

    def test_horizontal_wind(self):
        dataset = _make_quadratic_dataset(20, np.linspace(100, 1000, 10),
                                          wind=np.array([3.0, 0.0, 0.0]))
        u_free_x, pitch = extract_wind_condition(dataset)
        self.assertAlmostEqual(u_free_x, 3.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)

    def test_vertical_wind(self):
        dataset = _make_quadratic_dataset(20, np.linspace(100, 1000, 10),
                                          wind=np.array([0.0, 0.0, 2.0]))
        u_free_x, pitch = extract_wind_condition(dataset)
        self.assertAlmostEqual(u_free_x, 2.0, places=4)
        self.assertAlmostEqual(pitch, np.pi / 2, places=4)

    def test_pitch_ignores_disk_tilt(self):
        """Pitch is taken in the inertial frame, so a tilted disk must not change it."""
        dataset = _make_quadratic_dataset(20, np.linspace(100, 1000, 10),
                                          wind=np.array([-3.0, 0.0, 0.0]))
        # Tilt every disk by 40 deg about y; label must stay horizontal (pitch 0).
        theta = np.radians(40.0)
        r = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        dataset.shared_r_disk = np.array([r] * len(dataset))
        u_free_x, pitch = extract_wind_condition(dataset)
        self.assertAlmostEqual(u_free_x, 3.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)

    def test_snap_to_nominal_grid(self):
        """A slightly-off condition snaps onto the nominal sweep grid."""
        pitch_deg, mag = 29.0, 2.98
        pr = np.radians(pitch_deg)
        wind = np.array([-mag * np.cos(pr), 0.0, mag * np.sin(pr)])
        dataset = _make_quadratic_dataset(20, np.linspace(100, 1000, 10), wind=wind)
        u_free_x, pitch = extract_wind_condition(dataset, NOMINAL_U_FREE_X, NOMINAL_PITCH)
        self.assertEqual(u_free_x, 3.0)
        self.assertAlmostEqual(pitch, np.radians(30.0), places=12)

    def test_snap_warns_when_far(self):
        """Snapping a condition far from any nominal value emits a warning."""
        dataset = _make_quadratic_dataset(20, np.linspace(100, 1000, 10),
                                          wind=np.array([-7.0, 0.0, 0.0]))
        with self.assertWarns(UserWarning):
            u_free_x, _ = extract_wind_condition(dataset, NOMINAL_U_FREE_X, NOMINAL_PITCH)
        self.assertEqual(u_free_x, 5.0)  # nearest of {3,5,10}


# ---------------------------------------------------------------------------
# make_lookup_table_from_quadratic_fit tests
# ---------------------------------------------------------------------------

class TestMakeLookupTableFromQuadraticFit(unittest.TestCase):

    def _make_results(self, datasets, params_list):
        """Wrap bare param arrays into a list of PerConditionResult using the dataset wind."""
        results = []
        for p, ds in zip(params_list, datasets):
            u_free_x, pitch = extract_wind_condition(ds)
            results.append(PerConditionResult(params=p, u_free_x=u_free_x, pitch=pitch))
        return results

    def _run_with_mock_save(self, datasets, params_list, omega_range):
        from inflow_model import propeller_lookup_table as plt_mod
        captured = {}

        def mock_save(filename, u_range, p_range, _omega_range, table):
            captured.update(u_range=u_range, p_range=p_range,
                            table=table, omega_range=_omega_range)

        results = self._make_results(datasets, params_list)
        with patch.object(plt_mod.PropellerLookupTable.Maker, 'save_data',
                          staticmethod(mock_save)):
            make_lookup_table_from_quadratic_fit(results, "test", omega_range=omega_range)
        return captured

    def test_table_shape(self):
        omega_vals = np.linspace(100, 1000, 20)
        ds0 = _make_quadratic_dataset(200, omega_vals)
        ds1 = _make_quadratic_dataset(200, omega_vals, wind=np.array([3.0, 0.0, 0.0]))
        params = [np.zeros(9), np.zeros(9)]
        omega_range = np.array([0.0, 500.0, 1000.0])
        c = self._run_with_mock_save([ds0, ds1], params, omega_range)
        self.assertEqual(c['table'].shape,
                         (len(c['u_range']), len(c['p_range']), len(c['omega_range']), 4))

    def test_thrust_matches_quadratic(self):
        a_fz = 2e-5
        omega_vals = np.linspace(100, 1000, 20)
        ds = _make_quadratic_dataset(200, omega_vals, coeffs_fz=(a_fz, 0.0, 0.0))
        params = [np.array([0, 0, 0,  0, 0, 0,  a_fz, 0.0, 0.0])]
        omega_range = np.array([0.0, 500.0])
        c = self._run_with_mock_save([ds], params, omega_range)
        self.assertAlmostEqual(c['table'][0, 0, 1, 2], a_fz * 500.0 ** 2, places=8)

    def test_vi_is_zero(self):
        omega_vals = np.linspace(100, 1000, 20)
        ds = _make_quadratic_dataset(200, omega_vals)
        c = self._run_with_mock_save([ds], [np.zeros(9)], np.array([0.0, 500.0]))
        np.testing.assert_array_equal(c['table'][:, :, :, 3], 0.0)

    def test_fy_negated_for_cw_rotor(self):
        a_fy = 1e-5
        omega_vals = np.linspace(100, 1000, 20)
        ds = _make_quadratic_dataset(200, omega_vals, coeffs_fy=(a_fy, 0.0, 0.0))
        p_arr = np.array([0, 0, 0,  a_fy, 0, 0,  0, 0, 0])
        omega_range = np.array([0.0, 500.0])

        from inflow_model import propeller_lookup_table as plt_mod
        saved = {}

        def mock_ccw(f, u, p, o, table): saved['ccw'] = table.copy()
        def mock_cw(f, u, p, o, table):  saved['cw']  = table.copy()

        results = self._make_results([ds], [p_arr])
        with patch.object(plt_mod.PropellerLookupTable.Maker, 'save_data', staticmethod(mock_ccw)):
            make_lookup_table_from_quadratic_fit(results, "t", omega_range=omega_range,
                                                 is_ccw_rotor0=True)
        with patch.object(plt_mod.PropellerLookupTable.Maker, 'save_data', staticmethod(mock_cw)):
            make_lookup_table_from_quadratic_fit(results, "t", omega_range=omega_range,
                                                 is_ccw_rotor0=False)

        self.assertAlmostEqual(saved['ccw'][0, 0, 1, 1],
                               -saved['cw'][0, 0, 1, 1], places=8)


# ---------------------------------------------------------------------------
# QuadraticFitPlotter tests — one figure per condition, fit vs measured
# ---------------------------------------------------------------------------

class TestQuadraticFitPlotter(unittest.TestCase):

    def _fitted_results(self, specs):
        """Build real fitted results (with datasets attached) via the engine.

        specs: list of (a_fz, wind) tuples.
        """
        omega_vals = np.linspace(100, 1000, 20)
        datasets = [_make_quadratic_dataset(200, omega_vals, coeffs_fz=(a_fz, 0.0, 0.0), wind=w)
                    for a_fz, w in specs]
        return QuadraticFittingEngine(QuadraticRotorModel()).fit(datasets)

    def test_one_figure_per_condition(self):
        results = self._fitted_results([(2e-5, None),
                                        (1.5e-5, np.array([3.0, 0.0, 0.0]))])
        figs = QuadraticFitPlotter.plot_fit_comparison(results)
        self.assertEqual(len(figs), 2)

    def test_empty_results_no_figures(self):
        figs = QuadraticFitPlotter.plot_fit_comparison([])
        self.assertEqual(figs, [])

    def test_each_figure_has_fit_and_measured_line(self):
        results = self._fitted_results([(2e-5, None)])
        fig = QuadraticFitPlotter.plot_fit_comparison(results)[0]
        ax = fig.axes[0]
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)  # one measured (markers) + one fit curve
        measured = [ln for ln in lines if ln.get_linestyle() == "None"]
        self.assertEqual(len(measured), 1)

    def test_curve_matches_measured_on_noiseless_data(self):
        a_fz = 2e-5
        fig = QuadraticFitPlotter.plot_fit_comparison(self._fitted_results([(a_fz, None)]))[0]
        measured = next(ln for ln in fig.axes[0].get_lines() if ln.get_linestyle() == "None")
        x, y = measured.get_xdata(), measured.get_ydata()
        np.testing.assert_allclose(y, a_fz * x ** 2, atol=1e-6)

    def test_raises_without_dataset(self):
        result = PerConditionResult(params=np.zeros(9), u_free_x=0.0, pitch=0.0)  # dataset=None
        with self.assertRaises(ValueError):
            QuadraticFitPlotter.plot_fit_comparison([result])


# ---------------------------------------------------------------------------
# QuadraticFittingEngine tests — the engine owns the per-condition loop
# ---------------------------------------------------------------------------

class TestQuadraticFittingEngine(unittest.TestCase):

    def test_fit_returns_results_for_all_conditions(self):
        omega_vals = np.linspace(100, 1000, 20)
        ds0 = _make_quadratic_dataset(200, omega_vals, coeffs_fz=(2e-5, 0.0, 0.0))
        ds1 = _make_quadratic_dataset(200, omega_vals, coeffs_fz=(1.5e-5, 0.0, 0.0),
                                      wind=np.array([3.0, 0.0, 0.0]))
        engine = QuadraticFittingEngine(QuadraticRotorModel())
        results = engine.fit([ds0, ds1])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].params[6], 2e-5, places=10)
        self.assertAlmostEqual(results[1].params[6], 1.5e-5, places=10)
        self.assertAlmostEqual(results[1].u_free_x, 3.0, places=4)
        # Each result carries the dataset it was fitted on (for comparison plots).
        self.assertIs(results[0].dataset, ds0)
        self.assertIs(results[1].dataset, ds1)

    def test_fit_empty_datasets_gives_empty_results(self):
        engine = QuadraticFittingEngine(QuadraticRotorModel())
        results = engine.fit([])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)


# ---------------------------------------------------------------------------
# FittingManager integration (no real data needed)
# ---------------------------------------------------------------------------

class TestFittingManagerQuadratic(unittest.TestCase):

    def test_for_quadratic_rotor_stores_engine(self):
        from learning.bemt_traditional_fit.manager import FittingManager
        omega_vals = np.linspace(100, 1000, 20)
        ds = _make_quadratic_dataset(200, omega_vals, coeffs_fz=(2e-5, 0.0, 0.0))
        mgr = FittingManager.for_quadratic_rotor([ds])
        self.assertIsInstance(mgr.engine, QuadraticFittingEngine)
        self.assertFalse(hasattr(mgr, '_quadratic_solver'))

    def test_run_quadratic_fit_returns_list_results(self):
        from learning.bemt_traditional_fit.manager import FittingManager
        omega_vals = np.linspace(100, 1000, 20)
        ds0 = _make_quadratic_dataset(200, omega_vals, coeffs_fz=(2e-5, 0.0, 0.0))
        ds1 = _make_quadratic_dataset(200, omega_vals, coeffs_fz=(1.5e-5, 0.0, 0.0),
                                      wind=np.array([3.0, 0.0, 0.0]))
        mgr = FittingManager.for_quadratic_rotor([ds0, ds1])
        results = mgr.run_quadratic_fit()
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, PerConditionResult)
            self.assertEqual(r.params.shape, (9,))
        # Wind condition is attached to each result (second dataset has 3 m/s wind).
        self.assertAlmostEqual(results[0].u_free_x, 0.0, places=4)
        self.assertAlmostEqual(results[1].u_free_x, 3.0, places=4)

    def test_run_quadratic_fit_recovers_fz(self):
        from learning.bemt_traditional_fit.manager import FittingManager
        a_fz = 2.5e-5
        omega_vals = np.linspace(100, 1000, 30)
        ds = _make_quadratic_dataset(300, omega_vals, coeffs_fz=(a_fz, 0.0, 0.0))
        mgr = FittingManager.for_quadratic_rotor([ds])
        result = mgr.run_quadratic_fit()[0]
        self.assertAlmostEqual(result.params[6], a_fz, places=10)

    def test_run_quadratic_fit_raises_without_engine(self):
        from learning.bemt_traditional_fit.manager import FittingManager
        omega_vals = np.linspace(100, 1000, 20)
        ds = _make_quadratic_dataset(200, omega_vals)
        # Manually build a manager with engine=None to confirm the guard fires.
        mgr = FittingManager(QuadraticRotorModel(), engine=None, datasets=[ds])
        with self.assertRaises(RuntimeError):
            mgr.run_quadratic_fit()


if __name__ == "__main__":
    unittest.main()
