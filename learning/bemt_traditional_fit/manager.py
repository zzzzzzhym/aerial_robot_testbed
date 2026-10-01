from pathlib import Path
import numpy as np

import data_factory
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.fitting_config import FittingConfig
from learning.bemt_traditional_fit.objective import FittingObjective
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective
from learning.bemt_traditional_fit.body_drag_objective import BodyDragObjective
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.quadratic_fitting_engine import QuadraticFittingEngine
from learning.bemt_traditional_fit.per_condition_result import PerConditionResult
from learning.bemt_traditional_fit.fit_plotter import FitPlotter
from learning.bemt_traditional_fit.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.seed_generator import MultiSeedGenerator, SingleSeedGenerator
from learning.bemt_traditional_fit.solver import Solver

_CONFIG_DIR = Path(__file__).parent


def _build_engine(model, objective, config: FittingConfig) -> FittingEngine:
    bounds = model.BOUNDS
    sc = config.seed
    sv = config.solver
    seed_gen = MultiSeedGenerator(n_lhs=sc.n_lhs, n_keep=sc.n_keep, random_seed=sc.random_seed)
    coarse_solver = Solver(bounds, maxiter=sv.coarse_maxiter, options={'ftol': sv.ftol, 'xtol': sv.xtol, 'disp': True})
    fine_solver = Solver(bounds, maxiter=sv.fine_maxiter, options={'ftol': sv.ftol, 'xtol': sv.xtol, 'disp': True})
    single_solver = Solver(bounds, maxiter=sv.single_maxiter, options={'disp': True, 'fatol': 1e-1})
    single_fine_solver = Solver(bounds, maxiter=sv.single_fine_maxiter, options={'disp': True, 'fatol': 1e-1})
    return FittingEngine(model, objective, seed_gen, coarse_solver, fine_solver, single_solver, single_fine_solver)


class FittingManager:
    """User-facing facade for BEMT parameter fitting.

    Construct via factory methods rather than directly:
        FittingManager.for_full_vehicle(blade, params, datasets)
        FittingManager.for_single_rotor(blade, is_ccw_rotor0, datasets)
    """

    def __init__(self, model, engine, datasets: list[data_factory.FittingDataset],
                 init_guess=None):
        self.model = model
        self.engine = engine
        self.datasets = datasets
        self.init_guess = (
            np.asarray(init_guess, dtype=float) if init_guess is not None else None
        )

    @classmethod
    def for_full_vehicle(cls, blade, params, datasets: list[data_factory.FittingDataset],
                         init_guess=None, config: FittingConfig = None):
        """Full four-rotor vehicle fitting. Fits cl_1, cl_2, cd, alpha_0, k_body_drag."""
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_full_vehicle.yaml")
        model = BemtModel(blade, params, model_config=config.model)
        objective = FittingObjective(model)
        engine = _build_engine(model, objective, config)
        return cls(model, engine, datasets, init_guess)

    @classmethod
    def for_body_drag(cls, blade, params, datasets: list[data_factory.FittingDataset],
                      lookup_table, fixed_aero_params,
                      init_guess=None, config: FittingConfig = None):
        """Body-drag-only fitting given a fixed propeller map.

        Aero coefficients (cl_1, cl_2, cd, alpha_0) are frozen via fixed_aero_params;
        only k_body_drag is optimized. Thrust is computed via lookup_table.

        Args:
            fixed_aero_params: sequence (cl_1, cl_2, cd, alpha_0) from a prior single-rotor fit.
            lookup_table: prepared PropellerLookupTable.Reader for thrust computation.
            init_guess: optional initial k_body_drag scalar or 1-element array.
        """
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_body_drag.yaml")
        model = BemtModel(blade, params, model_config=config.model)
        cl_1, cl_2, cd, alpha_0 = fixed_aero_params
        model.blade.cl_1 = cl_1
        model.blade.cl_2 = cl_2
        model.blade.cd = cd
        model.blade.alpha_0 = alpha_0
        model.bet_instance.refresh_blade()
        model.configure_for_body_drag_fit()
        objective = BodyDragObjective(model, lookup_table)
        engine = _build_engine(model, objective, config)
        return cls(model, engine, datasets, init_guess)

    @classmethod
    def for_single_rotor(cls, blade, is_ccw_rotor0: bool,
                         datasets: list[data_factory.FittingDataset], init_guess=None,
                         config: FittingConfig = None):
        """Single-rotor test-stand fitting. Fits cl_1, cl_2, cd, alpha_0."""
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_single_rotor.yaml")
        model = SingleRotorBemtModel(blade, is_ccw_rotor0, model_config=config.model)
        objective = SingleRotorObjective(model)
        engine = _build_engine(model, objective, config)
        return cls(model, engine, datasets, init_guess)

    @classmethod
    def for_quadratic_rotor(cls, datasets: list[data_factory.FittingDataset]):
        """Quadratic per-condition fitting via closed-form least squares.

        Fits [a_fx, b_fx, c_fx, a_fy, b_fy, c_fy, a_fz, b_fz, c_fz] for each
        dataset independently using QuadraticSolver (lstsq — no iterative optimizer).
        Call run_quadratic_fit() to run.
        """
        model = QuadraticRotorModel()
        engine = QuadraticFittingEngine(model)
        return cls(model, engine, datasets)

    @classmethod
    def for_single_rotor_bemt(cls, blade, is_ccw_rotor0: bool,
                              datasets: list[data_factory.FittingDataset], init_guess=None,
                              config: FittingConfig = None):
        """BEMT single-rotor fitting with root-found v_i from background wind.

        Intended as a second-stage refinement after for_single_rotor.  Uses
        momentum-theory root-finding (get_rotor_forces) per element instead of
        integrating from sensed wind, so the aero coefficients are optimized
        against the self-consistent BEMT solution.
        """
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_single_rotor.yaml")
        model = SingleRotorBemtModel(blade, is_ccw_rotor0, model_config=config.model)
        model.use_bemt = True
        objective = SingleRotorObjective(model)
        engine = _build_engine(model, objective, config)
        return cls(model, engine, datasets, init_guess)

    def run(self, is_multiseed: bool = True, is_fine_tune: bool = False):
        """Run fitting.

        Args:
            is_multiseed: True → multi-start LHS pipeline;
                          False → single-start; init_guess must be set.
            is_fine_tune: only relevant when is_multiseed=False; selects BET resolution.
        """
        if is_multiseed:
            return self.engine.fit(self.datasets, custom_init=self.init_guess)

        if self.init_guess is None:
            raise ValueError("init_guess is required for single-seed fitting")
        seed_gen = SingleSeedGenerator(self.init_guess.tolist())
        return self.engine.fit_single(self.datasets, seed_generator=seed_gen, is_fine_tune=is_fine_tune)

    def run_quadratic_fit(self) -> list[PerConditionResult]:
        """Fit one quadratic profile per wind condition by handing the datasets to the engine.

        The manager doesn't know or care how many conditions there are — the engine
        owns the loop and returns a list of PerConditionResult.
        """
        if self.engine is None:
            raise RuntimeError("run_quadratic_fit() requires an engine; use a factory method to build the manager.")
        return self.engine.fit(self.datasets)

    def plot(self, dataset_idx: int = 0, lookup_table=None, is_using_lookup_table: bool = False,
             sample_step: int = 1, params=None):
        """Plot fit vs ground truth for one dataset (full-vehicle model only).

        Args:
            params: optional parameter vector to apply before plotting; if None, uses
                    whatever is currently set on the model (e.g. from a prior run()).
        """
        if params is not None:
            self.model.apply_params(params)
        return FitPlotter.plot_the_fit(
            self.model,
            self.datasets[dataset_idx],
            lookup_table=lookup_table,
            is_using_lookup_table=is_using_lookup_table,
            sample_step=sample_step,
        )
