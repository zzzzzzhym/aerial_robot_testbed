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
from learning.bemt_traditional_fit import decision_var as dv

_CONFIG_DIR = Path(__file__).parent


def _build_seed_gen(config: FittingConfig) -> MultiSeedGenerator:
    sc = config.seed
    return MultiSeedGenerator(n_lhs=sc.n_lhs, n_keep=sc.n_keep, random_seed=sc.random_seed)


def _build_engine(model, objective, decision_var, config: FittingConfig) -> FittingEngine:
    bounds = decision_var.bounds
    sv = config.solver
    coarse_solver = Solver(bounds, maxiter=sv.coarse_maxiter, options={'ftol': sv.ftol, 'xtol': sv.xtol, 'disp': True})
    fine_solver = Solver(bounds, maxiter=sv.fine_maxiter, options={'ftol': sv.ftol, 'xtol': sv.xtol, 'disp': True})
    single_solver = Solver(bounds, maxiter=sv.single_maxiter, options={'disp': True, 'fatol': 1e-1})
    single_fine_solver = Solver(bounds, maxiter=sv.single_fine_maxiter, options={'disp': True, 'fatol': 1e-1})
    return FittingEngine(model, objective, decision_var, coarse_solver, fine_solver, single_solver, single_fine_solver)


class FittingManager:
    """User-facing facade for BEMT parameter fitting.

    Construct via factory methods rather than directly:
        FittingManager.for_full_vehicle(blade, params, datasets)
        FittingManager.for_single_rotor(blade, is_ccw_rotor0, datasets)
    """

    def __init__(self, model, engine, datasets: list[data_factory.FittingDataset],
                 init_guess=None, seed_gen: MultiSeedGenerator = None):
        self.model = model
        self.engine = engine
        self.datasets = datasets
        self.seed_gen = seed_gen
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
        engine = _build_engine(model, objective, dv.full_vehicle(), config)
        return cls(model, engine, datasets, init_guess, seed_gen=_build_seed_gen(config))

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
        model.blade.aero_coeff.update_fit_params(cl_1, cl_2, cd, alpha_0)
        model.configure_for_body_drag_fit()
        objective = BodyDragObjective(model, lookup_table)
        engine = _build_engine(model, objective, dv.body_drag(), config)
        return cls(model, engine, datasets, init_guess, seed_gen=_build_seed_gen(config))

    @classmethod
    def for_single_rotor(cls, blade, is_ccw_rotor0: bool,
                         datasets: list[data_factory.FittingDataset], init_guess=None,
                         config: FittingConfig = None):
        """Single-rotor test-stand fitting. Fits cl_1, cl_2, cd, alpha_0."""
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_single_rotor.yaml")
        model = SingleRotorBemtModel(blade, is_ccw_rotor0, model_config=config.model)
        objective = SingleRotorObjective(model)
        engine = _build_engine(model, objective, dv.OffsetAugmentedAeroCoeffDecisionVar(), config)
        return cls(model, engine, datasets, init_guess, seed_gen=_build_seed_gen(config))

    @classmethod
    def for_single_rotor_band_correction(cls, blade, is_ccw_rotor0: bool,
                                         datasets: list[data_factory.FittingDataset], init_guess=None,
                                         config: FittingConfig = None):
        """Single-rotor fit with a bounded piecewise-linear CL/CD correction band.

        Fits the offset-augmented aero coefficients plus per-knot CL/CD correction offsets
        (see BandCorrectedDecisionVar). init_guess, if given, is a physical vector:
        (cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min) followed by the CL then CD
        offsets; zero offsets reproduce the classical fit.
        """
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_single_rotor.yaml")
        model = SingleRotorBemtModel(blade, is_ccw_rotor0, model_config=config.model)
        decision_var = dv.BandCorrectedDecisionVar()
        model.configure_for_band_correction_fit(decision_var.cl_knots, decision_var.cd_knots)
        objective = SingleRotorObjective(model)
        engine = _build_engine(model, objective, decision_var, config)
        return cls(model, engine, datasets, init_guess, seed_gen=_build_seed_gen(config))

    @classmethod
    def for_quadratic_rotor(cls, datasets: list[data_factory.FittingDataset],
                            nominal_u_free_x=None, nominal_pitch=None):
        """Quadratic per-condition fitting via closed-form least squares.

        Fits [a_fx, b_fx, c_fx, a_fy, b_fy, c_fy, a_fz, b_fz, c_fz] for each
        dataset independently using QuadraticSolver (lstsq — no iterative optimizer).
        Call run_quadratic_fit() to run.

        nominal_u_free_x / nominal_pitch: optional sweep grids; when given, each
        condition's (u_free_x, pitch) label is snapped onto them so the resulting
        lookup table lands on a clean regular grid.
        """
        model = QuadraticRotorModel()
        engine = QuadraticFittingEngine(model, nominal_u_free_x, nominal_pitch)
        return cls(model, engine, datasets)

    @classmethod
    def for_single_rotor_staged(cls, blade, is_ccw_rotor0: bool,
                                datasets: list[data_factory.FittingDataset], init_guess=None,
                                config: FittingConfig = None):
        """Staged single-rotor fit that limits the decision variables per stage (goal 0).

        One SingleRotorBemtModel is reconfigured across three stages so the number of
        free parameters stays small in each:

          Stage 1   — BET with measured (sensed-wind) v_i; global multi-start over the
                      six base aero coefficients; CL/CD correction offsets held at 0.
          Stage 1.5 — BEMT with root-found v_i; refine the same six base coefficients
                      with the coarse single solver (single_* in config).
          Stage 2   — BEMT with root-found v_i; freeze the base coefficients and vary
                      only the CL/CD correction offsets with the fine single solver.

        All iteration budgets / resolutions come from the config; stage 1.5 uses the
        single_* solver (its maxiter is the "fine-tune N steps" budget) and stage 2 the
        single_fine_* solver.  Run with run_staged(); returns the band-correction vector.
        """
        config = config or FittingConfig.from_yaml(_CONFIG_DIR / "config_single_rotor.yaml")
        model = SingleRotorBemtModel(blade, is_ccw_rotor0, model_config=config.model)
        manager = cls(model, engine=None, datasets=datasets, init_guess=init_guess)
        manager._config = config
        return manager

    def run_staged(self):
        """Execute the three-stage single-rotor fit (see for_single_rotor_staged)."""
        if getattr(self, "_config", None) is None:
            raise RuntimeError("run_staged() requires a manager built via for_single_rotor_staged.")
        model = self.model
        datasets = self.datasets
        config = self._config
        base_dv = dv.OffsetAugmentedAeroCoeffDecisionVar()
        n_base = len(base_dv.search_names)

        # Stage 1: BET (measured v_i) — global multi-start over the six base coefficients.
        print("\n===== Stage 1: BET global fit (base aero coefficients) =====")
        model.use_bemt = False
        base_engine = _build_engine(model, SingleRotorObjective(model), base_dv, config)
        seed_gen = _build_seed_gen(config)
        seed_gen.physical_seed = (
            base_dv.to_search(self.init_guess) if self.init_guess is not None else None
        )
        physical_base = base_engine.fit(seed_gen, datasets)

        # Stage 1.5: BEMT (root-found v_i) — refine the same six coefficients.
        print("\n===== Stage 1.5: BEMT refine (base coefficients) =====")
        model.use_bemt = True
        physical_base = base_engine.fit_single(
            SingleSeedGenerator(physical_base), datasets, is_fine_tune=False,
        )

        # Stage 2: BEMT (root-found v_i) — freeze base coefficients, fit only the offsets.
        print("\n===== Stage 2: BEMT fine-tune (CL/CD correction offsets only) =====")
        band_dv = dv.BandCorrectedDecisionVar()
        model.configure_for_band_correction_fit(band_dv.cl_knots, band_dv.cd_knots)
        band_engine = _build_engine(model, SingleRotorObjective(model), band_dv, config)
        n_offsets = len(band_dv.search_names) - n_base
        seed_physical = np.concatenate([physical_base, np.zeros(n_offsets)])
        offset_indices = range(n_base, len(band_dv.search_names))
        physical = band_engine.fit_single(
            SingleSeedGenerator(seed_physical, free_indices=offset_indices),
            datasets, is_fine_tune=True,
        )
        self.engine = band_engine
        return physical

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
        engine = _build_engine(model, objective, dv.OffsetAugmentedAeroCoeffDecisionVar(), config)
        return cls(model, engine, datasets, init_guess, seed_gen=_build_seed_gen(config))

    def run(self, is_multiseed: bool = True, is_fine_tune: bool = False):
        """Run fitting.

        Args:
            is_multiseed: True → multi-start LHS pipeline;
                          False → single-start; init_guess must be set.
            is_fine_tune: only relevant when is_multiseed=False; selects BET resolution.
        """
        if is_multiseed:
            if self.seed_gen is None:
                raise RuntimeError("multiseed run requires a seed generator; use a factory method.")
            self.seed_gen.physical_seed = (
                self.engine.decision_var.to_search(self.init_guess)
                if self.init_guess is not None else None
            )
            return self.engine.fit(self.seed_gen, self.datasets)

        if self.init_guess is None:
            raise ValueError("init_guess is required for single-seed fitting")
        seed_gen = SingleSeedGenerator(self.init_guess.tolist())
        return self.engine.fit_single(seed_gen, self.datasets, is_fine_tune=is_fine_tune)

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
