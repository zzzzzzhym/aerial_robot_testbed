import numpy as np

import data_factory
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.objective import FittingObjective
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective
from learning.bemt_traditional_fit.fit_plotter import FitPlotter
from learning.bemt_traditional_fit.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.seed_generator import SingleSeedGenerator


class FittingManager:
    """User-facing facade for BEMT parameter fitting.

    Construct via factory methods rather than directly:
        FittingManager.for_full_vehicle(blade, params, datasets)
        FittingManager.for_single_rotor(blade, is_ccw_rotor0, datasets)
    """

    def __init__(self, model, objective, datasets: list[data_factory.FittingDataset],
                 init_guess=None):
        self.model = model
        self.datasets = datasets
        self.init_guess = (
            np.asarray(init_guess, dtype=float) if init_guess is not None else None
        )
        self.engine = FittingEngine(model, objective)

    @classmethod
    def for_full_vehicle(cls, blade, params, datasets: list[data_factory.FittingDataset],
                         init_guess=None):
        """Full four-rotor vehicle fitting. Fits cl_1, cl_2, cd, alpha_0, k_body_drag."""
        model = BemtModel(blade, params)
        objective = FittingObjective(model)
        return cls(model, objective, datasets, init_guess)

    @classmethod
    def for_single_rotor(cls, blade, is_ccw_rotor0: bool,
                         datasets: list[data_factory.FittingDataset], init_guess=None):
        """Single-rotor test-stand fitting. Fits cl_1, cl_2, cd, alpha_0."""
        model = SingleRotorBemtModel(blade, is_ccw_rotor0)
        objective = SingleRotorObjective(model)
        return cls(model, objective, datasets, init_guess)

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

    def plot(self, dataset_idx: int = 0, lookup_table=None, is_using_lookup_table: bool = False,
             sample_step: int = 1):
        """Plot fit vs ground truth for one dataset (full-vehicle model only)."""
        return FitPlotter.plot_the_fit(
            self.model,
            self.datasets[dataset_idx],
            lookup_table=lookup_table,
            is_using_lookup_table=is_using_lookup_table,
            sample_step=sample_step,
        )
