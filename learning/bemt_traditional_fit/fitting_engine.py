import numpy as np

import data_factory
from learning.bemt_traditional_fit.seed_generator import SeedGenerator, MultiSeedGenerator
from learning.bemt_traditional_fit.solver import Solver


class FittingEngine:
    """Orchestrates the multi-start fitting pipeline."""

    def __init__(self, model, objective, decision_var,
                 seed_gen: MultiSeedGenerator,
                 coarse_solver: Solver,
                 fine_solver: Solver,
                 single_solver: Solver,
                 single_fine_solver: Solver):
        self.model = model
        self.objective = objective
        self.decision_var = decision_var
        self.seed_gen = seed_gen
        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver
        self.single_solver = single_solver
        self.single_fine_solver = single_fine_solver

    def _loss(self, search_x, datasets):
        """Evaluate the objective on a search-space vector (converted to physical)."""
        return self.objective.get_loss(self.decision_var.to_physical(search_x), datasets)

    def _print_result(self, label: str, loss: float, x):
        print(f"{label}: loss={loss:.4f}  {self.decision_var.format(x)}")

    def _screen_seeds(self, seeds, datasets):
        """Evaluate each seed once and return the n_keep lowest-loss ones."""
        evaluated = []
        for seed in seeds:
            loss = self._loss(seed, datasets)
            evaluated.append((loss, seed.copy()))
        evaluated.sort(key=lambda item: item[0])
        return evaluated[:self.seed_gen.n_keep]

    def fit(self, datasets: list[data_factory.FittingDataset], custom_init: np.ndarray = None) -> np.ndarray:
        """Three-stage multistart fitting: LHS screening → coarse → fine-tune."""
        # Stage 1: inexpensive global screening
        self.model.adjust_resolution(is_fine_tune=False)
        # custom_init arrives in physical coefficients; seed the optimizer in search space.
        self.seed_gen.physical_seed = (
            self.decision_var.to_search(custom_init) if custom_init is not None else None
        )
        seeds = self.seed_gen.get_seeds(self.decision_var.bounds)
        selected = self._screen_seeds(seeds, datasets)
        for rank, (loss, seed) in enumerate(selected, start=1):
            self._print_result(f"Selected seed {rank}", loss, seed)

        # Stage 2: coarse local optimization from the best seeds
        coarse_results = []
        for candidate, (_, seed) in enumerate(selected, start=1):
            print(f"Evaluating: {self.decision_var.format(seed)}")
            result = self.coarse_solver.run(lambda x: self._loss(x, datasets), seed)
            coarse_results.append((candidate, result))
        coarse_results.sort(key=lambda item: item[1].fun)
        best_candidate = coarse_results[0][0]
        best_coarse = coarse_results[0][1]

        print("\nCoarse tune results:")
        print("candidate | final loss | (best)")
        for candidate, result in sorted(coarse_results, key=lambda item: item[0]):
            marker = " | (best)" if candidate == best_candidate else ""
            print(f"{candidate} | {result.fun:.4f}{marker}")

        self._print_result("Best coarse result", best_coarse.fun, best_coarse.x_physical)

        # Stage 3: further iterate only the best coarse result
        self.model.adjust_resolution(is_fine_tune=True)
        fine_result = self.fine_solver.run(lambda x: self._loss(x, datasets), best_coarse.x_physical)
        self._print_result("Final result", fine_result.fun, fine_result.x_physical)
        physical = self.decision_var.to_physical(fine_result.x_physical)
        self.model.apply_params(physical)
        return physical

    def fit_single(self, datasets: list[data_factory.FittingDataset],
                   seed_generator: SeedGenerator, is_fine_tune: bool = False):
        """Single-start fitting from a provided seed generator."""
        self.model.adjust_resolution(is_fine_tune)
        solver = self.single_fine_solver if is_fine_tune else self.single_solver

        # The provided seed is in physical coefficients; search in decision space.
        initial_guess = self.decision_var.to_search(seed_generator.get_seeds(self.decision_var.bounds)[0])
        print("Initial guess:", initial_guess)

        step_counter = {"count": 0}

        def callback(zk):
            x = solver.denormalize(zk)
            step_counter["count"] += 1
            print(f"Step {step_counter['count']:3d}: {self.decision_var.format(x)}")

        result = solver.run(lambda x: self._loss(x, datasets), initial_guess, callback=callback)

        fitted_search = result.x_physical
        if result.success:
            physical = self.decision_var.to_physical(fitted_search)
            print("Fitted parameters: " + self.decision_var.format(fitted_search))
            self.model.apply_params(physical)
            return physical
        else:
            print("Optimization failed:", result.message)
            return None
