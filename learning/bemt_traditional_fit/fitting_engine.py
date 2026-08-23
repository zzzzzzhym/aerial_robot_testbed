import numpy as np

import data_factory
from learning.bemt_traditional_fit.seed_generator import SeedGenerator, MultiSeedGenerator
from learning.bemt_traditional_fit.solver import Solver


class FittingEngine:
    """Orchestrates the multi-start fitting pipeline."""

    def __init__(self, model, objective):
        self.model = model
        self.objective = objective

    def _format_parameters(self, values) -> str:
        parts = []
        for name, value in zip(self.model.PARAMETER_NAMES, values):
            if name == "alpha_0":
                parts.append(f"{name}={np.degrees(value):.3f}deg")
            else:
                parts.append(f"{name}={value:.3f}")
        return "  ".join(parts)

    def _print_result(self, label: str, loss: float, x):
        print(f"{label}: loss={loss:.4f}  {self._format_parameters(x)}")

    def _screen_seeds(self, seeds, datasets, n_keep=3):
        """Evaluate each seed once and return the n_keep lowest-loss ones."""
        evaluated = []
        for seed in seeds:
            loss = self.objective.get_loss(seed, datasets)
            evaluated.append((loss, seed.copy()))
        evaluated.sort(key=lambda item: item[0])
        return evaluated[:n_keep]

    def fit(self, datasets: list[data_factory.FittingDataset], custom_init: np.ndarray = None) -> np.ndarray:
        """Three-stage multistart fitting: LHS screening → coarse → fine-tune."""
        bounds = self.model.BOUNDS

        # Stage 1: inexpensive global screening
        self.model.adjust_resolution(is_fine_tune=False)
        seed_gen = MultiSeedGenerator(n_lhs=128, physical_seed=custom_init, random_seed=42)
        seeds = seed_gen.get_seeds(bounds)
        selected = self._screen_seeds(seeds, datasets, n_keep=3)
        for rank, (loss, seed) in enumerate(selected, start=1):
            self._print_result(f"Selected seed {rank}", loss, seed)

        # Stage 2: coarse local optimization from the best seeds
        coarse_solver = Solver(bounds, maxiter=20, options={'ftol': 1e-4, 'xtol': 1e-3, 'disp': True})
        coarse_results = []
        for _, seed in selected:
            print(f"Evaluating: {self._format_parameters(seed)}")
            result = coarse_solver.run(lambda x: self.objective.get_loss(x, datasets), seed)
            coarse_results.append(result)
        coarse_results.sort(key=lambda r: r.fun)
        best_coarse = coarse_results[0]
        self._print_result("Best coarse result", best_coarse.fun, best_coarse.x_physical)

        # Stage 3: further iterate only the best coarse result
        self.model.adjust_resolution(is_fine_tune=False)
        fine_solver = Solver(bounds, maxiter=80, options={'ftol': 1e-4, 'xtol': 1e-3, 'disp': True})
        fine_result = fine_solver.run(lambda x: self.objective.get_loss(x, datasets), best_coarse.x_physical)
        self._print_result("Final result", fine_result.fun, fine_result.x_physical)
        return fine_result.x_physical

    def fit_single(self, datasets: list[data_factory.FittingDataset],
                   seed_generator: SeedGenerator, is_fine_tune=False):
        """Single-start fitting from a provided seed generator."""
        bounds = self.model.BOUNDS
        self.model.adjust_resolution(is_fine_tune)

        initial_guess = seed_generator.get_seeds(bounds)[0]
        print("Initial guess:", initial_guess)

        maxiter = 200 if is_fine_tune else 100
        solver = Solver(bounds, maxiter=maxiter, options={'disp': True, 'fatol': 1e-1})
        step_counter = {"count": 0}

        def callback(zk):
            x = solver.denormalize(zk)
            step_counter["count"] += 1
            print(f"Step {step_counter['count']:3d}: {self._format_parameters(x)}")

        result = solver.run(lambda x: self.objective.get_loss(x, datasets), initial_guess, callback=callback)

        fitted_params = result.x_physical
        if result.success:
            print("Fitted parameters: " + self._format_parameters(fitted_params))
            return fitted_params
        else:
            print("Optimization failed:", result.message)
            return None
