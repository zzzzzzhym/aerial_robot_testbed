import numpy as np

import data_factory
from learning.bemt_traditional_fit.seed_generator import SeedGenerator, MultiSeedGenerator
from learning.bemt_traditional_fit.solver import Solver


class FittingEngine:
    """Orchestrates the fitting pipeline.

    Holds the model, objective, decision space and the config-built solvers. The seed
    strategy is NOT stored: both fit() and fit_single() receive a SeedGenerator per call,
    so the engine stays agnostic to where the starting points come from (global LHS screen
    vs. a single warm start, full search vs. a restricted subset of the decision vars).
    """

    def __init__(self, model, objective, decision_var,
                 coarse_solver: Solver,
                 fine_solver: Solver,
                 single_solver: Solver,
                 single_fine_solver: Solver):
        self.model = model
        self.objective = objective
        self.decision_var = decision_var
        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver
        self.single_solver = single_solver
        self.single_fine_solver = single_fine_solver

    def _loss(self, search_x, datasets):
        """Evaluate the objective on a search-space vector (converted to physical)."""
        return self.objective.get_loss(self.decision_var.to_physical(search_x), datasets)

    def _print_result(self, label: str, loss: float, x):
        print(f"{label}: loss={loss:.4f}  {self.decision_var.format(x)}")

    def _screen_seeds(self, seeds, datasets, n_keep):
        """Evaluate each seed once and return the n_keep lowest-loss ones."""
        evaluated = []
        for seed in seeds:
            loss = self._loss(seed, datasets)
            evaluated.append((loss, seed.copy()))
        evaluated.sort(key=lambda item: item[0])
        return evaluated[:n_keep]

    def fit(self, seed_generator: MultiSeedGenerator,
            datasets: list[data_factory.FittingDataset]) -> np.ndarray:
        """Three-stage multistart fitting: LHS screening → coarse → fine-tune."""
        # Stage 1: inexpensive global screening
        self.model.adjust_resolution(is_fine_tune=False)
        seeds = seed_generator.get_seeds(self.decision_var.bounds)
        selected = self._screen_seeds(seeds, datasets, seed_generator.n_keep)
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

    def fit_single(self, seed_generator: SeedGenerator,
                   datasets: list[data_factory.FittingDataset],
                   is_fine_tune: bool = False) -> np.ndarray:
        """Single-start fitting from a provided seed generator.

        The seed generator supplies the (physical) warm start and, optionally, the subset
        of the decision vector to vary via its free_indices (None → vary everything). When
        a subset is given the remaining components are held fixed at the seed — this is how
        a stage "limits the decision vars" (e.g. freeze base aero coefficients and vary only
        the CL/CD correction offsets).

        Solver (iteration budget, tolerances) and BET resolution come from config via
        is_fine_tune: the single-fine solver and fine resolution when True, else the coarse
        single solver and coarse resolution.
        """
        self.model.adjust_resolution(is_fine_tune)
        template = self.single_fine_solver if is_fine_tune else self.single_solver

        seed_search = np.asarray(
            self.decision_var.to_search(seed_generator.get_seeds(self.decision_var.bounds)[0]),
            dtype=float,
        )
        free_indices = getattr(seed_generator, "free_indices", None)
        if free_indices is None:
            free_indices = range(len(seed_search))
        free_indices = list(free_indices)

        def embed(free_x):
            full = seed_search.copy()
            full[free_indices] = free_x
            return full

        # Reduced-dimension solver over just the free components; dimension depends on the
        # free set, so it can't be one of the pre-built full-dim solvers. Reuse the template
        # solver's method/options (the config-derived budget and tolerances).
        free_bounds = [self.decision_var.bounds[i] for i in free_indices]
        solver = Solver(free_bounds, method=template.method, options=dict(template.options))

        if len(free_indices) < len(seed_search):
            print("Single-start fit — free: "
                  + ", ".join(self.decision_var.search_names[i] for i in free_indices))
        print("Initial guess: " + self.decision_var.format(seed_search))

        step_counter = {"count": 0}

        def callback(zk):
            free_x = solver.denormalize(zk)
            step_counter["count"] += 1
            print(f"Step {step_counter['count']:3d}: {self.decision_var.format(embed(free_x))}")

        result = solver.run(lambda free_x: self._loss(embed(free_x), datasets),
                            seed_search[free_indices], callback=callback)

        full_search = embed(result.x_physical)
        physical = self.decision_var.to_physical(full_search)
        if not result.success:
            # Nelder-Mead commonly reports success=False when it hits maxiter; the best
            # point found is still useful (and a staged pipeline must carry it forward).
            print("Optimization did not fully converge:", result.message)
        self._print_result("Fitted parameters", result.fun, full_search)
        self.model.apply_params(physical)
        return physical
