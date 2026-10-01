"""Engine for QuadraticRotorModel — fits one quadratic profile per wind condition."""

import data_factory
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.quadratic_solver import QuadraticSolver
from learning.bemt_traditional_fit.per_condition_result import (
    PerConditionResult, extract_wind_condition,
)


class QuadraticFittingEngine:
    """Fits an independent quadratic profile for every wind condition (dataset).

    Owns the per-condition loop so the manager never has to know how many
    conditions there are — it just calls fit(datasets) and gets a list back.
    """

    def __init__(self, model: QuadraticRotorModel):
        self.model = model
        self._solver = QuadraticSolver()

    def fit(self, datasets: list[data_factory.FittingDataset]) -> list[PerConditionResult]:
        """Fit one quadratic per dataset; return a list of PerConditionResult."""
        results = []
        for idx, dataset in enumerate(datasets):
            print(f"\n=== Fitting condition {idx + 1}/{len(datasets)}: {dataset.path_to_data_file} ===")
            params = self._solver.run(self.model, dataset)
            u_free_x, pitch = extract_wind_condition(dataset)
            results.append(PerConditionResult(params=params, u_free_x=u_free_x,
                                              pitch=pitch, dataset=dataset))
        return results
