from dataclasses import dataclass

import data_factory
from learning.bemt_traditional_fit.bemt_model import BemtModel


@dataclass
class ObjectiveWeights:
    """Configuration for the fitting objective loss function."""
    horizontal_weight: float


class FittingObjective:
    """Computes the fitting loss given a parameter vector."""

    def __init__(self, model: BemtModel, weights: ObjectiveWeights = None):
        self.model = model
        self.weights = weights

    def get_loss(self, x, datasets: list[data_factory.FittingDataset], lookup_table=None, is_using_lookup_table=False):
        self.model.blade.cl_1, self.model.blade.cl_2, self.model.blade.cd, self.model.blade.alpha_0, self.model.k_body_drag = x[:5]
        self.model.bet_instance.refresh_blade()

        horizontal_weight = self.weights.horizontal_weight if self.weights is not None else self.model.horizontal_weight

        loss = 0.0
        for dataset in datasets:
            data_len = len(dataset.u_free_0)
            loss_per_data_set = 0.0
            num_of_samples_per_dataset = data_len // self.model.sample_distance
            for i in range(0, data_len, self.model.sample_distance):
                f_residual = self.model.get_residual_force(dataset, i, lookup_table, is_using_lookup_table)
                loss_f = horizontal_weight*(f_residual[0]**2 + f_residual[1]**2) + f_residual[2]**2
                loss_per_data_set += loss_f
            loss += loss_per_data_set / num_of_samples_per_dataset
        loss = loss / len(datasets)
        print(f"Current loss: {loss}")
        return loss
