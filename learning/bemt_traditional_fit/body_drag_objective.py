import data_factory
from learning.bemt_traditional_fit.bemt_model import BemtModel
import inflow_model.propeller_lookup_table as propeller_lookup_table


class BodyDragObjective:
    """Fitting loss for body-drag-only estimation given a fixed propeller map.

    Aero coefficients (cl_1, cl_2, cd, alpha_0) are frozen; only k_body_drag
    is optimized. Thrust is computed via lookup table, not live BEMT.
    """

    def __init__(self, model: BemtModel, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        self.model = model
        self.lookup_table = lookup_table

    def get_loss(self, x, datasets: list[data_factory.FittingDataset]) -> float:
        self.model.apply_params(x)

        loss = 0.0
        for dataset in datasets:
            data_len = len(dataset.u_free_0)
            n_samples = max(1, data_len // self.model.sample_distance)
            loss_per_dataset = 0.0
            for i in range(0, data_len, self.model.sample_distance):
                f_residual = self.model.get_residual_force(
                    dataset, i, self.lookup_table, is_using_lookup_table=True
                )
                loss_per_dataset += f_residual[2]**2
            loss += loss_per_dataset / n_samples
        loss /= len(datasets)
        print(f"Current loss: {loss:.6f}")
        return loss
