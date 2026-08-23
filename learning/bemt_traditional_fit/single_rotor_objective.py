import data_factory
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel


class SingleRotorObjective:
    """Fitting loss for the single-rotor test stand.

    Residual = f_BET_disk - f_sensor_disk for rotor 0.
    Loss is the mean squared residual norm over all datasets and samples.
    """

    def __init__(self, model: SingleRotorBemtModel):
        self.model = model

    def get_loss(self, x, datasets: list[data_factory.FittingDataset]) -> float:
        self.model.blade.cl_1, self.model.blade.cl_2, self.model.blade.cd, self.model.blade.alpha_0 = x[:4]
        self.model.bet_instance.refresh_blade()

        loss = 0.0
        for dataset in datasets:
            data_len = len(dataset)
            n_samples = max(1, data_len // self.model.sample_distance)
            loss_per_dataset = 0.0
            for i in range(0, data_len, self.model.sample_distance):
                f_residual = self.model.get_residual_force(dataset, i)
                loss_per_dataset += float(f_residual @ f_residual)
            loss += loss_per_dataset / n_samples
        loss /= len(datasets)
        print(f"Current loss: {loss:.6f}")
        return loss
