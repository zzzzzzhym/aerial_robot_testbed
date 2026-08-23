import numpy as np
from scipy.optimize import minimize


class Solver:
    """Normalizing optimizer: normalizes parameters before passing to scipy.minimize, denormalizes result."""

    def __init__(self, bounds, maxiter=100, method='Nelder-Mead', options=None):
        self.bounds = bounds
        self._lower = np.array([b[0] for b in bounds])
        self._scale = np.array([b[1] - b[0] for b in bounds])
        self.method = method
        default_opts = {'maxiter': maxiter, 'disp': True}
        self.options = {**default_opts, **(options or {})}

    def normalize(self, x):
        return (np.array(x) - self._lower) / self._scale

    def denormalize(self, z):
        return self._lower + np.array(z) * self._scale

    def run(self, loss_fn, seed, callback=None):
        """Run optimization from seed (physical space). Returns result with result.x_physical added."""
        z0 = self.normalize(seed)

        def normalized_loss(z):
            return loss_fn(self.denormalize(z))

        result = minimize(
            normalized_loss,
            z0,
            method=self.method,
            bounds=[(0.0, 1.0)] * len(self.bounds),
            callback=callback,
            options=self.options,
        )
        result.x_physical = self.denormalize(result.x)
        return result
