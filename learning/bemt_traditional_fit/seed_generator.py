import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import qmc


def generate_seeds(bounds, n_lhs=30, physical_seed=None, random_seed=42):
    """Generate candidate starting points via Latin Hypercube Sampling."""
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    sampler = qmc.LatinHypercube(d=len(bounds), seed=random_seed)
    z_seeds = sampler.random(n=n_lhs)
    seeds = qmc.scale(z_seeds, lower, upper)
    if physical_seed is not None:
        seeds = np.vstack([physical_seed, seeds])
    return seeds


class SeedGenerator(ABC):
    """Abstract base for seed generation strategies."""

    @abstractmethod
    def get_seeds(self, bounds) -> np.ndarray:
        raise NotImplementedError


class MultiSeedGenerator(SeedGenerator):
    """LHS-based multi-start seed generator."""

    def __init__(self, n_lhs, n_keep, random_seed, physical_seed=None):
        self.n_lhs = n_lhs
        self.n_keep = n_keep
        self.random_seed = random_seed
        self.physical_seed = physical_seed

    def get_seeds(self, bounds) -> np.ndarray:
        return generate_seeds(bounds, self.n_lhs, self.physical_seed, self.random_seed)


class SingleSeedGenerator(SeedGenerator):
    """Single fixed initial guess; passes it directly to the solver without screening."""

    def __init__(self, initial_guess):
        self.initial_guess = np.array(initial_guess, dtype=float)

    def get_seeds(self, bounds) -> np.ndarray:
        return self.initial_guess.reshape(1, -1)
