from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """BET integration resolution for coarse and fine passes."""
    coarse_n_elements: int
    coarse_n_rotation_segments: int
    coarse_sample_distance: int
    fine_n_elements: int
    fine_n_rotation_segments: int
    fine_sample_distance: int


@dataclass
class SeedConfig:
    """LHS screening parameters."""
    n_lhs: int
    n_keep: int
    random_seed: int


@dataclass
class SolverConfig:
    """Iteration budgets and tolerances for each solver stage."""
    coarse_maxiter: int
    fine_maxiter: int
    single_maxiter: int
    single_fine_maxiter: int
    ftol: float
    xtol: float


@dataclass
class FittingConfig:
    """Top-level config passed to FittingManager; unpacked internally per layer."""
    model: ModelConfig
    seed: SeedConfig
    solver: SolverConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> FittingConfig:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            seed=SeedConfig(**raw.get("seed", {})),
            solver=SolverConfig(**raw.get("solver", {})),
        )
