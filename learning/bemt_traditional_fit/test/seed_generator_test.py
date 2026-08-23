import unittest
import numpy as np

from learning.bemt_traditional_fit.seed_generator import (
    generate_seeds, MultiSeedGenerator, SingleSeedGenerator,
)
from learning.bemt_traditional_fit.bemt_model import BemtModel


class TestGenerateSeeds(unittest.TestCase):

    def test_lhs_only_shape(self):
        bounds = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
        seeds = generate_seeds(bounds, n_lhs=10)
        self.assertEqual(seeds.shape, (10, 3))

    def test_bounds_respected(self):
        bounds = [(2.0, 50.0), (0.0, 50.0), (0.0, 5.0), (np.radians(10), np.radians(40)), (0.0, 10.0)]
        seeds = generate_seeds(bounds, n_lhs=30)
        for col, (lo, hi) in enumerate(bounds):
            self.assertTrue(np.all(seeds[:, col] >= lo), f"col {col} below lower bound")
            self.assertTrue(np.all(seeds[:, col] <= hi), f"col {col} above upper bound")

    def test_physical_seed_prepended_as_first_row(self):
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        physical_seed = np.array([0.3, 0.7])
        seeds = generate_seeds(bounds, n_lhs=5, physical_seed=physical_seed)
        np.testing.assert_array_equal(seeds[0], physical_seed)
        self.assertEqual(seeds.shape, (6, 2))  # 1 physical + 5 LHS

    def test_reproducible_with_same_random_seed(self):
        bounds = [(0.0, 1.0), (0.0, 2.0)]
        seeds1 = generate_seeds(bounds, n_lhs=10, random_seed=42)
        seeds2 = generate_seeds(bounds, n_lhs=10, random_seed=42)
        np.testing.assert_array_equal(seeds1, seeds2)

    def test_different_random_seeds_give_different_results(self):
        bounds = [(0.0, 1.0), (0.0, 2.0)]
        seeds1 = generate_seeds(bounds, n_lhs=10, random_seed=1)
        seeds2 = generate_seeds(bounds, n_lhs=10, random_seed=99)
        self.assertFalse(np.allclose(seeds1, seeds2))


class TestSeedGenerators(unittest.TestCase):

    def test_multi_seed_generator_shape(self):
        bounds = BemtModel.BOUNDS
        sg = MultiSeedGenerator(n_lhs=8)
        seeds = sg.get_seeds(bounds)
        self.assertEqual(seeds.shape, (8, len(bounds)))

    def test_multi_seed_generator_bounds_respected(self):
        bounds = BemtModel.BOUNDS
        sg = MultiSeedGenerator(n_lhs=20, random_seed=7)
        seeds = sg.get_seeds(bounds)
        for col, (lo, hi) in enumerate(bounds):
            self.assertTrue(np.all(seeds[:, col] >= lo))
            self.assertTrue(np.all(seeds[:, col] <= hi))

    def test_single_seed_generator_returns_one_row(self):
        guess = [5.3, 1.7, 1.8, np.radians(20.6), 0.0]
        sg = SingleSeedGenerator(guess)
        seeds = sg.get_seeds(BemtModel.BOUNDS)
        self.assertEqual(seeds.shape, (1, 5))
        np.testing.assert_array_almost_equal(seeds[0], guess)


if __name__ == "__main__":
    unittest.main()
