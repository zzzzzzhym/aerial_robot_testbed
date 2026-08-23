import unittest
import numpy as np

from learning.bemt_traditional_fit.solver import Solver


class TestSolver(unittest.TestCase):

    def setUp(self):
        self.bounds = [(0.0, 10.0), (0.0, 5.0), (np.radians(10), np.radians(40))]
        self.solver = Solver(self.bounds, maxiter=5)

    def test_normalize_denormalize_roundtrip(self):
        x = np.array([5.0, 2.5, np.radians(25)])
        np.testing.assert_array_almost_equal(self.solver.denormalize(self.solver.normalize(x)), x)

    def test_normalize_lower_bound_gives_zero(self):
        lo = np.array([b[0] for b in self.bounds])
        z = self.solver.normalize(lo)
        np.testing.assert_array_almost_equal(z, np.zeros(len(self.bounds)))

    def test_normalize_upper_bound_gives_one(self):
        hi = np.array([b[1] for b in self.bounds])
        z = self.solver.normalize(hi)
        np.testing.assert_array_almost_equal(z, np.ones(len(self.bounds)))

    def test_run_returns_x_physical(self):
        # Trivial loss: minimum at lower bound
        loss_fn = lambda x: float(np.sum(x**2))
        seed = np.array([5.0, 2.5, np.radians(25)])
        result = self.solver.run(loss_fn, seed)
        self.assertTrue(hasattr(result, 'x_physical'))
        self.assertEqual(len(result.x_physical), len(self.bounds))


if __name__ == "__main__":
    unittest.main()
