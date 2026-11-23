import unittest
from unittest.mock import patch
import numpy as np

import inflow_model.blade_params as blade_params
from inflow_model.propeller_lookup_table import PropellerLookupTable


class TestMakerParallel(unittest.TestCase):
    """Parallel table build must give exactly the sequential result."""

    def _build(self, n_workers):
        maker = PropellerLookupTable.Maker(
            blade_params.P600_Blade(),
            omega_range=np.array([500.0]),
            u_free_x_range=np.array([0.0, 9.0]),
            pitch_range=np.array([0.0, np.radians(30.0)]),
        )
        with patch.object(PropellerLookupTable.Maker, "save_data"):   # do not write a yaml
            maker.make_propeller_lookup_table("unit_tmp", n_workers=n_workers)
        return maker.table

    def test_parallel_matches_sequential(self):
        sequential = self._build(n_workers=1)
        parallel = self._build(n_workers=2)
        self.assertEqual(parallel.shape, (2, 2, 1, 4))
        np.testing.assert_array_equal(parallel, sequential)
        self.assertGreater(sequential[0, 0, 0, 2], 0.0)   # thrust computed (not left at zero)


if __name__ == '__main__':
    unittest.main()
