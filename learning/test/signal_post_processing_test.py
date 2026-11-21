"""Unit tests for the production signal post-processing data layer.

Runs the layer on a SMALL temp copy of a real training CSV (never the real data) and checks the
generated signal columns. From repo root:
    python -m unittest learning.test.signal_post_processing_test
"""
import glob
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import data_factory
from learning import signal_post_processing as spp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN = os.path.join(REPO, "data", "training", "wind_free_space_training_cfd_our_proposal")
N_ROWS = 25  # keep the per-frame lookup-table loop fast


def _dataset_from(path):
    df = pd.read_csv(path)
    data_factory.DataFactory.process_raw_data_frame(df)
    return data_factory.FittingDataset(df, path)


class TestSignalPostProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = sorted(glob.glob(os.path.join(TRAIN, "*.csv")))
        if not files:
            raise unittest.SkipTest("training CSVs not present")
        # pick the xn3_z5 corner condition (non-zero wind) so wind_magnitude is meaningful
        cls.src = next((f for f in files if "xn3" in f and "z5" in f), files[-1])
        cls.proc = spp.SignalPostProcessor()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv = os.path.join(self.tmp, "slice.csv")
        pd.read_csv(self.src).head(N_ROWS).to_csv(self.csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_columns_written_and_shaped(self):
        cols = self.proc.augment_csv(_dataset_from(self.csv))
        df = pd.read_csv(self.csv)
        for name in spp.GENERATED_COLUMNS:
            self.assertIn(name, df.columns)
        self.assertEqual(len(cols[spp.RESIDUAL]), N_ROWS)
        self.assertEqual(np.array(cols[spp.RESIDUAL]).shape, (N_ROWS, 3))
        self.assertEqual(np.array(cols[spp.WIND_BODY_DIRECTION]).shape, (N_ROWS, 3))
        self.assertEqual(np.array(cols[spp.ROTOR_V_I[0]]).shape, (N_ROWS,))

    def test_residual_matches_bemt_fit_path(self):
        """The migrated residual must equal the direct BemtModel.get_residual_force (body frame)."""
        ds = _dataset_from(self.csv)
        cols = self.proc.augment_csv(ds)
        direct = np.array([self.proc.model.get_residual_force(ds, i, self.proc.lookup_table, True, True)
                           for i in range(N_ROWS)])
        np.testing.assert_allclose(np.array(cols[spp.RESIDUAL]), direct, atol=1e-9)

    def test_wind_direction_unit_and_magnitude_consistent(self):
        cols = self.proc.augment_csv(_dataset_from(self.csv))
        wdir = np.array(cols[spp.WIND_BODY_DIRECTION])
        wmag = np.array(cols[spp.WIND_MAGNITUDE])
        np.testing.assert_allclose(np.linalg.norm(wdir, axis=1), 1.0, atol=1e-6)
        ds = _dataset_from(self.csv)
        u = 0.25 * (ds.u_free_0 + ds.u_free_1 + ds.u_free_2 + ds.u_free_3)
        np.testing.assert_allclose(wmag, np.linalg.norm(u, axis=1), atol=1e-6)

    def test_v_i_positive_and_plausible(self):
        cols = self.proc.augment_csv(_dataset_from(self.csv))
        v_i = np.stack([np.array(cols[spp.ROTOR_V_I[r]]) for r in range(4)], axis=1)
        self.assertTrue(np.all(v_i > 0.0))
        self.assertTrue(v_i.max() < 20.0)

    def test_idempotent(self):
        self.proc.augment_csv(_dataset_from(self.csv))
        df1 = pd.read_csv(self.csv)
        self.proc.augment_csv(_dataset_from(self.csv))
        df2 = pd.read_csv(self.csv)
        self.assertEqual(list(df1.columns), list(df2.columns))
        self.assertEqual(len(df1.columns), len(set(df1.columns)))
        pd.testing.assert_series_equal(df1[spp.WIND_MAGNITUDE], df2[spp.WIND_MAGNITUDE])


if __name__ == "__main__":
    unittest.main()
