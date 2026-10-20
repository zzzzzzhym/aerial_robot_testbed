"""Unit tests for the normalization-params generation in data_factory.

Guards the three bugs fixed in step 2:
  1. normalize_data must call the instance load_sim_data (not a static call).
  2. each feature must get its OWN mean/scale (per-column), not one global value.
  3. generated file must use the nested `field -> dim -> {mean,scale}` layout under the
     `input`/`label` keys so make_normalization_params can read it back.

Run from the repo root:
    python -m unittest learning.test.normalization_test
or directly:
    python learning/test/normalization_test.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

# make `import data_factory` / `common_utils` work regardless of launch directory
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "learning"))

import data_factory


class TestNormalizationParams(unittest.TestCase):
    def setUp(self):
        # use the shipped template map (21 input features, 3 label dims)
        self.map_file = os.path.join("template", "input_label_map_disturbance_force_label.yaml")
        self.factory = data_factory.SimpleDataFactory(self.map_file)
        self.n_in = len(self.factory.input_headers)
        self.n_lab = len(self.factory.label_headers)

        # synthetic data with a DISTINCT mean/std per column so a global-scale bug is caught
        rng = np.random.default_rng(0)
        self.N = 4000
        self.col_mean = np.arange(self.n_in, dtype=float) + 1.0            # 1,2,3,...
        self.col_std = (np.arange(self.n_in, dtype=float) + 1.0) * 0.5     # 0.5,1.0,1.5,...
        self.input_data = self.col_mean + self.col_std * rng.standard_normal((self.N, self.n_in))
        self.lab_mean = np.array([10.0, -5.0, 3.0])
        self.lab_std = np.array([2.0, 4.0, 0.5])
        self.label_data = self.lab_mean + self.lab_std * rng.standard_normal((self.N, self.n_lab))

        # feed our synthetic arrays instead of reading CSVs
        self.factory.load_sim_data = lambda file_name: (self.input_data, self.label_data)

        # write the params to a throwaway path so we never touch real data folders
        self.tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        self.tmp.close()
        os.remove(self.tmp.name)  # must not exist so make_normalization_params generates it
        self.factory.normalization_params_file_path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.remove(self.tmp.name)

    def test_per_feature_params_roundtrip(self):
        self.factory.make_normalization_params(["dummy_file.csv"])
        # population std (ddof=0), matching Normalization's variance definition
        exp_in_scale = 1.0 / self.input_data.std(0)
        exp_lab_scale = 1.0 / self.label_data.std(0)

        np.testing.assert_allclose(self.factory.input_mean_vector, self.input_data.mean(0), rtol=1e-6)
        np.testing.assert_allclose(self.factory.input_scale_vector, exp_in_scale, rtol=1e-6)
        np.testing.assert_allclose(self.factory.label_mean_vector, self.label_data.mean(0), rtol=1e-6)
        np.testing.assert_allclose(self.factory.label_scale_vector, exp_lab_scale, rtol=1e-6)

    def test_features_are_not_globally_scaled(self):
        """Regression test for the global-scale bug: distinct columns must get distinct params."""
        self.factory.make_normalization_params(["dummy_file.csv"])
        self.assertGreater(np.std(self.factory.input_mean_vector), 1.0)   # means span 1..21
        self.assertGreater(np.ptp(self.factory.input_scale_vector), 0.1)  # scales clearly differ

    def test_generated_file_structure(self):
        self.factory.make_normalization_params(["dummy_file.csv"])
        with open(self.tmp.name) as f:
            params = yaml.safe_load(f)
        self.assertIn("input", params)
        self.assertIn("label", params)              # not "output"
        self.assertNotIn("output", params)
        # nested field -> dim -> {mean, scale} for a multi-dim field
        self.assertIn("v", params["input"])
        self.assertIn(0, params["input"]["v"])
        self.assertIn("mean", params["input"]["v"][0])
        self.assertIn("scale", params["input"]["v"][0])

    def test_reload_matches_generation(self):
        """Second call (file now exists) must read back identical vectors."""
        self.factory.make_normalization_params(["dummy_file.csv"])
        first = self.factory.input_scale_vector.copy()
        factory2 = data_factory.SimpleDataFactory(self.map_file)
        factory2.normalization_params_file_path = self.tmp.name
        factory2.make_normalization_params(["dummy_file.csv"])
        np.testing.assert_allclose(factory2.input_scale_vector, first, rtol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
