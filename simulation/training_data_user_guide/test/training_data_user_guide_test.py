"""Unit tests for the training-data user-guide script.

These verify that the config-driven logic reproduces exactly what the original
``simulation/training_data_user_guide.ipynb`` produced: the same file-name formatting,
the same enumerated wind conditions, and the same trajectory selection per use case.
"""

import os
import sys
import unittest

import numpy as np
import yaml

# Make 'training_data_user_guide' importable (it lives one directory up from this test).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import training_data_user_guide as tdug  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(tdug.__file__), "config.yaml")


class TestExtraFormat(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(tdug.extra_format(1.5), "1_5")

    def test_zero(self):
        self.assertEqual(tdug.extra_format(0.0), "0_0")

    def test_negative(self):
        self.assertEqual(tdug.extra_format(-3.0), "n3_0")

    def test_large_wall(self):
        self.assertEqual(tdug.extra_format(100), "100_0")


class TestWindConditions(unittest.TestCase):
    def test_grid_style(self):
        spec = {"style": "grid", "x": [-3.0, -1.0, 0.0], "y": [0.0], "z": [-5.0, 3.0]}
        expected = [(x, 0.0, z) for x in [-3.0, -1.0, 0.0] for z in [-5.0, 3.0]]
        self.assertEqual(tdug.build_wind_conditions(spec), expected)

    def test_polar_matches_original_formula(self):
        magnitudes = [3.0, 5.0]
        pitch_angles_deg = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0]
        spec = {"style": "polar", "magnitudes": magnitudes, "pitch_angles_deg": pitch_angles_deg}
        expected = [
            (-m * np.cos(np.radians(p)), 0.0, m * np.sin(np.radians(p)))
            for m in magnitudes
            for p in pitch_angles_deg
        ]
        self.assertEqual(tdug.build_wind_conditions(spec), expected)

    def test_unknown_style_raises(self):
        with self.assertRaises(ValueError):
            tdug.build_wind_conditions({"style": "spiral"})


class TestTrajectory(unittest.TestCase):
    def test_random_waypoints_in_constrained_space(self):
        traj = tdug.build_trajectory({"type": "RandomWaypointsInConstrainedSpace", "args": [200, False]})
        self.assertEqual(type(traj).__name__, "RandomWaypointsInConstrainedSpace")

    def test_figure8(self):
        traj = tdug.build_trajectory({"type": "Figure8", "args": [20]})
        self.assertEqual(type(traj).__name__, "Figure8")


class TestConfigFileMatchesOriginalNotebook(unittest.TestCase):
    """Guard against config drift from the original notebook's intended values."""

    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH, "r") as f:
            cls.cfg = yaml.safe_load(f)["configs"]

    def _winds(self, name):
        return tdug.build_wind_conditions(self.cfg[name]["wind_conditions"])

    def test_training(self):
        c = self.cfg["training"]
        self.assertEqual(c["t_data_collection"], 20.0)
        self.assertEqual(c["wall_distance_range"], [100])
        self.assertEqual(c["trajectory"]["type"], "RandomWaypointsInConstrainedSpace")
        self.assertEqual(
            self._winds("training"),
            [(x, 0.0, z) for x in [-3.0, -1.0, 0.0] for z in [-5.0, -3.0, 0.0, 3.0, 5.0]],
        )

    def test_validation(self):
        c = self.cfg["validation"]
        self.assertEqual(c["t_data_collection"], 10.0)
        self.assertEqual(c["wall_distance_range"], [0.5])
        self.assertEqual(c["trajectory"]["type"], "Figure8")
        self.assertEqual(
            self._winds("validation"),
            [(x, 0.0, z) for x in [-2.0, 0.0] for z in [-4.0, -1.0, 1.0, 4.0]],
        )

    def test_fitting(self):
        c = self.cfg["fitting"]
        self.assertEqual(c["t_data_collection"], 15.0)
        self.assertEqual(c["wall_distance_range"], [100.0])
        self.assertEqual(c["trajectory"]["type"], "RandomWaypointsInConstrainedSpace")
        self.assertEqual(len(self._winds("fitting")), 14)

    def test_fitting_validation(self):
        c = self.cfg["fitting_validation"]
        self.assertEqual(c["t_data_collection"], 40.0)
        self.assertEqual(c["wall_distance_range"], [0.1])
        self.assertEqual(c["trajectory"]["type"], "Figure8")
        self.assertEqual(
            self._winds("fitting_validation"),
            [(x, 0.0, 0.0) for x in [-10.0, -8.0, -5.0, -2.0, 0.0]],
        )


if __name__ == "__main__":
    unittest.main()
