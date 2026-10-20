"""Unit tests for the training-data user-guide script.

These verify the config-driven logic: file-name formatting, the enumerated wind
conditions, and the trajectory selection per use case.
"""

import os
import sys
import unittest

import numpy as np
import yaml

# Make 'training_data_user_guide' importable (it lives one directory up from this test).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import training_data_user_guide as tdug  # noqa: E402

CONFIG_PATH = tdug.DEFAULT_CONFIG_FILE


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

    def test_polar_formula(self):
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


class TestVirtualConfig(unittest.TestCase):
    """Exercise the config-driven logic against a fixed virtual config.

    The live config.yaml values change over time, so values are asserted against
    this in-test config rather than whatever the live file currently holds.
    """

    VIRTUAL_CONFIG = {
        "t_data_collection": 12.0,
        "file_name_template": "wind_x{}_y{}_z{}_d{}",
        "wall_distance_range": [0.5, 100.0],
        "is_csv": True,
        "trajectory": {"type": "Figure8", "args": [20]},
        "wind_conditions": {
            "style": "grid",
            "x": [-2.0, 0.0],
            "y": [0.0],
            "z": [-4.0, 4.0],
        },
    }

    def test_wind_conditions(self):
        self.assertEqual(
            tdug.build_wind_conditions(self.VIRTUAL_CONFIG["wind_conditions"]),
            [(x, 0.0, z) for x in [-2.0, 0.0] for z in [-4.0, 4.0]],
        )

    def test_trajectory(self):
        traj = tdug.build_trajectory(self.VIRTUAL_CONFIG["trajectory"])
        self.assertEqual(type(traj).__name__, "Figure8")

    def test_file_name_formatting(self):
        name = self.VIRTUAL_CONFIG["file_name_template"].format(
            tdug.extra_format(-2.0),
            tdug.extra_format(0.0),
            tdug.extra_format(4.0),
            tdug.extra_format(100.0),
        )
        self.assertEqual(name, "wind_xn2_0_y0_0_z4_0_d100_0")


class TestRealConfigStructure(unittest.TestCase):
    """The live config.yaml must stay structurally valid; its values may change."""

    REQUIRED_KEYS = (
        "t_data_collection",
        "file_name_template",
        "wall_distance_range",
        "is_csv",
        "trajectory",
        "wind_conditions",
    )

    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH, "r") as f:
            cls.full = yaml.safe_load(f)

    def test_selected_config_exists(self):
        self.assertIn(self.full["selected_config"], self.full["configs"])

    def test_every_config_is_buildable(self):
        for name, cfg in self.full["configs"].items():
            with self.subTest(config=name):
                for key in self.REQUIRED_KEYS:
                    self.assertIn(key, cfg)
                self.assertTrue(hasattr(tdug.drone.trajectory, cfg["trajectory"]["type"]))
                self.assertGreater(len(tdug.build_wind_conditions(cfg["wind_conditions"])), 0)


if __name__ == "__main__":
    unittest.main()
