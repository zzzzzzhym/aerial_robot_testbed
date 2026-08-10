import unittest
import unittest.mock
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv

from vtk_reader import VtkReader, get_wind_velocity_folder_name, get_file_path


def _make_synthetic_mesh(free_stream: np.ndarray) -> pv.UnstructuredGrid:
    grid = pv.ImageData(dimensions=(5, 5, 5), spacing=(2, 2, 2), origin=(-4, -4, -4))
    n_pts = grid.n_points
    grid = grid.cast_to_unstructured_grid()
    grid.point_data["velocity"] = np.tile(free_stream, (n_pts, 1)).astype(float)
    grid.point_data["density"] = np.ones(n_pts, dtype=float)
    return grid


def _build_reader(free_stream: np.ndarray) -> VtkReader:
    reader = VtkReader(base_path=Path("."))
    reader.free_stream_velocity = np.asarray(free_stream, dtype=float).copy()
    reader.mesh = _make_synthetic_mesh(free_stream)
    reader._is_outside_mesh_warned = False
    return reader


FREE_STREAM = np.array([3.0, 0.0, 0.0])


class TestGetVelocityAt(unittest.TestCase):
    def test_inside_point_returns_mesh_velocity(self):
        reader = _build_reader(FREE_STREAM)
        v = reader.get_velocity_at(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(v.shape, (3,))
        np.testing.assert_allclose(v, FREE_STREAM, atol=0.5)

    def test_outside_point_returns_freestream_not_zeros(self):
        reader = _build_reader(FREE_STREAM)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = reader.get_velocity_at(np.array([1000.0, 1000.0, 1000.0]))
        self.assertEqual(v.shape, (3,))
        self.assertFalse(np.allclose(v, 0.0), "Out-of-range point returned zeros; expected free-stream")
        np.testing.assert_allclose(v, FREE_STREAM, atol=1e-6)
        self.assertTrue(any("free-stream" in str(w.message).lower() for w in caught))

    def test_warning_issued_only_once(self):
        reader = _build_reader(FREE_STREAM)
        far = np.array([1000.0, 1000.0, 1000.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reader.get_velocity_at(far)
            reader.get_velocity_at(far)
            reader.get_velocity_at(far)
        freestream_warnings = [w for w in caught if "free-stream" in str(w.message).lower()]
        self.assertEqual(len(freestream_warnings), 1)


class TestGetWindVelocityFolderName(unittest.TestCase):
    def test_no_wall_distance(self):
        name = get_wind_velocity_folder_name(np.array([-3.0, 0.0, 5.0]))
        self.assertEqual(name, "xn3_yp0_zp5")

    def test_wall_distance_negative_integer(self):
        name = get_wind_velocity_folder_name(np.array([-3.0, 0.0, 5.0]), wall_distance=-1.0)
        self.assertEqual(name, "xn3_yp0_zp5_dn1")

    def test_wall_distance_negative_fractional(self):
        name = get_wind_velocity_folder_name(np.array([-3.0, 0.0, 0.0]), wall_distance=-0.5)
        self.assertEqual(name, "xn3_yp0_zp0_dn0.5")

    def test_wall_distance_positive(self):
        name = get_wind_velocity_folder_name(np.array([0.0, 0.0, 0.0]), wall_distance=0.5)
        self.assertEqual(name, "xp0_yp0_zp0_dp0.5")

    def test_wall_distance_large_no_wall(self):
        name = get_wind_velocity_folder_name(np.array([0.0, 0.0, 0.0]), wall_distance=-100.0)
        self.assertEqual(name, "xp0_yp0_zp0_dn100")

    def test_backward_compat_none_identical_to_no_arg(self):
        name_no_arg = get_wind_velocity_folder_name(np.array([-5.0, 0.0, 3.0]))
        name_none = get_wind_velocity_folder_name(np.array([-5.0, 0.0, 3.0]), wall_distance=None)
        self.assertEqual(name_no_arg, name_none)
        self.assertNotIn("_d", name_no_arg)


class TestLoadMeshFolderPathConstruction(unittest.TestCase):
    """Verify load_mesh_by_wind_velocity constructs the correct folder path,
    including wall distance when provided."""

    def _captured_base_dir(self, wind_velocity, wall_distance=None):
        """Run load_mesh_by_wind_velocity and return the base_dir passed to get_file_path."""
        reader = VtkReader(base_path=Path("/fake/export"))
        captured = []

        def fake_get_file_path(base_dir, known_filename):
            captured.append(Path(base_dir))
            raise FileNotFoundError("test-stop")

        with unittest.mock.patch("vtk_reader.get_file_path", side_effect=fake_get_file_path):
            try:
                reader.load_mesh_by_wind_velocity(np.asarray(wind_velocity), wall_distance=wall_distance)
            except FileNotFoundError:
                pass

        self.assertEqual(len(captured), 1, "get_file_path should be called exactly once")
        return captured[0]

    def test_wall_distance_included_in_folder_path(self):
        base_dir = self._captured_base_dir([-3.0, 0.0, 5.0], wall_distance=-0.5)
        self.assertEqual(base_dir.name, "xn3_yp0_zp5_dn0.5")

    def test_positive_wall_distance_in_folder_path(self):
        base_dir = self._captured_base_dir([0.0, 0.0, 0.0], wall_distance=1.0)
        self.assertEqual(base_dir.name, "xp0_yp0_zp0_dp1")

    def test_no_wall_distance_gives_wind_only_folder(self):
        base_dir = self._captured_base_dir([-3.0, 0.0, 5.0])
        self.assertEqual(base_dir.name, "xn3_yp0_zp5")
        self.assertNotIn("_d", base_dir.name)

    def test_base_path_is_prepended(self):
        base_dir = self._captured_base_dir([5.0, 0.0, 0.0], wall_distance=-0.5)
        self.assertEqual(base_dir.parent, Path("/fake/export"))


if __name__ == '__main__':
    unittest.main()
