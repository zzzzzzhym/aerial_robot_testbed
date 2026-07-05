"""Tests for the sensed-wind lookup table API.

Tests cover:
  1. Save/load round-trip for disk-frame fields.
  2. query_data_from_table_sensed_wind returns stored forces at exact grid points.
  3. Thrust consistency between free-stream and sensed-wind queries.
  4. get_rotor_forces_sensed_wind returns correct forces for horizontal (pitch=0) disk.
  5. get_rotation_speed_sensed_wind converges to same answer as free-stream path.
  6. Geometry: disk-frame decomposition for horizontal and vertical disk.
"""
import os
import sys
import unittest
import tempfile
import shutil
import numpy as np

# Allow running as `python test/propeller_lookup_table_sensed_wind_test.py` from inflow_model dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from propeller_lookup_table import PropellerLookupTable


def _make_tiny_table(tmp_dir, filename="test_tiny"):
    """Build a minimal synthetic lookup table with known values and save it to tmp_dir.

    Grid: u_free_x ∈ {0, 5}, pitch ∈ {0, π/6}, omega ∈ {100, 500}
    Forces are set analytically so we can verify them.
    """
    u_free_x_range = np.array([0.0, 5.0])
    pitch_range = np.array([0.0, np.pi / 6])
    omega_range = np.array([100.0, 500.0])
    n_wind_speeds, n_pitches, n_omegas = 2, 2, 2
    table = np.zeros((n_wind_speeds, n_pitches, n_omegas, 4))

    # Fill with deterministic synthetic values.
    for i, u in enumerate(u_free_x_range):
        for j, p in enumerate(pitch_range):
            for k, om in enumerate(omega_range):
                # Thrust proportional to omega^2, drag proportional to u*omega.
                f_z = 0.001 * om ** 2 + 0.01 * u * om
                f_x = -0.0005 * om * u
                f_y = 0.0
                v_i = 0.5 * np.sqrt(f_z + 1e-6)
                table[i, j, k, :] = [f_x, f_y, f_z, v_i]

    # Compute expected disk-frame sensed wind fields using the Maker helper.
    expected_disk_plane, expected_normal = PropellerLookupTable.Maker._compute_sensed_tables(
        u_free_x_range, pitch_range, table
    )

    # Write manually so we control the file path.
    import yaml
    u_disk_plane = expected_disk_plane
    u_normal = expected_normal
    data = {
        "omega_range": omega_range.tolist(),
        "u_free_x_range": u_free_x_range.tolist(),
        "pitch_range": pitch_range.tolist(),
        "table": table.tolist(),
        "u_sensed_disk_plane_table": u_disk_plane.tolist(),
        "u_sensed_normal_table": u_normal.tolist(),
    }
    fpath = os.path.join(tmp_dir, filename + ".yaml")
    with open(fpath, 'w') as f:
        yaml.dump(data, f)

    return table, expected_disk_plane, expected_normal, u_free_x_range, pitch_range, omega_range


class _TmpTableReader(PropellerLookupTable.Reader):
    """Reader subclass that looks for files in a temporary directory."""
    def __init__(self, filename: str, tmp_dir: str):
        self._tmp_dir = tmp_dir
        super().__init__(filename)

    def read_data(self, filename: str):
        import yaml
        fpath = os.path.join(self._tmp_dir, filename + ".yaml")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")
        with open(fpath, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.omega_range = np.array(data["omega_range"])
        self.u_free_x_range = np.array(data["u_free_x_range"])
        self.pitch_range = np.array(data["pitch_range"])
        self.table = np.array(data["table"])
        self.u_sensed_disk_plane_table = np.array(data["u_sensed_disk_plane_table"])
        self.u_sensed_normal_table = np.array(data["u_sensed_normal_table"])


class TestSaveLoadRoundTrip(unittest.TestCase):
    """Test that save_data correctly populates disk-frame fields and they round-trip."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_disk_plane_and_normal_tables_saved(self):
        table, exp_disk_plane, exp_normal, u_range, p_range, o_range = _make_tiny_table(self.tmp_dir)
        reader = _TmpTableReader("test_tiny", self.tmp_dir)

        np.testing.assert_array_almost_equal(reader.u_sensed_disk_plane_table, exp_disk_plane)
        np.testing.assert_array_almost_equal(reader.u_sensed_normal_table, exp_normal)

    def test_shapes(self):
        _make_tiny_table(self.tmp_dir)
        reader = _TmpTableReader("test_tiny", self.tmp_dir)
        n_wind_speeds, n_pitches, n_omegas = 2, 2, 2
        self.assertEqual(reader.u_sensed_disk_plane_table.shape, (n_wind_speeds, n_pitches, n_omegas))
        self.assertEqual(reader.u_sensed_normal_table.shape, (n_wind_speeds, n_pitches, n_omegas))


class TestQuerySensedWind(unittest.TestCase):
    """Test that query_data_from_table_sensed_wind returns correct forces at grid points."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.table, _, _, self.u_range, self.p_range, self.o_range = _make_tiny_table(self.tmp_dir)
        self.reader = _TmpTableReader("test_tiny", self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_query_at_grid_points(self):
        """At each grid vertex, querying with the stored disk-plane/normal values should recover forces."""
        for i in range(len(self.u_range)):
            for j in range(len(self.p_range)):
                for k in range(len(self.o_range)):
                    u_disk_plane = float(self.reader.u_sensed_disk_plane_table[i, j, k])
                    u_normal = float(self.reader.u_sensed_normal_table[i, j, k])
                    omega = float(self.o_range[k])
                    result = self.reader.query_data_from_table_sensed_wind(u_disk_plane, u_normal, omega)
                    expected = self.table[i, j, k, :]
                    np.testing.assert_array_almost_equal(
                        result, expected, decimal=2,
                        err_msg=f"Mismatch at grid point i={i}, j={j}, k={k}"
                    )


class TestDiskFrameDecomposition(unittest.TestCase):
    """Test get_rotor_forces_sensed_wind geometry for well-known disk orientations."""

    def setUp(self):
        self.reader = PropellerLookupTable.Reader("apc_8x6_with_trail")

    def test_horizontal_disk_no_wind_zero_thrust(self):
        """With omega=0 and no wind, thrust should be zero."""
        r_disk = np.eye(3)
        u_sensed = np.array([0.0, 0.0, 0.0])
        v_forward = np.array([0.0, 0.0, 0.0])
        omega = 0.0
        forces, v_i = self.reader.get_rotor_forces_sensed_wind(u_sensed, v_forward, r_disk, omega, True)
        np.testing.assert_array_almost_equal(forces, [0, 0, 0], decimal=3)

    def test_horizontal_disk_in_plane_wind_only_affects_plane_component(self):
        """For horizontal disk, wind in x-y plane → disk-plane = |wind|, disk-normal from v_i only."""
        r_disk = np.eye(3)  # disk normal = z-axis
        u_sensed = np.array([3.0, 0.0, 0.0])  # sensed wind entirely in disk plane
        v_forward = np.array([0.0, 0.0, 0.0])

        disk_normal = r_disk[:, 2]
        u_normal_scalar = float(u_sensed @ disk_normal)
        u_plane_vec = u_sensed - u_normal_scalar * disk_normal
        u_plane_mag = float(np.linalg.norm(u_plane_vec))

        self.assertAlmostEqual(u_normal_scalar, 0.0, places=10)
        self.assertAlmostEqual(u_plane_mag, 3.0, places=10)

    def test_vertical_disk_wind_along_normal(self):
        """For disk tilted 90° (normal along x), wind in x-direction → all in disk-normal, none in plane."""
        from bet import BladeElementTheory
        r_disk = BladeElementTheory.pitch_rotor_disk_along_y_axis(np.pi / 2)
        disk_normal = r_disk[:, 2]
        u_sensed = np.array([5.0, 0.0, 0.0])
        v_forward = np.zeros(3)
        u_relative = u_sensed - v_forward
        u_normal_scalar = float(u_relative @ disk_normal)
        u_plane_vec = u_relative - u_normal_scalar * disk_normal
        u_plane_mag = float(np.linalg.norm(u_plane_vec))

        self.assertAlmostEqual(u_plane_mag, 0.0, places=6)
        self.assertAlmostEqual(abs(u_normal_scalar), 5.0, places=6)

    def test_force_direction_in_inertial(self):
        """Thrust should be along disk normal in inertial frame for horizontal disk."""
        r_disk = np.eye(3)  # disk normal = z-axis
        v_forward = np.zeros(3)
        omega = 500.0
        u_sensed = np.zeros(3)  # no sensed wind
        forces, v_i = self.reader.get_rotor_forces_sensed_wind(u_sensed, v_forward, r_disk, omega, True)
        # For horizontal disk, thrust is in z direction
        self.assertGreater(forces[2], 0.0, "Thrust should be positive (upward) for horizontal disk")
        self.assertAlmostEqual(forces[0], 0.0, places=3)
        self.assertAlmostEqual(forces[1], 0.0, places=3)


class TestThrustConsistency(unittest.TestCase):
    """Thrust from sensed-wind query should match free-stream thrust for hovering (pitch=0, v_forward=0)."""

    def setUp(self):
        self.reader = PropellerLookupTable.Reader("apc_8x6_with_trail")

    def _free_stream_thrust(self, u_free_x, omega):
        u_free = np.array([u_free_x, 0.0, 0.0])
        v_forward = np.zeros(3)
        r_disk = np.eye(3)
        forces, _ = self.reader.get_rotor_forces(u_free, v_forward, r_disk, omega, True)
        return forces[2]  # z-component = thrust for horizontal disk

    def _sensed_wind_thrust(self, u_free_x, omega, v_i):
        """Construct sensed wind from u_free_x and v_i, then query sensed-wind path."""
        # For pitch=0: u_sensed_disk_plane = u_free_x, u_sensed_normal = -v_i (v_i in -z disk direction)
        # Sensed wind vector (horizontal disk): [u_free_x, 0, -v_i] in inertial (disk frame = inertial for pitch=0)
        u_sensed = np.array([u_free_x, 0.0, -v_i])
        v_forward = np.zeros(3)
        r_disk = np.eye(3)
        forces, _ = self.reader.get_rotor_forces_sensed_wind(u_sensed, v_forward, r_disk, omega, True)
        return forces[2]

    def test_hover_thrust_matches(self):
        """At specific grid nodes where we can extract v_i from the table, thrusts should match."""
        pitch_idx = np.argmin(np.abs(self.reader.pitch_range))
        omega_list = [500.0, 1000.0]
        u_free_x = 0.0
        u_idx = np.argmin(np.abs(self.reader.u_free_x_range - u_free_x))

        for omega_target in omega_list:
            omega_idx = np.argmin(np.abs(self.reader.omega_range - omega_target))
            omega_actual = self.reader.omega_range[omega_idx]
            v_i = float(self.reader.table[u_idx, pitch_idx, omega_idx, 3])

            thrust_free = self._free_stream_thrust(u_free_x, omega_actual)
            thrust_sensed = self._sensed_wind_thrust(u_free_x, omega_actual, v_i)

            # The lookup table interpolation may not be exact at grid points for the sensed
            # path (different interpolator), so allow 5% relative tolerance.
            if abs(thrust_free) > 1e-4:
                rel_err = abs(thrust_sensed - thrust_free) / abs(thrust_free)
                self.assertLess(rel_err, 0.05,
                    f"Thrust mismatch at omega={omega_actual}: free={thrust_free:.4f}, sensed={thrust_sensed:.4f}")
            else:
                self.assertAlmostEqual(thrust_sensed, thrust_free, places=3)


class TestGetRotationSpeed(unittest.TestCase):
    """Test get_rotation_speed (free-stream path) with a synthetic thrust profile."""

    def setUp(self):
        reader = PropellerLookupTable.Reader("apc_8x6_with_trail")
        omega_range = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        thrust_profile = np.array([0.1, 0.5, 0.3, 0.6, 0.9])  # non-monotonic

        self.reader = reader
        self.reader.omega_range = omega_range
        self.reader.u_free_x_range = np.array([0.0])
        self.reader.pitch_range = np.array([0.0])
        self.reader.table = np.zeros((1, 1, len(omega_range), 4))
        self.reader.table[0, 0, :, 2] = thrust_profile
        self.reader.get_interpolator()

        # zero wind + horizontal disk → u_norm=0, pitch=0, same as querying at (0.0, 0.0, omega)
        self.u_free = np.zeros(3)
        self.v_forward = np.zeros(3)
        self.r_disk = np.eye(3)

    def test_within_range(self):
        omega = self.reader.get_rotation_speed(self.u_free, self.v_forward, self.r_disk, omega_current=2000, thrust_desired=0.49)
        self.assertAlmostEqual(omega, 1975.0, places=1)

    def test_clip_below_min(self):
        omega = self.reader.get_rotation_speed(self.u_free, self.v_forward, self.r_disk, omega_current=3000, thrust_desired=-1.0)
        self.assertEqual(omega, 1000)

    def test_clip_above_max(self):
        omega = self.reader.get_rotation_speed(self.u_free, self.v_forward, self.r_disk, omega_current=1000, thrust_desired=5.0)
        self.assertEqual(omega, 5000)

    def test_flat_segment(self):
        self.reader.table[0, 0, :, 2] = np.array([0.1, 0.5, 0.5, 0.6, 0.9])
        self.reader.get_interpolator()
        omega = self.reader.get_rotation_speed(self.u_free, self.v_forward, self.r_disk, omega_current=2400, thrust_desired=0.5)
        self.assertEqual(omega, 2400)


class TestRotationSpeedSensedWind(unittest.TestCase):
    """Test get_rotation_speed_sensed_wind for a horizontal disk.

    Sensed wind convention: u_sensed = u_eq + v_i_vector, where v_i_vector = -v_i * disk_z.
    For hover (u_free_x=0, pitch=0) with horizontal disk (r_disk=I),
    u_sensed = [0, 0, -v_i] in the inertial frame.
    """

    def setUp(self):
        self.reader = PropellerLookupTable.Reader("apc_8x6_with_trail")
        self.r_disk = np.eye(3)
        self.v_forward = np.zeros(3)
        # Pick a grid omega close to 500 rad/s and read the corresponding v_i
        u_idx = np.argmin(np.abs(self.reader.u_free_x_range - 0.0))
        pitch_idx = np.argmin(np.abs(self.reader.pitch_range - 0.0))
        omega_idx = np.argmin(np.abs(self.reader.omega_range - 500.0))
        self.omega_grid = float(self.reader.omega_range[omega_idx])
        v_i = float(self.reader.table[u_idx, pitch_idx, omega_idx, 3])
        self.thrust_grid = float(self.reader.table[u_idx, pitch_idx, omega_idx, 2])
        # Horizontal disk: disk normal = +z; v_i in -z → sensed wind = [0, 0, -v_i]
        self.u_sensed = np.array([0.0, 0.0, -v_i])

    def test_sensed_wind_speed_returns_float(self):
        omega = self.reader.get_rotation_speed_sensed_wind(
            self.u_sensed, self.v_forward, self.r_disk, self.omega_grid, self.thrust_grid)
        self.assertIsInstance(omega, (float, np.floating))

    def test_higher_thrust_requires_higher_omega(self):
        """Requesting more thrust than the grid point should yield a higher omega."""
        omega_low = self.reader.get_rotation_speed_sensed_wind(
            self.u_sensed, self.v_forward, self.r_disk, self.omega_grid, self.thrust_grid * 0.5)
        omega_high = self.reader.get_rotation_speed_sensed_wind(
            self.u_sensed, self.v_forward, self.r_disk, self.omega_grid, self.thrust_grid * 2.0)
        self.assertLessEqual(omega_low, omega_high)

    def test_sensed_wind_speed_returns_positive(self):
        """get_rotation_speed_sensed_wind should return a finite positive value."""
        omega = self.reader.get_rotation_speed_sensed_wind(
            self.u_sensed, self.v_forward, self.r_disk, self.omega_grid, self.thrust_grid)
        self.assertGreater(omega, 0.0)
        self.assertTrue(np.isfinite(omega))


class TestNoWindSensedWindConsistency(unittest.TestCase):
    """Test the core 'no wind' scenario.

    Scenario: a table is written with known values.  When read back, we simulate
    a hovering rotor (no background wind, horizontal disk).  The sensed wind
    convention is u_sensed = u_eq + v_i_vector where v_i_vector = -v_i * disk_z
    (BET: v_i > 0 is flow in -z direction of disk).  In hover u_sensed = [0, 0, -v_i].
    Querying the sensed-wind path must return the same rotor forces as u_free = 0.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.table, _, _, self.u_range, self.p_range, self.o_range = _make_tiny_table(self.tmp_dir)
        self.reader = _TmpTableReader("test_tiny", self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_no_wind_sensed_equals_free_stream(self):
        """u_sensed = [0, 0, -v_i] must recover the same forces as u_free = 0."""
        r_disk = np.eye(3)       # horizontal disk: disk normal = +z
        v_forward = np.zeros(3)
        u_free_zero = np.zeros(3)

        pitch_idx = 0  # pitch_range[0] = 0.0
        u_idx = 0      # u_free_x_range[0] = 0.0

        for k, omega in enumerate(self.o_range):
            v_i = float(self.table[u_idx, pitch_idx, k, 3])
            # v_i > 0 is inflow in -z direction of disk (BET convention); disk normal = +z
            # u_sensed = u_eq + v_i_vector = 0 + (-v_i)*z_hat = [0, 0, -v_i]
            u_sensed = np.array([0.0, 0.0, -v_i])

            forces_free, _ = self.reader.get_rotor_forces(
                u_free_zero, v_forward, r_disk, omega, is_ccw_blade=True)
            forces_sensed, _ = self.reader.get_rotor_forces_sensed_wind(
                u_sensed, v_forward, r_disk, omega, is_ccw_blade=True)

            np.testing.assert_array_almost_equal(
                forces_sensed, forces_free, decimal=4,
                err_msg=f"Force mismatch at omega={omega}: "
                        f"free={forces_free}, sensed={forces_sensed}")



class TestGetBackgroundWind(unittest.TestCase):
    """Tests for Reader.get_background_wind(u_sensed, r_disk, omega)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.table, _, _, self.u_range, self.p_range, self.o_range = _make_tiny_table(self.tmp_dir)
        self.reader = _TmpTableReader("test_tiny", self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_returns_3d_array(self):
        r_disk = np.eye(3)
        u_sensed = np.array([0.0, 0.0, -1.0])
        result = self.reader.get_background_wind(u_sensed, r_disk, self.o_range[1])
        self.assertEqual(result.shape, (3,))

    def test_recovers_zero_background_wind(self):
        """At u_free=0, pitch=0 (horizontal disk), u_sensed = [0, 0, -v_i].
        The recovered u_background should be [0, 0, 0].
        """
        r_disk = np.eye(3)
        u_idx, pitch_idx = 0, 0  # u_free=0, pitch=0
        for k, omega in enumerate(self.o_range):
            v_i = float(self.table[u_idx, pitch_idx, k, 3])
            u_sensed = np.array([0.0, 0.0, -v_i])
            u_background = self.reader.get_background_wind(u_sensed, r_disk, omega)
            np.testing.assert_array_almost_equal(
                u_background, [0.0, 0.0, 0.0], decimal=3,
                err_msg=f"Background wind not zero at omega={omega}: got {u_background}")

    def test_recovers_free_stream_at_grid_point(self):
        """At a grid point (u_free=5, pitch=0), u_sensed = [5, 0, -v_i].
        get_background_wind should return [5, 0, 0] (= the original free-stream).
        """
        r_disk = np.eye(3)
        u_idx = 1   # u_free=5.0
        pitch_idx = 0  # pitch=0
        for k, omega in enumerate(self.o_range):
            u_free = self.u_range[u_idx]  # = 5.0
            v_i = float(self.table[u_idx, pitch_idx, k, 3])
            u_sensed = np.array([u_free, 0.0, -v_i])
            u_background = self.reader.get_background_wind(u_sensed, r_disk, omega)
            np.testing.assert_array_almost_equal(
                u_background, [u_free, 0.0, 0.0], decimal=2,
                err_msg=f"Background wind mismatch at u_free={u_free}, omega={omega}: got {u_background}")

    def test_no_inplane_component_unchanged(self):
        """The disk-plane component of u_background should match u_sensed's plane component
        since v_i only affects the disk-normal direction.
        """
        r_disk = np.eye(3)
        u_sensed = np.array([3.0, 2.0, -1.0])
        omega = self.o_range[0]
        u_background = self.reader.get_background_wind(u_sensed, r_disk, omega)
        # x and y (in-plane) should be unchanged
        np.testing.assert_almost_equal(u_background[0], u_sensed[0], decimal=3)
        np.testing.assert_almost_equal(u_background[1], u_sensed[1], decimal=3)

    def test_with_real_table(self):
        """Smoke test on the real p600 table: result should be finite and 3D."""
        reader = PropellerLookupTable.Reader("p600")
        r_disk = np.eye(3)
        u_idx = np.argmin(np.abs(reader.u_free_x_range - 0.0))
        pitch_idx = np.argmin(np.abs(reader.pitch_range - 0.0))
        omega_idx = np.argmin(np.abs(reader.omega_range - 500.0))
        omega = float(reader.omega_range[omega_idx])
        v_i = float(reader.table[u_idx, pitch_idx, omega_idx, 3])
        u_sensed = np.array([0.0, 0.0, -v_i])
        u_background = reader.get_background_wind(u_sensed, r_disk, omega)
        self.assertEqual(u_background.shape, (3,))
        self.assertTrue(np.all(np.isfinite(u_background)))
        np.testing.assert_array_almost_equal(u_background, [0.0, 0.0, 0.0], decimal=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
