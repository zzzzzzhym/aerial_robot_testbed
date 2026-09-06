import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from inflow_model.blade_params import APC_8x6
import data_factory
from learning.bemt_traditional_fit.fitting_config import FittingConfig
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective

_CONFIG = FittingConfig.from_yaml(
    Path(__file__).parent.parent / "config_single_rotor.yaml"
)


def _make_mock_dataset(n: int, omega_val: float = 300.0,
                       sensed_wind: np.ndarray = None) -> data_factory.FittingDataset:
    """Build a minimal FittingDataset with rotor_0_sensed_wind_velocity set.

    sensed_wind: 3-vector applied uniformly to all n samples (default zeros).
    """
    if sensed_wind is None:
        sensed_wind = np.zeros(3)
    df_data = {}
    for rotor in range(4):
        df_data[f"rotor_{rotor}_local_wind_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data[f"rotor_{rotor}_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data[f"rotor_{rotor}_rotation_spd"] = [omega_val if rotor == 0 else 0.0] * n
        df_data[f"rotor_{rotor}_f_rotor_inertial_frame"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data[f"rotor_{rotor}_sensed_wind_velocity"] = [sensed_wind.tolist() for _ in range(n)]
    df_data["shared_r_disk"] = [np.eye(3).tolist() for _ in range(n)]
    df_data["sensed_dv"] = [np.zeros(3).tolist() for _ in range(n)]
    df_data["sensed_omega"] = [np.zeros(3).tolist() for _ in range(n)]
    return data_factory.FittingDataset(pd.DataFrame(df_data), "mock")


_STAGE_0 = 1.0
_STAGE_1 = 10.0


def _throttle(t):
    """Replicates PropellerTestStand._compute_throttle_rotor0 without importing the class."""
    if t < _STAGE_0:
        return 0.0
    return float(np.clip((t - _STAGE_0) / _STAGE_1, 0.0, 1.0))


class TestPropellerTestStandThrottleLogic(unittest.TestCase):
    """Tests stage transition logic without importing arcpy/pyvista."""

    def test_stage0_zero_throttle(self):
        self.assertAlmostEqual(_throttle(0.0), 0.0)
        self.assertAlmostEqual(_throttle(0.99), 0.0)

    def test_stage1_start(self):
        self.assertAlmostEqual(_throttle(1.0), 0.0)

    def test_stage1_midpoint(self):
        self.assertAlmostEqual(_throttle(6.0), 0.5)

    def test_stage1_end_clipped(self):
        self.assertAlmostEqual(_throttle(11.0), 1.0)
        self.assertAlmostEqual(_throttle(20.0), 1.0)


class TestSingleRotorBemtModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.blade = APC_8x6()
        cls.model = SingleRotorBemtModel(cls.blade, is_ccw_rotor0=False, model_config=_CONFIG.model)

    def test_compute_rotor0_thrust_zero_omega(self):
        f = self.model.compute_rotor0_thrust(
            u_sensed=np.zeros(3),
            r_disk=np.eye(3),
            omega=0.0,
        )
        np.testing.assert_array_almost_equal(f, np.zeros(3), decimal=5)

    def test_compute_rotor0_thrust_positive_z_for_spinning(self):
        # With downward sensed wind (simulating induced inflow) and spinning rotor,
        # BET should predict positive z-thrust (lift direction)
        u_sensed = np.array([0.0, 0.0, -2.0])  # downward inflow in inertial (FLU)
        f = self.model.compute_rotor0_thrust(
            u_sensed=u_sensed,
            r_disk=np.eye(3),
            omega=300.0,
        )
        self.assertGreater(f[2], 0.0, "BET should predict positive z-thrust (lift)")

    def test_get_residual_force_shape(self):
        dataset = _make_mock_dataset(5, omega_val=300.0)
        residual = self.model.get_residual_force(dataset, 0)
        self.assertEqual(residual.shape, (3,))

    def test_get_residual_force_zero_when_sensor_matches_bet(self):
        u_sensed = np.array([0.0, 0.0, -2.0])
        dataset = _make_mock_dataset(2, omega_val=200.0, sensed_wind=u_sensed)
        f_bet = self.model.compute_rotor0_thrust(
            u_sensed=u_sensed, r_disk=np.eye(3), omega=200.0
        )
        dataset.rotor_0_f_rotor_inertial_frame[0] = f_bet  # r_disk = I so disk == inertial
        residual = self.model.get_residual_force(dataset, 0)
        np.testing.assert_array_almost_equal(residual, np.zeros(3), decimal=5)

    def test_bounds_have_four_entries(self):
        self.assertEqual(len(SingleRotorBemtModel.BOUNDS), 4)

    def test_sensed_wind_loaded_from_dataset(self):
        u_sensed = np.array([1.0, 0.5, -2.0])
        dataset = _make_mock_dataset(3, sensed_wind=u_sensed)
        np.testing.assert_allclose(dataset.rotor_0_sensed_wind_velocity[0], u_sensed)

    def test_dataset_without_sensed_wind_loads_none(self):
        """Datasets recorded before this field was added should still load."""
        n = 2
        df_data = {}
        for rotor in range(4):
            df_data[f"rotor_{rotor}_local_wind_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
            df_data[f"rotor_{rotor}_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
            df_data[f"rotor_{rotor}_rotation_spd"] = [0.0] * n
            df_data[f"rotor_{rotor}_f_rotor_inertial_frame"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data["shared_r_disk"] = [np.eye(3).tolist() for _ in range(n)]
        df_data["sensed_dv"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data["sensed_omega"] = [np.zeros(3).tolist() for _ in range(n)]
        dataset = data_factory.FittingDataset(pd.DataFrame(df_data), "mock")
        self.assertIsNone(dataset.rotor_0_sensed_wind_velocity)


class TestSingleRotorObjective(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        blade = APC_8x6()
        cls.model = SingleRotorBemtModel(blade, is_ccw_rotor0=False, model_config=_CONFIG.model)
        cls.objective = SingleRotorObjective(cls.model)

    def test_get_loss_returns_scalar(self):
        dataset = _make_mock_dataset(10, omega_val=200.0)
        x = np.array([5.3, 1.7, 1.8, np.radians(20.6)])
        loss = self.objective.get_loss(x, [dataset])
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_get_loss_lower_when_sensor_matches_bet(self):
        x = np.array([5.3, 1.7, 1.8, np.radians(20.6)])
        u_sensed = np.array([0.0, 0.0, -2.0])

        dataset_zero = _make_mock_dataset(10, omega_val=200.0, sensed_wind=u_sensed)
        loss_zero = self.objective.get_loss(x, [dataset_zero])

        dataset_match = _make_mock_dataset(10, omega_val=200.0, sensed_wind=u_sensed)
        model_tmp = SingleRotorBemtModel(APC_8x6(), is_ccw_rotor0=False, model_config=_CONFIG.model)
        model_tmp.blade.cl_1, model_tmp.blade.cl_2 = 5.3, 1.7
        model_tmp.blade.cd, model_tmp.blade.alpha_0 = 1.8, np.radians(20.6)
        model_tmp.bet_instance.refresh_blade()
        for i in range(10):
            f_bet = model_tmp.compute_rotor0_thrust(
                u_sensed=u_sensed, r_disk=np.eye(3), omega=200.0
            )
            dataset_match.rotor_0_f_rotor_inertial_frame[i] = f_bet
        loss_match = self.objective.get_loss(x, [dataset_match])

        self.assertLess(loss_match, loss_zero)


if __name__ == "__main__":
    unittest.main()
