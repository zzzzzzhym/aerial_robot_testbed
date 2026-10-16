import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from inflow_model.blade_params import APC_8x6
import data_factory
from learning.bemt_traditional_fit.fitting_config import FittingConfig
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.manager import FittingManager

_CONFIG = FittingConfig.from_yaml(
    Path(__file__).parent.parent / "config_single_rotor.yaml"
)


def _make_mock_dataset(n: int, omega_val: float = 300.0,
                       u_free: np.ndarray = None,
                       sensed_wind: np.ndarray = None) -> data_factory.FittingDataset:
    if u_free is None:
        u_free = np.zeros(3)
    if sensed_wind is None:
        sensed_wind = np.zeros(3)
    df_data = {}
    for rotor in range(4):
        df_data[f"rotor_{rotor}_local_wind_velocity"] = [u_free.tolist() for _ in range(n)]
        df_data[f"rotor_{rotor}_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data[f"rotor_{rotor}_rotation_spd"] = [omega_val if rotor == 0 else 0.0] * n
        df_data[f"rotor_{rotor}_f_rotor_inertial_frame"] = [np.zeros(3).tolist() for _ in range(n)]
        df_data[f"rotor_{rotor}_sensed_wind_velocity"] = [sensed_wind.tolist() for _ in range(n)]
    df_data["shared_r_disk"] = [np.eye(3).tolist() for _ in range(n)]
    df_data["sensed_dv"] = [np.zeros(3).tolist() for _ in range(n)]
    df_data["sensed_omega"] = [np.zeros(3).tolist() for _ in range(n)]
    df_data["v"] = [np.zeros(3).tolist() for _ in range(n)]
    return data_factory.FittingDataset(pd.DataFrame(df_data), "mock")


class TestComputeRotor0ThrustBemt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.blade = APC_8x6()
        cls.model = SingleRotorBemtModel(cls.blade, is_ccw_rotor0=False, model_config=_CONFIG.model)

    def test_returns_ndarray_shape_3(self):
        f = self.model.compute_rotor0_thrust_bemt(
            u_free=np.zeros(3),
            v_forward=np.zeros(3),
            r_disk=np.eye(3),
            omega=300.0,
        )
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.shape, (3,))

    def test_zero_omega_yields_near_zero_thrust(self):
        f = self.model.compute_rotor0_thrust_bemt(
            u_free=np.zeros(3),
            v_forward=np.zeros(3),
            r_disk=np.eye(3),
            omega=0.0,
        )
        self.assertAlmostEqual(np.linalg.norm(f), 0.0, places=5)

    def test_positive_z_thrust_in_hover(self):
        f = self.model.compute_rotor0_thrust_bemt(
            u_free=np.zeros(3),
            v_forward=np.zeros(3),
            r_disk=np.eye(3),
            omega=300.0,
        )
        self.assertGreater(f[2], 0.0, "BEMT should predict positive z-thrust in hover")

    def test_bemt_and_bet_differ_in_hover(self):
        # BET with zero sensed wind gives different result from BEMT with zero background wind
        # because BEMT root-finds v_i while BET sees zero inflow (different operating point)
        f_bemt = self.model.compute_rotor0_thrust_bemt(
            u_free=np.zeros(3), v_forward=np.zeros(3), r_disk=np.eye(3), omega=300.0
        )
        f_bet = self.model.compute_rotor0_thrust(
            u_sensed=np.zeros(3), r_disk=np.eye(3), omega=300.0
        )
        # BEMT accounts for induced velocity, so thrust should be lower than BET-with-zero-inflow
        self.assertFalse(
            np.allclose(f_bemt, f_bet, atol=1e-4),
            "BEMT and BET should differ when inflow differs",
        )


class TestUseBemtFlag(unittest.TestCase):

    def test_use_bemt_default_false(self):
        blade = APC_8x6()
        model = SingleRotorBemtModel(blade, is_ccw_rotor0=False, model_config=_CONFIG.model)
        self.assertFalse(model.use_bemt)

    def test_get_residual_force_bemt_mode_shape(self):
        blade = APC_8x6()
        model = SingleRotorBemtModel(blade, is_ccw_rotor0=False, model_config=_CONFIG.model)
        model.use_bemt = True
        dataset = _make_mock_dataset(3, omega_val=300.0)
        residual = model.get_residual_force(dataset, 0)
        self.assertEqual(residual.shape, (3,))

    def test_get_residual_force_bemt_vs_bet_differ_when_winds_differ(self):
        # When u_free (background) differs from sensed wind, BET and BEMT modes give different residuals.
        u_free = np.zeros(3)
        sensed = np.array([0.0, 0.0, -2.0])  # includes induced velocity

        blade_bet = APC_8x6()
        model_bet = SingleRotorBemtModel(blade_bet, is_ccw_rotor0=False, model_config=_CONFIG.model)
        model_bet.use_bemt = False

        blade_bemt = APC_8x6()
        model_bemt = SingleRotorBemtModel(blade_bemt, is_ccw_rotor0=False, model_config=_CONFIG.model)
        model_bemt.use_bemt = True

        dataset = _make_mock_dataset(2, omega_val=200.0, u_free=u_free, sensed_wind=sensed)

        r_bet = model_bet.get_residual_force(dataset, 0)
        r_bemt = model_bemt.get_residual_force(dataset, 0)

        self.assertFalse(
            np.allclose(r_bet, r_bemt, atol=1e-4),
            "BET and BEMT residuals should differ when background wind != sensed wind",
        )

    def test_get_residual_force_bemt_uses_u_free_not_sensed(self):
        # Changing u_free changes the BEMT residual; changing sensed wind does not.
        blade = APC_8x6()
        model = SingleRotorBemtModel(blade, is_ccw_rotor0=False, model_config=_CONFIG.model)
        model.use_bemt = True

        dataset_a = _make_mock_dataset(2, omega_val=200.0, u_free=np.zeros(3), sensed_wind=np.array([0.0, 0.0, -2.0]))
        dataset_b = _make_mock_dataset(2, omega_val=200.0, u_free=np.array([0.0, 0.0, -1.0]), sensed_wind=np.array([0.0, 0.0, -2.0]))

        r_a = model.get_residual_force(dataset_a, 0)
        r_b = model.get_residual_force(dataset_b, 0)
        self.assertFalse(np.allclose(r_a, r_b, atol=1e-4), "Different u_free should change BEMT residual")

    def test_get_residual_force_bemt_same_sensed_different_u_free(self):
        # Opposite check: same u_free, different sensed wind → BEMT residual unchanged
        blade = APC_8x6()
        model = SingleRotorBemtModel(blade, is_ccw_rotor0=False, model_config=_CONFIG.model)
        model.use_bemt = True

        dataset_a = _make_mock_dataset(2, omega_val=200.0, u_free=np.zeros(3), sensed_wind=np.array([0.0, 0.0, -1.0]))
        dataset_b = _make_mock_dataset(2, omega_val=200.0, u_free=np.zeros(3), sensed_wind=np.array([0.0, 0.0, -3.0]))

        r_a = model.get_residual_force(dataset_a, 0)
        r_b = model.get_residual_force(dataset_b, 0)
        np.testing.assert_allclose(r_a, r_b, atol=1e-6,
                                   err_msg="BEMT residual should not change with sensed wind alone")


class TestForSingleRotorBemt(unittest.TestCase):

    def _make_datasets(self):
        return [_make_mock_dataset(4, omega_val=300.0)]

    def test_factory_sets_use_bemt_true(self):
        datasets = self._make_datasets()
        manager = FittingManager.for_single_rotor_bemt(
            APC_8x6(), is_ccw_rotor0=False, datasets=datasets
        )
        self.assertTrue(manager.model.use_bemt)

    def test_factory_for_single_rotor_use_bemt_false(self):
        datasets = self._make_datasets()
        manager = FittingManager.for_single_rotor(
            APC_8x6(), is_ccw_rotor0=False, datasets=datasets
        )
        self.assertFalse(manager.model.use_bemt)

    def test_init_guess_stored(self):
        datasets = self._make_datasets()
        init = [10.0, 2.0, 0.5, np.radians(15.0)]
        manager = FittingManager.for_single_rotor_bemt(
            APC_8x6(), is_ccw_rotor0=False, datasets=datasets, init_guess=init
        )
        np.testing.assert_allclose(manager.init_guess, init)

    def test_datasets_stored(self):
        datasets = self._make_datasets()
        manager = FittingManager.for_single_rotor_bemt(
            APC_8x6(), is_ccw_rotor0=False, datasets=datasets
        )
        self.assertIs(manager.datasets, datasets)


if __name__ == "__main__":
    unittest.main()
