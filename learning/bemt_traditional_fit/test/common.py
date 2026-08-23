import numpy as np
import pandas as pd


def _make_mock_df(n: int) -> pd.DataFrame:
    """Build a minimal DataFrame that satisfies FittingDataset.__init__."""
    rng = np.random.default_rng(0)
    data = {}
    for rotor in range(4):
        data[f"rotor_{rotor}_local_wind_velocity"] = [rng.standard_normal(3).tolist() for _ in range(n)]
        data[f"rotor_{rotor}_velocity"] = [rng.standard_normal(3).tolist() for _ in range(n)]
        data[f"rotor_{rotor}_rotation_spd"] = rng.uniform(100, 600, n).tolist()
        data[f"rotor_{rotor}_f_rotor_inertial_frame"] = [rng.standard_normal(3).tolist() for _ in range(n)]
    data["shared_r_disk"] = [np.eye(3).tolist() for _ in range(n)]
    data["sensed_dv"] = [rng.standard_normal(3).tolist() for _ in range(n)]
    data["sensed_omega"] = [rng.standard_normal(3).tolist() for _ in range(n)]
    return pd.DataFrame(data)
