import numpy as np
import pandas as pd
from typing import Optional

import inflow_model.blade_params
from inflow_model.propeller_lookup_table import PropellerLookupTable
import data_factory
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.per_condition_result import PerConditionResult


def make_lookup_table(fitted_params, blade: inflow_model.blade_params.Blade, table_name: str, is_hover_only: bool = False):
    """Write a lookup table YAML from the first four fitted parameters (cl_1, cl_2, cd, alpha_0).

    Any trailing parameters (e.g. k_body_drag) are ignored.
    """
    blade.cl_1, blade.cl_2, blade.cd, blade.alpha_0 = fitted_params[:4]
    print(
        "Making lookup table with parameters:\n"
        f"cl_1 = {blade.cl_1}\n"
        f"cl_2 = {blade.cl_2}\n"
        f"cd = {blade.cd}\n"
        f"alpha_0 = {blade.alpha_0}"
    )
    if is_hover_only:
        PropellerLookupTable.Maker.make_propeller_lookup_table(
            table_name, blade,
            u_free_x_range=np.array([0.0]),
            pitch_range=np.array([0.0]),
        )
    else:
        PropellerLookupTable.Maker.make_propeller_lookup_table(table_name, blade)

_DEFAULT_QUADRATIC_OMEGA_RANGE = np.array(
    [0, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2600], dtype=float
)


def make_lookup_table_from_quadratic_fit(
        results: list[PerConditionResult],
        filename: str,
        omega_range: Optional[np.ndarray] = None,
        is_ccw_rotor0: bool = True,
) -> None:
    """Build and save a PropellerLookupTable YAML from per-condition quadratic fits.

    This is the post-processing step after FittingManager.run_quadratic_fit().
    It traverses whatever conditions are present in `results` — no assumption on
    how many there are.

    Args:
        results: list of PerConditionResult from manager.run_quadratic_fit();
                 each carries its fitted params and wind condition (u_free_x, pitch).
        filename: output YAML name (no .yaml extension).
        omega_range: rotor speeds (rad/s) at which to evaluate the quadratics.
        is_ccw_rotor0: if False (CW rotor), f_y is negated in the stored table.
    """
    if omega_range is None:
        omega_range = _DEFAULT_QUADRATIC_OMEGA_RANGE.copy()

    model = QuadraticRotorModel()

    print(f"[make_lookup_table_from_quadratic_fit] Building table from {len(results)} condition(s)...")

    # Build sorted unique grid axes from the wind conditions already in each result.
    conditions_info = []
    for r in results:
        conditions_info.append((round(r.u_free_x, 4), round(r.pitch, 6), r.params))
        print(f"  u_free_x={r.u_free_x:.3f} m/s  pitch={np.degrees(r.pitch):.1f} deg")

    u_vals = sorted(set(c[0] for c in conditions_info))
    p_vals = sorted(set(c[1] for c in conditions_info))
    cond_map = {(c[0], c[1]): c[2] for c in conditions_info}

    n_u, n_p, n_w = len(u_vals), len(p_vals), len(omega_range)
    table = np.zeros((n_u, n_p, n_w, 4))  # [f_x, f_y, f_z, v_i]

    for i, u in enumerate(u_vals):
        for j, p in enumerate(p_vals):
            params = cond_map.get((u, p))
            if params is None:
                print(f"  Warning: no data for u_free_x={u:.4f}, pitch={np.degrees(p):.1f} deg — left as zero.")
                continue
            model.apply_params(params)
            for k, omega in enumerate(omega_range):
                forces = model.predict_forces_disk(float(omega))
                if not is_ccw_rotor0:
                    forces[1] *= -1
                table[i, j, k, :3] = forces
                # v_i (index 3) stays zero

    PropellerLookupTable.Maker.save_data(
        filename,
        np.array(u_vals),
        np.array(p_vals),
        np.array(omega_range, dtype=float),
        table,
    )


def make_residual_force_columns(model, dataset: data_factory.FittingDataset, lookup_table):
    """Append an f_residual column to the dataset's CSV file.

    model must implement get_residual_force(dataset, i, lookup_table, True, True).
    """
    f_residual = [
        model.get_residual_force(dataset, i, lookup_table, True, True)
        for i in range(len(dataset))
    ]
    f_residual = np.array(f_residual)
    df = pd.read_csv(dataset.path_to_data_file)
    df["f_residual"] = f_residual.tolist()
    df.to_csv(dataset.path_to_data_file, index=False, float_format='%.17f')
