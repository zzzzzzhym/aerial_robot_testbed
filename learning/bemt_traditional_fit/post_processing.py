import numpy as np
import pandas as pd

import inflow_model.blade_params
from inflow_model.propeller_lookup_table import PropellerLookupTable
import data_factory


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
