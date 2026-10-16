"""Generate training data for the drone flying in the y-z plane at various wall distances.

This is the script form of ``simulation/training_data_user_guide.ipynb``. All logic
lives here; tuning variables live in ``config.yaml`` next to this file. Run it from the
command line (the integrated project is less unstable from the command line than from a
notebook):

    python training_data_user_guide.py                 # uses config.yaml
    python training_data_user_guide.py --config training
    python training_data_user_guide.py --config-file my_config.yaml

Notes (from the original user guide):
- Turning off the disturbance observer is not a requirement. Any given controller output
  can correspond to a disturbance force.
- Run the scenario in ``simulation_user_guide.ipynb`` first to make sure the drone does
  not hit the wall. Adjust the wall distance based on performance.
- Use the same trajectory in every condition (same setting as the Neural Fly paper).
- ``RandomWaypointsInConstrainedSpace`` flies in a constrained 3D space in front of a wall.
- When training a disturbance model with the inflow model in the controller, enable the
  inflow model option in the controller.
- Use the baseline disturbance observer to generate training data.
- Ensure the disturbance model is set to the correct condition, i.e. the setting of the
  sinusoidal wind.
"""

import argparse
import os
import sys

import numpy as np
import yaml

# Make the simulation package modules (manager, ...) and the repo root (drone, ...)
# importable regardless of the current working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_SIM_DIR)
for _p in (_SIM_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

import manager
import drone.disturbance_model
import drone.trajectory

DEFAULT_CONFIG_FILE = os.path.join(_THIS_DIR, "config.yaml")


def extra_format(value):
    """Format a signed float for use in a file name: 1.5 -> '1_5', -1.5 -> 'n1_5'."""
    val = abs(value)
    s = f"{val:.1f}".replace('.', '_')
    return f"n{s}" if value < 0 else s


def build_trajectory(spec):
    """Instantiate a drone.trajectory class from a {type, args} spec."""
    traj_type = spec["type"]
    args = spec.get("args", [])
    traj_class = getattr(drone.trajectory, traj_type)
    return traj_class(*args)


def build_wind_conditions(spec):
    """Enumerate the list of (x, y, z) free-wind tuples described by the config."""
    style = spec["style"]
    if style == "grid":
        return [
            (x, y, z)
            for x in spec["x"]
            for y in spec["y"]
            for z in spec["z"]
        ]
    if style == "polar":
        # x is horizontal (always negative), z is vertical.
        return [
            (-magnitude * np.cos(np.radians(pitch_deg)), 0.0, magnitude * np.sin(np.radians(pitch_deg)))
            for magnitude in spec["magnitudes"]
            for pitch_deg in spec["pitch_angles_deg"]
        ]
    raise ValueError(f"Unknown wind_conditions style: {style!r}")


def run(config_file=DEFAULT_CONFIG_FILE, config_name=None):
    with open(config_file, "r") as f:
        full_config = yaml.safe_load(f)

    config_name = config_name or full_config["selected_config"]
    try:
        selected_config = full_config["configs"][config_name]
    except KeyError:
        raise ValueError(
            f"Unknown config {config_name!r}. "
            f"Available: {list(full_config['configs'])}"
        )

    print(f"Using config: {config_name}")

    trajectory = build_trajectory(selected_config["trajectory"])
    wind_conditions = build_wind_conditions(selected_config["wind_conditions"])
    is_csv = selected_config["is_csv"]
    t_data_collection = selected_config["t_data_collection"]
    file_name_template = selected_config["file_name_template"]

    # Account for controller tracking error to prevent collision with the wall.
    x_position_error = full_config.get("x_position_error", 0.0)
    wall_distance_range = [distance + x_position_error for distance in selected_config["wall_distance_range"]]

    sim_test = None
    for wall_distance in wall_distance_range:
        for wind_speed_x, wind_speed_y, wind_speed_z in wind_conditions:
            file_name = file_name_template.format(
                extra_format(wind_speed_x),
                extra_format(wind_speed_y),
                extra_format(wind_speed_z),
                extra_format(wall_distance),
            )
            # Skip cases already collected so a re-run resumes instead of recomputing.
            if manager.Manager.result_exists(file_name, is_csv):
                print(f"Skipping existing result: {file_name}")
                continue
            print(
                f"Running simulation with wind speed "
                f"({wind_speed_x}, {wind_speed_y}, {wind_speed_z}) and wall distance {wall_distance}"
            )
            # Initialize the simulation manager
            sim_test = manager.Manager()
            model = drone.disturbance_model.WindEffectNearWall(
                wall_origin=np.array([-wall_distance, 0, 0]),
                u_free=np.array([wind_speed_x, wind_speed_y, wind_speed_z]),
            )
            sim_test.set_up(disturbance_model=model, trajectory=trajectory)
            sim_test.run(t_data_collection)

            # Save the result
            if is_csv:
                sim_test.save_result_as_csv(file_name)
            else:
                sim_test.save_result_as_pkl(file_name)

    # Optionally, save the header index into a file for future reference.
    if full_config.get("save_column_map", False):
        if sim_test is not None:
            sim_test.logger.generate_column_map()
        else:
            print("No simulation was run (all results existed); skipping column map.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_FILE,
        help="Path to the YAML config file (default: config.yaml next to this script).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override selected_config: training | validation | fitting | fitting_validation.",
    )
    args = parser.parse_args()
    run(config_file=args.config_file, config_name=args.config)


if __name__ == "__main__":
    main()
