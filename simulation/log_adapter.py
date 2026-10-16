import drone.utils as utils
from simulation import interface


def log_data_from_sensor(sensor_data: interface.SensorData) -> dict:
    return {
        "sensed_dv":    sensor_data.v_dot.copy(),
        "sensed_omega": sensor_data.omega.copy(),
    }


def log_data_from_dynamics(output: interface.DynamicsOutput) -> dict:
    data = {
        "position":  output.get_position_in("inertial").copy(),
        "q":         output.q.copy(),
        "v":         output.get_velocity_in("inertial").copy(),
        "dv":        output.get_v_dot_in("inertial").copy(),
        "pose":      output.get_pose_in("inertial").copy(),
        "omega":     output.get_omega_in("inertial").copy(),
        "omega_dot": output.get_omega_dot_in("inertial").copy(),
        "pose_dot":  (utils.get_hat_map(output.pose @ output.omega) @ output.pose).copy(),
    }
    for i, rotor in enumerate(output.rotors.rotors):
        data[f"rotor_{i}_rotation_spd"]           = rotor.rotation_speed
        data[f"rotor_{i}_thrust"]                 = rotor.thrust
        data[f"rotor_{i}_position"]               = rotor.position_inertial_frame.copy()
        data[f"rotor_{i}_velocity"]               = rotor.velocity_inertial_frame.copy()
        data[f"rotor_{i}_local_wind_velocity"]    = rotor.local_wind_velocity.copy()
        data[f"rotor_{i}_sensed_wind_velocity"]   = rotor.sensed_wind_velocity.copy()
        data[f"rotor_{i}_f_rotor_inertial_frame"] = rotor.f_rotor_inertial_frame.copy()
    data["shared_r_disk"] = output.rotors.rotors[0].pose.copy()
    return data


def log_data_from_perception(perception: interface.ExtendedWorldPerception) -> dict:
    return {
        "f_contact_normal": perception.contact_force.copy(),
        "tip_position":     perception.tip_position.copy(),
    }


def log_data_from_disturbance(disturbance) -> dict:
    if hasattr(disturbance, 'get_log_data'):
        return disturbance.get_log_data()
    return {}
