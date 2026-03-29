import numpy as np

import dynamics
import imu_model
import simulation.interface
import simulation.scenario

class Sensor(simulation.scenario.Sensor):
    """Converts ground-truth dynamics into simulated sensor outputs."""
    def __init__(self):
        self.imu_model = imu_model.ImuModel()

    def get_sensor_data(self, state: simulation.interface.DynamicsOutput, t: float):
        """Get all sensor data"""
        return simulation.interface.SensorData(
            position=state.position,
            v=state.v,
            pose=state.pose,
            omega=self.imu_model.create_noisified_omega(state.omega, t),
            v_dot=self.imu_model.create_noisified_accel(state.v_dot, t),
            rotors=state.rotors,
            omega_dot=state.omega_dot
        )
