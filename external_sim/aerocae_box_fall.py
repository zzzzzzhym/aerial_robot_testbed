import arcpy as ar  # external simulation API
import numpy as np

import simulation.interface
import simulation.scenario
import drone.utils
import drone.propeller
import drone.parameters
import drone.dynamics_state
import drone.disturbance_model
import drone.rotor


class SimpleMotorModel:
    def __init__(self):
        self.throttle_threshold = 0.05
        self.slope = 523.6
        self.intercept = 104.7

    def speed_to_throttle(self, rotor_speeds: np.ndarray) -> np.ndarray:
        throttle = (rotor_speeds - self.intercept) / self.slope
        throttle = np.where(throttle < self.throttle_threshold, 0.0, throttle)
        throttle = np.clip(throttle, 0.0, 1.0)
        return throttle


class P600(simulation.scenario.Dynamics):
    def __init__(
        self,
        drone_params: drone.parameters.Multicopter,
        propeller: drone.propeller.Propeller,
        disturbance: drone.disturbance_model.WindEffectNearWall,
        init_state: drone.dynamics_state.State,
        dt: float = 0.01,
    ):
        print("-----------external simulator init-----------")
        self.t_end = 10.0
        self.dt = dt

        self.main_api = ar.Controller()
        self.sensor = self.main_api.get_object('body/info')
        self.motor_model = SimpleMotorModel()

        self.rotors = drone.rotor.RotorSet(drone_params, propeller)
        self.state = simulation.interface.DynamicsOutput(
            position=np.zeros(3),
            v=np.zeros(3),
            pose=np.eye(3),
            omega=np.zeros(3),
            v_dot=np.zeros(3),
            rotors=self.rotors,
            omega_dot=np.zeros(3),
        )
        self.contact_force = simulation.interface.ExtendedWorldPerception(
            contact_force=np.zeros(3),
            tip_position=np.zeros(3),
        )
        self.i = 0

        self.main_api.start()

    def filter_motor_speed(self, speeds: np.ndarray) -> np.ndarray:
        speeds = np.clip(speeds, 50, 550)
        if not hasattr(self, "_prev_speeds"):
            self._prev_speeds = speeds.copy()
        max_rate = 20 / self.dt
        max_delta = max_rate * self.dt
        delta = speeds - self._prev_speeds
        delta = np.clip(delta, -max_delta, max_delta)
        speeds_limited = self._prev_speeds + delta
        self._prev_speeds = speeds_limited.copy()
        return speeds_limited

    def set_motor_speed(self, speeds: np.ndarray):
        return {}

    def set_motor_throttles(self, speeds: np.ndarray):
        throttles = self.motor_model.speed_to_throttle(speeds)
        print("throttle level: ", throttles)
        return {}

    def get_motor_speed(self, reply):
        return np.zeros(4)

    def get_rotor_thrust(self, reply):
        return np.zeros(4)

    def get_wind_speed(self, reply):
        return np.zeros((4, 3))

    def get_end_effector_force(self, reply):
        return np.zeros(3)

    def save_dynamics_state(self, body_state_data, flow_speeds, rotation_speed, rotor_thrusts):
        body_state = drone.dynamics_state.State(
            position=np.array(body_state_data[0:3]),
            v=np.array(body_state_data[7:10]),
            pose=drone.utils.convert_quaternion_to_rotation_matrix(np.array(body_state_data[3:7])),
            omega=np.array(body_state_data[10:13]),
        )

        self.rotors.step_all_rotor_states(body_state, rotation_speed)
        for flow_speed, rotor in zip(flow_speeds, self.rotors.rotors):
            rotor.local_wind_velocity = np.array(flow_speed)
        for rotor_thrust, rotor in zip(rotor_thrusts, self.rotors.rotors):
            rotor.thrust = np.array(rotor_thrust)

        self.state = simulation.interface.DynamicsOutput(
            position=np.array(body_state_data[0:3]),
            v=np.array(body_state_data[7:10]),
            pose=drone.utils.convert_quaternion_to_rotation_matrix(np.array(body_state_data[3:7])),
            omega=np.array(body_state_data[10:13]),
            v_dot=np.array(body_state_data[13:16]),
            rotors=self.rotors,
            omega_dot=np.array(body_state_data[16:19]),
        )

    def save_extended_world_perception(self, reply):
        self.contact_force = simulation.interface.ExtendedWorldPerception(
            contact_force=np.zeros(3),
            tip_position=np.zeros(3),
        )

    def get_dynamics_output(self) -> simulation.interface.DynamicsOutput:
        return self.state

    def get_extended_world_perception(self) -> simulation.interface.ExtendedWorldPerception:
        return self.contact_force

    def step(self, t, command: simulation.interface.ControllerOutput) -> simulation.interface.DynamicsOutput:
        if False:
            filtered_motor_speed = self.filter_motor_speed(command.rotation_speed)
            world_intput = self.set_motor_speed(filtered_motor_speed)
        else:
            world_intput = self.set_motor_throttles(command.rotation_speed)

        print("-----------external simulator-----------")
        print(self.i)
        print("t=", t + self.dt)
        reply = self.main_api.simulate_until(
            t + self.dt,
            world_intput,
            [self.sensor])

        if reply.is_failed():
            raise RuntimeError('Simulation failed!')
        else:
            self.i += 1
            self.save_dynamics_state(reply.get_output_of(self.sensor),
                                     self.get_wind_speed(None),
                                     self.get_motor_speed(None),
                                     self.get_rotor_thrust(None))
            self.save_extended_world_perception(None)

    def shutdown(self):
        self.main_api.clear()
        self.main_api.close()
        print("Done.")

# cam
# floor/mesh
# floor/collider
# box/mesh
# box/info