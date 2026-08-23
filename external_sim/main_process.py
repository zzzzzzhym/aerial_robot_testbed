import arcpy as ar  # external simulation API
import numpy as np
from pathlib import Path

import simulation.interface
import simulation.scenario
import drone.utils
import drone.propeller
import drone.parameters
import drone.dynamics_state
import drone.disturbance_model
import drone.rotor
import external_sim.cfd_wind_field_lookup.vtk_reader


_NO_WALL_THRESHOLD = 50  # wall distances beyond this (m) are treated as free field


class SimpleMotorModel:
    def __init__(self):
        # parameters from screenshot
        self.throttle_threshold = 0.05
        self.slope = 523.6
        self.intercept = 104.7
        # identiffied slope and intercept:
        # self.slope = 325
        # self.intercept = 175

    def speed_to_throttle(self, rotor_speeds: np.ndarray) -> np.ndarray:
        """
        Convert rotor speed (rad/s) → throttle (0~1)
        """

        # inverse linear mapping
        throttle = (rotor_speeds - self.intercept) / self.slope

        # apply threshold (deadzone)
        throttle = np.where(throttle < self.throttle_threshold, 0.0, throttle)

        # clip to valid range
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
        self.configure_sim_environment(disturbance.u_free, disturbance.wind_field_model.wall_origin[0], init_state.get_position_in("inertial"))
        wall_distance = disturbance.wind_field_model.wall_origin[0]
        if abs(wall_distance) > _NO_WALL_THRESHOLD:
            print("Treating as no wall environment, using free stream wind velocity as background wind")
            self.background_wind_reader = external_sim.cfd_wind_field_lookup.vtk_reader.FreeStreamReader(disturbance.u_free)
        else:
            self.background_wind_reader = external_sim.cfd_wind_field_lookup.vtk_reader.VtkReader(Path(r"C:\Users\jiexu\Downloads\Aerocae Robotics - Geng 2025\export"))
            self.background_wind_reader.load_mesh_by_wind_velocity(disturbance.u_free, wall_distance)

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

    def configure_sim_environment(self, wind_speed, wall_offset, init_position):
        global_id = 0
        self.main_api.set_object_property_uint(global_id, 'export.auto_export', 0) 

        self.sensor = self.main_api.get_object('p600/core/info')

        self.flow_sensor_getters = [
            self.main_api.get_object('p600/flow_sensor_fl'),
            self.main_api.get_object('p600/flow_sensor_bl'),
            self.main_api.get_object('p600/flow_sensor_br'),
            self.main_api.get_object('p600/flow_sensor_fr')]

        self.motor_speed_setter_hub0_cw = self.main_api.get_object('p600/fl/motor_speed_control')
        self.motor_speed_setter_hub1_ccw = self.main_api.get_object('p600/bl/motor_speed_control')
        self.motor_speed_setter_hub2_cw = self.main_api.get_object('p600/br/motor_speed_control')
        self.motor_speed_setter_hub3_ccw = self.main_api.get_object('p600/fr/motor_speed_control')

        self.motor_model = SimpleMotorModel()
        self.motor_throttle_setters = [
            self.main_api.get_object('p600/fl/motor'),
            self.main_api.get_object('p600/bl/motor'),
            self.main_api.get_object('p600/br/motor'),
            self.main_api.get_object('p600/fr/motor'),
        ]

        self.rotor_force_getters = [
            self.main_api.get_object('p600/fl/rotor_force'),
            self.main_api.get_object('p600/bl/rotor_force'),
            self.main_api.get_object('p600/br/rotor_force'),
            self.main_api.get_object('p600/fr/rotor_force')]
        
        self.rotor_speed_getters = [
            self.main_api.get_object('p600/fl/rotor_speed_sensor'),
            self.main_api.get_object('p600/bl/rotor_speed_sensor'),
            self.main_api.get_object('p600/br/rotor_speed_sensor'),
            self.main_api.get_object('p600/fr/rotor_speed_sensor')
        ]

        self.end_effector_force_getters = self.main_api.get_object('p600/end_effector/force_sensor')
        
        block_fluid = self.main_api.get_system_by_name('Blocks Fluid')
        self.main_api.set_object_property_uint(block_fluid, 'export.enabled', 0)
        print("wind_speed: ", wind_speed)
        self.main_api.set_object_property_float(block_fluid, 'fluid.initial_velocity.x', wind_speed[0])
        self.main_api.set_object_property_float(block_fluid, 'fluid.initial_velocity.y', wind_speed[1])
        self.main_api.set_object_property_float(block_fluid, 'fluid.initial_velocity.z', wind_speed[2])

        drone_body = self.main_api.get_object('p600')
        print("init position: ", init_position)
        d_clipping_prevention = 0.1 # small distance to prevent clipping, init clipping may cause problems in contact force
        self.main_api.set_object_property_float(drone_body, 'position.x', init_position[0]+d_clipping_prevention) 
        self.main_api.set_object_property_float(drone_body, 'position.y', init_position[1])
        self.main_api.set_object_property_float(drone_body, 'position.z', init_position[2])

        # wall = self.main_api.get_object('floor/tracker')
        # self.main_api.set_object_property_float(wall, 'plane_origin.x', wall_offset)
        wall = self.main_api.get_object('floor')
        if abs(wall_offset) > _NO_WALL_THRESHOLD:
            self.main_api.set_object_property_uint(wall, 'enabled', 0)
        else:
            self.main_api.set_object_property_float(wall, 'position.x', wall_offset)

    def filter_motor_speed(self, speeds: np.ndarray) -> np.ndarray:
        # clip first (physical limits)
        speeds = np.clip(speeds, 50, 550)

        # initialize previous speeds if not exist
        if not hasattr(self, "_prev_speeds"):
            self._prev_speeds = speeds.copy()

        # rate limit
        max_rate = 20 / self.dt
        max_delta = max_rate * self.dt

        delta = speeds - self._prev_speeds
        delta = np.clip(delta, -max_delta, max_delta)

        speeds_limited = self._prev_speeds + delta

        # update state
        self._prev_speeds = speeds_limited.copy()

        return speeds_limited


    def set_motor_speed(self, speeds: np.ndarray):
        return {
            self.motor_speed_setter_hub0_cw: -speeds[0],
            self.motor_speed_setter_hub1_ccw: speeds[1],
            self.motor_speed_setter_hub2_cw: -speeds[2],
            self.motor_speed_setter_hub3_ccw: speeds[3]
        }
    
    def set_motor_throttles(self, speeds: np.ndarray):
        # speeds = np.array([1,1,1,1])*450
        # t = self.i*self.dt  

        # omega_mean = 250
        # omega_amp = 200
        # period = 1.0

        # omega = omega_mean + omega_amp * np.sin(
        #     2 * np.pi * t / period
        # )

        # speeds = np.ones(4) * omega
        throttles = self.motor_model.speed_to_throttle(speeds)
        # k = 0.5
        # throttles = [k, k, k, k]
        print("throttle level: ", throttles)
        return {
            setter: throttle
            for setter, throttle in zip(self.motor_throttle_setters, throttles)
        }

    def get_motor_speed(self, reply):
        speeds = [reply.get_output_of(getter)[0] for getter in self.rotor_speed_getters]
        if any(s is None for s in speeds):
            return np.zeros(len(self.rotor_speed_getters))

        speeds = np.array(speeds)
        speeds = np.abs(speeds)
        return speeds

    def get_rotor_forces_body_frame(self, reply):
        forces = [reply.get_output_of(getter)[0:3] for getter in self.rotor_force_getters]
        return forces

    def get_wind_speed(self, reply):
        return np.array([reply.get_output_of(getter) for getter in self.flow_sensor_getters])

    def get_end_effector_force(self, reply):
        package = reply.get_output_of(self.end_effector_force_getters)
        force = np.array(package[0:3])
        # torque = package[3:6]
        return force

    def save_dynamics_state(self, body_state_data, flow_speeds, rotation_speed, rotor_forces_body_frame):
        # body_state_data:
        # Dofs: 19
        # Dof 0-2: world pos x,y,z
        # Dof 3-6: world rotation quaternion w,x,y,z
        # Dof 7-9: world velocity x,y,z
        # Dof 10-12: local angular velocity x,y,z
        # Dof 13-15: world acceleration x,y,z
        # Dof 16-18: local angular acceleration x,y,z

        body_state = drone.dynamics_state.State(
            position=np.array(body_state_data[0:3]),
            v=np.array(body_state_data[7:10]),   
            pose=drone.utils.convert_quaternion_to_rotation_matrix(np.array(body_state_data[3:7])),
            omega=np.array(body_state_data[10:13]),  
        )

        print("flow speeds: ", flow_speeds)

        self.rotors.step_all_rotor_states(body_state, rotation_speed)
        for flow_speed, rotor in zip(flow_speeds, self.rotors.rotors):
            rotor.local_wind_velocity = self.background_wind_reader.get_velocity_at(rotor.position_inertial_frame)
            print("rotor.local_wind_velocity", rotor.local_wind_velocity) # debug
            # rotor.local_wind_velocity = np.array(flow_speed)  # somehow only this one works
            rotor.sensed_wind_velocity = np.array(flow_speed)
        for f_body, rotor in zip(rotor_forces_body_frame, self.rotors.rotors):
            rotor.set_force_from_body_frame(f_body)

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
            contact_force=self.get_end_effector_force(reply),
            tip_position=self.state.get_position_in("inertial"))    # temporary solution, should change back to tip

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
            [self.sensor, *self.flow_sensor_getters, *self.rotor_force_getters, *self.rotor_speed_getters, self.end_effector_force_getters])

        if reply.is_failed():
            raise RuntimeError('Simulation failed!')
        else:
            # flow_field = self.main_api.export(self.i)    # vtk for paraview will be saved in the same path to art file
            self.i += 1 # todo: convert to time-based index and better in string format
            self.save_dynamics_state(reply.get_output_of(self.sensor), 
                                     self.get_wind_speed(reply), 
                                     self.get_motor_speed(reply),
                                     self.get_rotor_forces_body_frame(reply))
            self.save_extended_world_perception(reply)            

    def shutdown(self):
        self.main_api.clear()
        self.main_api.close()
        print("Done.")


class Tarot960:
    pass



# p600/visual/actuator_surface
# p600/visual/collider

# p600/core/px4/info

# p600/core/fr/visual
# p600/core/fr/collider
# p600/core/fr/vp_fr
# p600/core/fr/fr_motor
# p600/core/fr/particle
# p600/core/fr/sensor
# p600/core/fr/flow_sensor
# p600/core/fr/motor_speed_control

# p600/core/bl/visual
# p600/core/bl/collider
# p600/core/bl/vp_bl
# p600/core/bl/bl_motor
# p600/core/bl/particle
# p600/core/bl/flow_sensor
# p600/core/bl/motor_speed_control

# p600/core/fl/visual
# p600/core/fl/collider
# p600/core/fl/vp_fl
# p600/core/fl/fl_motor
# p600/core/fl/particle
# p600/core/fl/motor_speed_control

# p600/core/br/visual
# p600/core/br/collider
# p600/core/br/vp_br
# p600/core/br/br_motor
# p600/core/br/particle
# p600/core/br/motor_speed_control

# p600/core/particle_export_center


# add force sensor for the rotors. 
# in drone param, consider offset of cog coordinate 
# when read back from the sim, convert position sensor to cog position