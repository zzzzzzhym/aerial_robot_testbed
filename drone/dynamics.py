import numpy as np
from scipy.integrate import solve_ivp

import simulation.interface
import simulation.scenario

from drone import quaternion_updater
from drone import parameters as params
from drone import utils
from drone import disturbance_model
from drone import propeller
from drone import rotor
from drone.dynamics_state import State


class DroneDynamics(simulation.scenario.Dynamics):
    def __init__(self, drone: params.Multicopter, propeller: propeller.Propeller, disturbance: disturbance_model.DisturbanceForce, init_state: State, dt: float = 0.01) -> None:
        """
        pose is a 3x3 rotation matrix from body to inertial frame
        omega is in body fix frame
        inertial frame has front-left-up as x-y-z direction (FLU)
        """
        self.drone = drone
        self.dt = dt
        self.state = init_state
        # derivatives
        self.v_dot = np.array([0.0, 0.0, 0.0])
        self.omega_dot = np.array([0.0, 0.0, 0.0])  # in body fix frame
        self.q_dot = np.quaternion(1.0, 0.0, 0.0, 0.0)
        # input
        self.f = np.array([0.0, 0.0, 0.0])  # propulsion force (positive in body frame z direction); directly assigned from controller
        self.torque = np.array([0.0, 0.0, 0.0]) # in body fix frame
        # disturbance
        self.disturbance = disturbance
        self.f_disturb = self.disturbance.f_implicit + self.disturbance.f_explicit
        self.torque_disturb = self.disturbance.t_implicit + self.disturbance.t_explicit
        # rotor states
        self.rotors = rotor.RotorSet(self.drone, propeller)
        self.rotation_speed = np.array([0.0, 0.0, 0.0, 0.0])    # rotor rotation speed in rad/s

    def step(self, t: float, controller_output: simulation.interface.ControllerOutput) -> None:
        """Entry point of drone dynamics"""
        self.take_control_output(controller_output)
        self.rotors.step_all_rotor_states(self.state, self.rotation_speed)  # just to update rotor speed (should separate state and speed)
        y_0 = self.pack_state_vector()
        self.step_disturbance_force(t)
        y_t = self.step_state_vector(y_0, t)
        self.unpack_state_vector(y_t)
        y_t_dot = self.get_derivatives_from_eom(t, y_t)   # argument t does affect output
        self.unpack_state_derivatives(y_t_dot)
        self.rotors.step_all_rotor_states(self.state, self.rotation_speed)

    def take_control_output(self, controller_output: simulation.interface.ControllerOutput) -> None:
        """Take control output from controller"""
        self.f = controller_output.f
        self.torque = controller_output.torque
        self.rotation_speed = controller_output.rotation_speed

    def pack_state_vector(self) -> np.ndarray:
        y = np.array([self.state.position[0], # 0
                      self.state.position[1], # 1
                      self.state.position[2], # 2
                      self.state.v[0],        # 3
                      self.state.v[1],        # 4
                      self.state.v[2],        # 5
                      self.state.q[0],        # 6
                      self.state.q[1],        # 7
                      self.state.q[2],        # 8
                      self.state.q[3],        # 9
                      self.state.omega[0],    # 10
                      self.state.omega[1],    # 11
                      self.state.omega[2],    # 12
                      self.disturbance.f_implicit[0],    # 13
                      self.disturbance.f_implicit[1],    # 14
                      self.disturbance.f_implicit[2],    # 15
                      self.disturbance.t_implicit[0],    # 16
                      self.disturbance.t_implicit[1],    # 17
                      self.disturbance.t_implicit[2]])   # 18
                      
        return y

    def unpack_state_vector(self, y):
        self.state.position = y[0:3]
        self.state.v = y[3:6]
        self.state.q = np.array([y[6], y[7], y[8], y[9]])
        self.state.pose = utils.convert_quaternion_to_rotation_matrix(self.state.q)
        self.state.omega = y[10:13]
        self.f_disturb_implicit = y[13:16]

    def unpack_state_derivatives(self, y_dot):
        self.v_dot = y_dot[3:6]
        self.q_dot = np.quaternion(y_dot[6], y_dot[7], y_dot[8], y_dot[9])
        self.omega_dot = y_dot[10:13]
        self.f_disturb_dot = y_dot[13:16]

    def step_state_vector(self, y_0: np.ndarray, t: float) -> np.ndarray:
        t_start = t
        t_end = t + self.dt
        # t_start = 0
        # t_end = 0 + self.dt
        result = solve_ivp(self.get_derivatives_from_eom, 
                           (t_start, t_end), 
                           y_0, 
                           method = 'RK45', 
                           t_eval=[t_end])
        if not result.success:
            raise ValueError(result.message)
        return result.y.reshape(-1)    # convert an nx1 matrix to 1xn vector

    def step_disturbance_force(self, t: float):
        self.disturbance.update_explicit_wrench(t, self.state, self.rotors, self.f, self.torque)
        self.f_disturb = self.disturbance.f_implicit + self.disturbance.f_explicit
        self.torque_disturb = self.disturbance.t_implicit + self.disturbance.t_explicit

    def get_derivatives_from_eom(self, t: float, y: np.ndarray) -> np.ndarray:
        '''
        equations of motion:
        y = np.array([position[0],  # 0
                      position[1],  # 1
                      position[2],  # 2
                      v[0],         # 3
                      v[1],         # 4
                      v[2],         # 5
                      q[0],         # 6
                      q[1],         # 7
                      q[2],         # 8
                      q[3],         # 9
                      omega[0],     # 10
                      omega[1],     # 11
                      omega[2]])    # 12
        '''
        position = y[0:3]
        v = y[3:6]
        q_instance = quaternion_updater.QuaternionOfRotation(np.quaternion(y[6], y[7], y[8], y[9]))
        omega = y[10:13]
        omega_in_inertial_frame = self.state.pose@omega # possible bug: using state.pose instead of pose computed from y may cause inconsistency
        pose = utils.convert_quaternion_to_rotation_matrix(np.array([y[6], y[7], y[8], y[9]]))
        self.disturbance.f_implicit = y[13:16]
        state = State(position, v, 
                      utils.convert_quaternion_to_rotation_matrix(np.array((y[6], y[7], y[8], y[9]))), 
                      omega)
        self.rotors.step_all_rotor_states(state, self.rotation_speed)
        # self.disturbance.update_explicit_wrench(t, state, self.rotors, self.f, self.torque) # may significantly increase computation time

        position_dot = v
        v_dot = params.Environment.g*np.array([0.0, 0.0, -1.0]) + \
                pose@(self.f + self.disturbance.f_implicit + self.disturbance.f_explicit)/self.drone.m
        q_instance.step_derivative(omega_in_inertial_frame)     # quaternion derivative takes omega in intertal frame
        omega_dot = self.drone.inertia_inv@(self.torque + self.disturbance.t_explicit + self.disturbance.t_implicit -
                                        utils.get_hat_map(omega)@self.drone.inertia@omega)
        f_disturb_implicit_dot, t_disturb_implicit_dot = self.disturbance.get_implicit_wrench_derivatives()
        y_dot = np.array([  position_dot[0],  # 0
                            position_dot[1],  # 1
                            position_dot[2],  # 2
                            v_dot[0],         # 3
                            v_dot[1],         # 4
                            v_dot[2],         # 5
                            q_instance.q_dot.w, # 6
                            q_instance.q_dot.x, # 7
                            q_instance.q_dot.y, # 8
                            q_instance.q_dot.z, # 9
                            omega_dot[0],   # 10
                            omega_dot[1],   # 11
                            omega_dot[2],   # 12
                            f_disturb_implicit_dot[0],   # 13
                            f_disturb_implicit_dot[1],   # 14
                            f_disturb_implicit_dot[2],   # 15
                            t_disturb_implicit_dot[0],   # 16
                            t_disturb_implicit_dot[1],   # 17
                            t_disturb_implicit_dot[2]])  # 18
        return y_dot
    
    def get_dynamics_output(self) -> simulation.interface.DynamicsOutput:
        output = simulation.interface.DynamicsOutput(
            position=self.state.position,
            v=self.state.v,
            pose=self.state.pose,
            omega=self.state.omega,
            v_dot=self.v_dot,
            rotors=self.rotors,
            omega_dot=self.omega_dot
        )
        return output

    def get_extended_world_perception(self):
        return simulation.interface.ExtendedWorldPerception(np.zeros(3), np.zeros(3))  # supposed to be contact force

    def shutdown(self):
        pass
