import numpy as np

import drone.utils as utils
import sim_logger
import scenario
import interface

class Engine:
    """Core looping mechanism of simulation. Coordinates controller, trajectory and dynamics model.
    """

    def __init__(self, scenario: scenario.Scenario) -> None:
        self.scenario = scenario
        self.t = 0.0    # simulation epoch time
        self.dt_log = 0.01  # simulation step to log output
        self.dt_controller = 0.01   # controller cycle time
        self.dt_dynamics = self.scenario.dynamics.dt    # dynamics model cycle time
        # controller steps per log step, must be an integer
        self.cl_ratio = round(self.dt_log/self.dt_controller)
        # dynamics steps per controller step, must be an integer
        self.dc_ratio = round(self.dt_controller/self.dt_dynamics)
        print("number of controller steps per simulation step: " + str(self.cl_ratio))
        print("number of dynamics model steps per controller step: " +
              str(self.dc_ratio))
        # data to plot
        self.t_span = []
        self.ani = None # for animation

    def step_simulation(self, t: float, logger: sim_logger.Logger):
        t_controller = t
        for i in range(self.cl_ratio):
            t_controller += i*self.dt_controller
            self.scenario.trajectory.step_reference_state(self.t)
            t_dynamics = t_controller
            sensor_data = self.scenario.sensor.get_sensor_data(self.scenario.dynamics.get_dynamics_output(), t)
            self.scenario.controller.step(
                sensor_data,
                self.scenario.trajectory)
            for j in range(self.dc_ratio):
                t_dynamics += j*self.dt_dynamics
                self.scenario.dynamics.step(
                    t_dynamics, 
                    self.scenario.controller.get_control_output()
                )

            self.t += self.dt_controller
        self.log_sensor_data(logger, sensor_data)
        self.log_dynamics_output(logger, self.scenario.dynamics.get_dynamics_output())
        self.log_extended_world_perception(logger, self.scenario.dynamics.get_extended_world_perception())
        self.log_control_output(logger, self.scenario.controller.get_control_output())

    def run_simulation(self, logger: sim_logger.Logger, t_end):
        self.t_span = np.arange(0.0, t_end + self.dt_log, self.dt_log)
        for t in self.t_span:
            self.step_simulation(t, logger)
            self.log_states(logger)
        self.shutdown()

    def shutdown(self):
        self.scenario.dynamics.shutdown()

    def log_extended_world_perception(self, logger: sim_logger.Logger, perception: interface.ExtendedWorldPerception):
        logger.buffer["f_contact_normal"].append(logger.buffer["pose"][-1] @ perception.contact_force.copy())
        logger.buffer["tip_position"].append(perception.tip_position.copy())

    def log_sensor_data(self, logger: sim_logger.Logger, sensor_data: interface.SensorData):
        logger.buffer["sensed_dv"].append(sensor_data.v_dot.copy())
        logger.buffer["sensed_omega"].append(sensor_data.omega.copy())

    def log_dynamics_output(self, logger: sim_logger.Logger, dynamics_output: interface.DynamicsOutput):
        logger.buffer["position"].append(dynamics_output.get_position_in("inertial").copy())
        logger.buffer["q"].append(dynamics_output.q.copy())
        logger.buffer["v"].append(dynamics_output.get_velocity_in("inertial").copy())
        logger.buffer["dv"].append(dynamics_output.get_v_dot_in("inertial").copy())
        logger.buffer["pose"].append(dynamics_output.get_pose_in("inertial").copy())
        logger.buffer["omega"].append(dynamics_output.get_omega_in("inertial").copy())
        logger.buffer["omega_dot"].append(dynamics_output.get_omega_dot_in("inertial").copy())

        logger.buffer["rotor_0_rotation_spd"].append(dynamics_output.rotors.rotors[0].rotation_speed)
        logger.buffer["rotor_1_rotation_spd"].append(dynamics_output.rotors.rotors[1].rotation_speed)
        logger.buffer["rotor_2_rotation_spd"].append(dynamics_output.rotors.rotors[2].rotation_speed)
        logger.buffer["rotor_3_rotation_spd"].append(dynamics_output.rotors.rotors[3].rotation_speed)
        logger.buffer["rotor_0_local_wind_velocity"].append(dynamics_output.rotors.rotors[0].local_wind_velocity.copy())
        logger.buffer["rotor_1_local_wind_velocity"].append(dynamics_output.rotors.rotors[1].local_wind_velocity.copy())
        logger.buffer["rotor_2_local_wind_velocity"].append(dynamics_output.rotors.rotors[2].local_wind_velocity.copy())
        logger.buffer["rotor_3_local_wind_velocity"].append(dynamics_output.rotors.rotors[3].local_wind_velocity.copy())
        logger.buffer["rotor_0_f_rotor_inertial_frame"].append(dynamics_output.rotors.rotors[0].f_rotor_inertial_frame.copy())
        logger.buffer["rotor_1_f_rotor_inertial_frame"].append(dynamics_output.rotors.rotors[1].f_rotor_inertial_frame.copy())
        logger.buffer["rotor_2_f_rotor_inertial_frame"].append(dynamics_output.rotors.rotors[2].f_rotor_inertial_frame.copy())
        logger.buffer["rotor_3_f_rotor_inertial_frame"].append(dynamics_output.rotors.rotors[3].f_rotor_inertial_frame.copy())
        logger.buffer["rotor_0_thrust"].append(dynamics_output.rotors.rotors[0].thrust)
        logger.buffer["rotor_1_thrust"].append(dynamics_output.rotors.rotors[1].thrust)
        logger.buffer["rotor_2_thrust"].append(dynamics_output.rotors.rotors[2].thrust)
        logger.buffer["rotor_3_thrust"].append(dynamics_output.rotors.rotors[3].thrust)
        logger.buffer["shared_r_disk"].append(dynamics_output.rotors.rotors[0].pose.copy())
        logger.buffer["rotor_0_position"].append(dynamics_output.rotors.rotors[0].position_inertial_frame.copy())
        logger.buffer["rotor_1_position"].append(dynamics_output.rotors.rotors[1].position_inertial_frame.copy())
        logger.buffer["rotor_2_position"].append(dynamics_output.rotors.rotors[2].position_inertial_frame.copy())
        logger.buffer["rotor_3_position"].append(dynamics_output.rotors.rotors[3].position_inertial_frame.copy())
        logger.buffer["rotor_0_velocity"].append(dynamics_output.rotors.rotors[0].velocity_inertial_frame.copy())
        logger.buffer["rotor_1_velocity"].append(dynamics_output.rotors.rotors[1].velocity_inertial_frame.copy())
        logger.buffer["rotor_2_velocity"].append(dynamics_output.rotors.rotors[2].velocity_inertial_frame.copy())
        logger.buffer["rotor_3_velocity"].append(dynamics_output.rotors.rotors[3].velocity_inertial_frame.copy())
        logger.buffer["pose_dot"].append(utils.get_hat_map(
            dynamics_output.pose@dynamics_output.omega)@dynamics_output.pose)
        
        # dummy signals
        logger.buffer["f_disturb"].append(np.zeros(3))
        logger.buffer["torque_disturb"].append(np.zeros(3))
        logger.buffer["f_propeller"].append(np.zeros(3))
        logger.buffer["f_body"].append(np.zeros(3))
        logger.buffer["rotor_0_rotation_spd_delayed"].append(0)
        logger.buffer["rotor_1_rotation_spd_delayed"].append(0)
        logger.buffer["rotor_2_rotation_spd_delayed"].append(0)
        logger.buffer["rotor_3_rotation_spd_delayed"].append(0)

    def log_control_output(self, logger: sim_logger.Logger, controller_output: interface.ControllerOutput):
        logger.buffer["rotor_speeds_desired"].append(controller_output.rotation_speed.copy())

    def log_states(self, logger: sim_logger.Logger):
        # logger.buffer["f_disturb"].append(self.scenario.dynamics.f_disturb.copy())
        # logger.buffer["torque_disturb"].append(self.scenario.dynamics.torque_disturb.copy())

        # logger.buffer["f_propeller"].append(self.scenario.dynamics.disturbance.f_propeller.copy()) # only works for wind near wall disturbance
        # logger.buffer["f_body"].append(self.scenario.dynamics.disturbance.f_body.copy()) # only works for wind near wall disturbance
        # logger.buffer["rotor_0_rotation_spd_delayed"].append(self.scenario.dynamics.disturbance.delayed_rotor_set_speed[0]) # only works for wind near wall disturbance
        # logger.buffer["rotor_1_rotation_spd_delayed"].append(self.scenario.dynamics.disturbance.delayed_rotor_set_speed[1]) # only works for wind near wall disturbance
        # logger.buffer["rotor_2_rotation_spd_delayed"].append(self.scenario.dynamics.disturbance.delayed_rotor_set_speed[2]) # only works for wind near wall disturbance
        # logger.buffer["rotor_3_rotation_spd_delayed"].append(self.scenario.dynamics.disturbance.delayed_rotor_set_speed[3]) # only works for wind near wall disturbance

        # logger.buffer["tip_position"].append(self.scenario.dynamics.disturbance.wall_contact.tip_position_inertial_frame) # only works wall contact disturbance
        # logger.buffer["f_contact_normal"].append(self.scenario.dynamics.disturbance.wall_contact.f_contact_normal) # only works wall contact disturbance

        logger.buffer["e_x"].append(self.scenario.controller.e_x.copy())
        logger.buffer["e_v"].append(self.scenario.controller.e_v.copy())
        logger.buffer["e_a"].append(self.scenario.controller.e_a.copy())
        logger.buffer["e_j"].append(self.scenario.controller.e_j.copy())
        logger.buffer["e_r"].append(self.scenario.controller.e_r.copy())
        logger.buffer["e_omega"].append(self.scenario.controller.e_omega.copy())
        logger.buffer["psi_r_rd"].append(self.scenario.controller.psi_r_rd.copy())
        logger.buffer["f_ctrl_input"].append(self.scenario.dynamics.state.pose@self.scenario.controller.f.copy())
        logger.buffer["f_ctrl_input_dot"].append(self.scenario.dynamics.state.pose@self.scenario.controller.f_dot.copy())
        logger.buffer["f_d"].append(self.scenario.controller.f_d.copy())
        logger.buffer["f_d_dot"].append(self.scenario.controller.f_d_dot.copy())
        logger.buffer["f_d_dot2"].append(self.scenario.controller.f_d_dot2.copy())
        logger.buffer["f_feedback"].append(self.scenario.controller.f_feedback.copy())
        logger.buffer["f_feedforward"].append(self.scenario.controller.f_feedforward.copy())
        logger.buffer["f_disturb_compensation"].append(self.scenario.controller.f_disturb_compensation.copy())
        logger.buffer["torque_ctrl_input"].append(self.scenario.controller.torque.copy())
        logger.buffer["torque_feedback"].append(self.scenario.controller.torque_feedback.copy())
        logger.buffer["torque_coriolis"].append(self.scenario.controller.torque_coriolis.copy())
        logger.buffer["torque_feedforward"].append(self.scenario.controller.torque_feedforward.copy())
        logger.buffer["x_d"].append(self.scenario.trajectory.x_d.copy())
        logger.buffer["v_d"].append(self.scenario.trajectory.v_d.copy())
        logger.buffer["x_d_dot2"].append(self.scenario.trajectory.x_d_dot2.copy())
        logger.buffer["x_d_dot3"].append(self.scenario.trajectory.x_d_dot3.copy())
        logger.buffer["f_motor_desired"].append(self.scenario.controller.force_motor_desired.copy())


        logger.buffer["f_disturb_est"].append(self.scenario.controller.f_disturb.copy())
        logger.buffer["f_disturb_est_base"].append(self.scenario.controller.f_disturb_base.copy())
        logger.buffer["f_disturb_est_bemt"].append(self.scenario.controller.f_disturb_bemt.copy())
        logger.buffer["f_disturb_sensed_raw"].append(self.scenario.controller.f_disturb_sensed_raw.copy())
        logger.buffer["torque_disturb_est"].append(self.scenario.controller.torque_disturb.copy())
        logger.buffer["torque_disturb_est_base"].append(self.scenario.controller.torque_disturb_base.copy())
        logger.buffer["torque_disturb_est_bemt"].append(self.scenario.controller.torque_disturb_bemt.copy())
        logger.buffer["omega_desired"].append(self.scenario.controller.omega_desired.copy())
        logger.buffer["pose_desired"].append(self.scenario.controller.pose_desired.copy())
        logger.buffer["pose_desired_dot"].append(self.scenario.controller.pose_desired_dot.copy())
        logger.buffer["pose_desired_dot2"].append(self.scenario.controller.pose_desired_dot2.copy())
        logger.buffer["b_1d"].append(self.scenario.trajectory.b_1d.copy())




