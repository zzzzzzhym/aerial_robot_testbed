import numpy as np
import warnings

import utils
import sensor
import parameters as params
import trajectory
import disturbance_estimator
import inflow_model.propeller_lookup_table as propeller_lookup_table
import propeller
import simulation.interface
import simulation.scenario

class DroneController(simulation.scenario.Controller):
    def __init__(self, drone: params.Multicopter) -> None:
        # config
        # disturbance_model_name = "wind_free_space_training_diaml"
        disturbance_model_name = "wind_free_space_training_in_house_sim_diaml"
        # disturbance_model_name = "wind_near_wall_wo_bemt_in_control_train_xz_wind"
        # disturbance_model_name = "wind_near_wall_bemt_in_control_train_xz_wind"
        # disturbance_model_name = "wind_near_wall_wo_bemt_in_control_far_from_wall"
        # bemt_disturbance_model_name = "wind_free_space_training_bemt"
        bemt_disturbance_model_name = "wind_free_space_training_in_house_sim_bemt"
        # bemt_disturbance_model_name = "wind_near_wall_bemt_fitting"
        # propeller_lookup_table_name = "apc_8x6_with_trail"
        # propeller_lookup_table_name = "apc_8x6_fitted_in_noise_and_vibration"
        # propeller_lookup_table_name = "p600"
        propeller_lookup_table_name = "p600_full_range"
        self.is_using_baseline_disturbance_estimator = False
        self.is_using_pure_daiml_disturbance_estimator = True
        self.is_using_bemt_disturbance_estimator = False
        self.is_using_inflow_model = False
        print("DroneController: using inflow model: ", self.is_using_inflow_model)
        print("DroneController: using pure DAIML disturbance estimator: ", self.is_using_pure_daiml_disturbance_estimator)
        print("DroneController: using baseline disturbance estimator: ", self.is_using_baseline_disturbance_estimator)
        print("DroneController: using BEMT disturbance estimator: ", self.is_using_bemt_disturbance_estimator)

        self.params = drone
        self.e_x = np.array([0.0, 0.0, 0.0])
        self.e_v = np.array([0.0, 0.0, 0.0])
        self.e_a = np.array([0.0, 0.0, 0.0])
        self.e_j = np.array([0.0, 0.0, 0.0])
        self.e_r = np.array([0.0, 0.0, 0.0])
        self.e_omega = np.array([0.0, 0.0, 0.0])
        self.pose_desired = np.eye(3)  # intertial frame
        self.pose_desired_dot = np.eye(3)  # intertial frame
        self.pose_desired_dot2 = np.eye(3) # intertial frame
        self.omega_desired = np.array([0.0, 0.0, 0.0])
        self.omega_desired_dot = np.array([0.0, 0.0, 0.0])
        self.b_3d = np.array([0.0, 0.0, 0.0])
        self.b_2d = np.array([0.0, 0.0, 0.0])
        self.f = np.array([0.0, 0.0, 0.0])  # propulsion force, positive in -z direction of body frame. same convention as the paper
        self.f_dot = np.array([0.0, 0.0, 0.0])
        self.torque = np.array([0.0, 0.0, 0.0])
        self.f_d = np.array([0.0, 0.0, 0.0])    # desired force in inertial frame
        self.f_d_dot = np.array([0.0, 0.0, 0.0])
        self.psi_r_rd = 0.0 # debug use only
        self.force_motor_desired = np.array([0.0, 0.0, 0.0, 0.0])   # motor thrust in body frame, positive in -z direction of body frame
        self.force_motor_available = np.array([0.0, 0.0, 0.0, 0.0])
        self.rotation_speed = np.array([0.0, 0.0, 0.0, 0.0])
        self.disturbance_estimator = disturbance_estimator.DisturbanceEstimator(disturbance_model_name, 0.01)
        self.f_disturb = np.array([0.0, 0.0, 0.0])
        self.torque_disturb = np.array([0.0, 0.0, 0.0])        
        self.baseline_disturbance_estimator = disturbance_estimator.BaselineDisturbanceEstimator(0.01)
        self.f_disturb_base = np.array([0.0, 0.0, 0.0])
        self.torque_disturb_base = np.array([0.0, 0.0, 0.0])
        self.bemt_disturbance_estimator = disturbance_estimator.BemtFittedDisturbanceEstimatorV1(bemt_disturbance_model_name, 0.01)
        self.f_disturb_bemt = np.array([0.0, 0.0, 0.0])
        self.torque_disturb_bemt = np.array([0.0, 0.0, 0.0])
        self.propeller_force_table = propeller_lookup_table.PropellerLookupTable.Reader(propeller_lookup_table_name)
        self.propeller = propeller.apc_8x6
        self.is_warmed_up = False
        self.warm_up_count = 0
        self.warm_up_count_max = 0

        # saturation parameters
        self.max_f_feedback = 30.0  # feedback force saturation
        self.max_f_disturb_compensation = 30.0  # disturbance compensation force saturation
        self.b_2d_norm_min = 1e-3   # minimum norm to avoid singularity
        self.max_torque = 10.0  # maximum torque
        
        # internal var logging
        self.torque_feedback = np.array([0.0, 0.0, 0.0])
        self.torque_coriolis = np.array([0.0, 0.0, 0.0])
        self.torque_feedforward = np.array([0.0, 0.0, 0.0])
        self.f_feedback = np.array([0.0, 0.0, 0.0])
        self.f_feedforward = np.array([0.0, 0.0, 0.0])
        self.f_disturb_compensation = np.array([0.0, 0.0, 0.0])
        self.f_disturb_sensed_raw = np.array([0.0, 0.0, 0.0])

    def step(self, sensor_data: simulation.interface.SensorData, ref: trajectory.TrajectoryReference):
        self.step_tracking_error(sensor_data, ref)
        self.step_disturbance_estimator(sensor_data)
        # if not self.is_warmed_up:
        #     self.warm_up_count += 1
        #     # skip disturbance estimator at the first cycle to prevent large initial deviation
        #     self.f_disturb = np.array([0.0, 0.0, 0.0])
        #     self.torque_disturb = np.array([0.0, 0.0, 0.0]) 
        #     if self.warm_up_count > self.warm_up_count_max:
        #         self.is_warmed_up = True
      
        self.step_desired_force(sensor_data, ref)
        self.step_tracking_control(sensor_data)
        self.step_desired_pose(ref)
        self.step_attitude_error(sensor_data)
        self.step_attitude_control(sensor_data)
        self.step_error_function_so3(sensor_data)
        self.step_motor_output(sensor_data)

    def step_disturbance_estimator(self, sensor_data: simulation.interface.SensorData):
        # tracking_error = np.zeros(6)  # assume no tracking error term in disturbance estimator
        tracking_error = np.hstack((self.e_v, np.zeros(3)))
        self.f_disturb_sensed_raw = self.get_sensed_disturbance(sensor_data)[0:3]
        self.disturbance_estimator.step_disturbance(
            sensor_data.position,
            sensor_data.v,
            sensor_data.q,
            sensor_data.omega,
            self.f[2],
            self.torque,
            np.array([sensor_data.rotors.rotors[0].rotation_speed, 
                      sensor_data.rotors.rotors[1].rotation_speed, 
                      sensor_data.rotors.rotors[2].rotation_speed, 
                      sensor_data.rotors.rotors[3].rotation_speed]),
            self.get_sensed_disturbance(sensor_data),
            tracking_error
        )
        self.f_disturb = self.disturbance_estimator.get_disturbance_force()
        self.torque_disturb = self.disturbance_estimator.get_disturbance_torque()

        self.baseline_disturbance_estimator.step_disturbance(
            self.get_sensed_disturbance(sensor_data),
            tracking_error
        )
        self.f_disturb_base = self.baseline_disturbance_estimator.get_disturbance_force()
        self.torque_disturb_base = self.baseline_disturbance_estimator.get_disturbance_torque()

        predicted_force, predicted_torque = self.get_predicted_air_force(sensor_data)
        # predicted_force = predicted_force + self.f  # f_predicted = f_control + f_disturb; f_control = -self.f
        predicted_force = predicted_force*np.array([1.0, 1.0, 0.0]) # z contains rotor thrust, is not part of disturbance
        predicted_torque = predicted_torque - self.torque  # t_predicted = t_control + t_disturb; t_control = self.torque
        self.bemt_disturbance_estimator.step_disturbance(
            sensor_data.v,
            sensor_data.q,
            sensor_data.omega,
            sensor_data.rotors.rotors[0].local_wind_velocity, 
            sensor_data.rotors.rotors[1].local_wind_velocity, 
            sensor_data.rotors.rotors[2].local_wind_velocity, 
            sensor_data.rotors.rotors[3].local_wind_velocity,
            np.array([sensor_data.rotors.rotors[0].rotation_speed, 
                      sensor_data.rotors.rotors[1].rotation_speed, 
                      sensor_data.rotors.rotors[2].rotation_speed, 
                      sensor_data.rotors.rotors[3].rotation_speed]),
            self.get_sensed_disturbance(sensor_data),
            tracking_error,
            predicted_force,
            predicted_torque
        )
        self.f_disturb_bemt = self.bemt_disturbance_estimator.get_disturbance_force()
        self.torque_disturb_bemt = self.bemt_disturbance_estimator.get_disturbance_torque()*0.0 # assume no angular acceleration available to estimate torque disturbance

    def get_predicted_air_force(self, sensor_data: simulation.interface.SensorData):
        """Get predicted air force and torque on drone from propeller lookup table.

        Args:
            sensor_data (simulation.interface.SensorData): Current sensor measurements including rotor states

        Returns:
            tuple: (f_propeller, t_propeller) - Predicted force and torque in body frame
        """
        forces = []
        torques = []
        for rotor in sensor_data.rotors.rotors:
            force, v_i = self.propeller_force_table.get_rotor_forces(
                rotor.local_wind_velocity,
                rotor.velocity_inertial_frame,
                rotor.pose,
                rotor.rotation_speed,
                rotor.is_ccw_blade
            )
            forces.append(force)
            torques.append(np.cross(rotor.relative_position_inertial_frame, force))
        f_propeller = sum(forces)
        t_propeller = sum(torques)
        f_propeller = utils.FrdFluConverter.flip_vector(f_propeller)
        t_propeller = utils.FrdFluConverter.flip_vector(t_propeller)
        f_propeller = sensor_data.pose.T@f_propeller  # convert to body frame
        t_propeller = sensor_data.pose.T@t_propeller  # convert to body frame
        
        return f_propeller, t_propeller

    def get_sensed_disturbance(self, sensor_data: simulation.interface.SensorData):
        """Disturbance force in body frame"""
        f_disturb = (
            self.f
            + sensor_data.pose.T @ (
            sensor_data.v_dot * self.params.m
            - self.params.m * params.Environment.g * np.array([0.0, 0.0, 1.0])
            )
        )

        tq_disturbance = (
            self.params.inertia @ sensor_data.omega_dot
            + utils.get_hat_map(sensor_data.omega) @ self.params.inertia @ sensor_data.omega
            - self.torque
        )
    
        return np.hstack((f_disturb, tq_disturbance))

    def step_desired_force(self, sensor_data: simulation.interface.SensorData, ref: trajectory.TrajectoryReference, can_sense_jerk: bool=False, can_plan_jerk: bool=False):
        f_feedback = -params.Control.k_x*self.e_x - params.Control.k_v*self.e_v
        f_feedback = utils.saturate_vector_norm(f_feedback, self.max_f_feedback)

        f_disturb_compensation = np.array([0.0, 0.0, 0.0])
        if self.is_using_baseline_disturbance_estimator:
            f_disturb_compensation = -sensor_data.pose@self.f_disturb_base # add disturbance force
        elif self.is_using_pure_daiml_disturbance_estimator:
            f_disturb_compensation = -sensor_data.pose@self.f_disturb # add disturbance force
        elif self.is_using_bemt_disturbance_estimator:
            f_disturb_compensation = -sensor_data.pose@self.f_disturb_bemt # add disturbance force

        f_disturb_compensation = utils.saturate_vector_norm(f_disturb_compensation, self.max_f_disturb_compensation)

        e3 = np.array([0.0, 0.0, 1.0])
        f_feedforward = -self.params.m*params.Environment.g*e3 + self.params.m*ref.x_d_dot2
        self.f_d = f_feedback + f_feedforward + f_disturb_compensation
        if np.abs(self.f_d@self.f_d) < 0.0001:
            warnings.warn("DroneController: f_d too close to 0")
            self.f_d = -0.0001*e3    # z positive points down
        self.e_a = (sensor_data.pose@(-self.f) + self.params.m*params.Environment.g*e3)/self.params.m - ref.x_d_dot2
        if can_plan_jerk:
            self.f_d_dot = (-params.Control.k_x*self.e_v - params.Control.k_v *
                            self.e_a + self.params.m*ref.x_d_dot3)
        else:
            self.f_d_dot = (-params.Control.k_x*self.e_v - params.Control.k_v *
                            self.e_a)
        if can_sense_jerk:
            self.e_j = sensor_data.pose@(-self.f_dot)/self.params.m - ref.x_d_dot3
        else:
            self.e_j = np.array([0.0, 0.0, 0.0])
        if can_plan_jerk:
            self.f_d_dot2 = (-params.Control.k_x*self.e_a - params.Control.k_v *
                            self.e_j + self.params.m*ref.x_d_dot4)
        else:
            self.f_d_dot2 = (-params.Control.k_x*self.e_a - params.Control.k_v *
                            self.e_j)
            
        # logging
        self.f_feedback = f_feedback
        self.f_feedforward = f_feedforward
        self.f_disturb_compensation = f_disturb_compensation

    def step_tracking_control(self, sensor_data: simulation.interface.SensorData):
        """
        project desired force on b3 to generate control input force
        """
        self.f = self.f_d@(-sensor_data.pose[:, 2])*np.array([0.0, 0.0, 1.0])
        self.f_dot = self.f_d_dot@(-sensor_data.pose[:, 2])*np.array([0.0, 0.0, 1.0]) + \
            self.f_d@np.cross(sensor_data.pose@sensor_data.omega, -sensor_data.pose[:, 2])*np.array([0.0, 0.0, 1.0]) + \
            self.f_d@(-sensor_data.pose[:, 2])*np.cross(sensor_data.omega, np.array([0.0, 0.0, 1.0]))
        '''
        Or equivalently
        self.f_dot = sensor_data.pose.T@(self.f_d_dot@(-sensor_data.pose[:, 2])*(sensor_data.pose[:, 2]) + \
            self.f_d@np.cross(sensor_data.pose@sensor_data.omega, -sensor_data.pose[:, 2])*(sensor_data.pose[:, 2]) + \
            self.f_d@(-sensor_data.pose[:, 2])*np.cross(sensor_data.pose@sensor_data.omega, sensor_data.pose[:, 2]))
        '''

    def step_attitude_control(self, sensor_data: simulation.interface.SensorData, can_use_omega_desired_dot: bool=False):
        yaw_weight = 0.2
        w = np.diag([1.0, 1.0, yaw_weight]) # Mz is weaker
        torque_feedback = -params.Control.k_r*self.e_r - params.Control.k_omega*self.e_omega
        torque_feedback = w@torque_feedback
        torque_coriolis = np.cross(sensor_data.omega, self.params.inertia@sensor_data.omega)
        if can_use_omega_desired_dot:
            torque_feedforward = - self.params.inertia@(
                utils.get_hat_map(sensor_data.omega)@sensor_data.pose.T@self.omega_desired - sensor_data.pose.T@self.omega_desired_dot)
        else:
            torque_feedforward = - self.params.inertia@(
                utils.get_hat_map(sensor_data.omega)@sensor_data.pose.T@self.omega_desired)
        torque_feedback = utils.saturate_vector_norm(torque_feedback, self.max_torque)
        torque_feedforward = utils.saturate_vector_norm(torque_feedforward, self.max_torque)
        self.torque = torque_feedback + torque_coriolis + torque_feedforward

        # torque saturation
        self.torque = utils.saturate_vector_norm(self.torque, self.max_torque)

        # logging
        self.torque_feedback = torque_feedback
        self.torque_coriolis = torque_coriolis
        self.torque_feedforward = torque_feedforward

    def step_desired_pose(self, ref: trajectory.TrajectoryReference):
        b_3d, b_3d_dot, b_3d_dot2 = utils.get_unit_vector_derivatives(-self.f_d,
                                                                      -self.f_d_dot, -self.f_d_dot2)
        self.b_3d = b_3d
        b_2d_unnormalized = np.cross(self.b_3d, ref.b_1d)
        norm_proj = np.linalg.norm(b_2d_unnormalized)
        if norm_proj < self.b_2d_norm_min:
            print('Warning: DroneController: b_2d_unnormalized too close to 0, use previous value')
            b_2d = self.b_2d
            b_2d_dot = np.zeros(3)
            b_2d_dot2 = np.zeros(3)
        else:
            b_2d_unnormalized_dot = np.cross(
                b_3d_dot, ref.b_1d) + np.cross(self.b_3d, ref.b_1d_dot)
            b_2d_unnormalized_dot2 = np.cross(b_3d_dot2, ref.b_1d) + 2*np.cross(
                b_3d_dot, ref.b_1d_dot) + np.cross(self.b_3d, ref.b_1d_dot2)
            (b_2d, b_2d_dot, b_2d_dot2) = utils.get_unit_vector_derivatives(
                b_2d_unnormalized, b_2d_unnormalized_dot, b_2d_unnormalized_dot2)
        self.b_2d = b_2d
        b2d_x_b3d_unnormalized = np.cross(self.b_2d, self.b_3d)
        b2d_x_b3d_unnormalized_dot = np.cross(
            b_2d_dot, self.b_3d) + np.cross(self.b_2d, b_3d_dot)
        b2d_x_b3d_unnormalized_dot2 = np.cross(b_2d_dot2, self.b_3d) + 2*np.cross(
            b_2d_dot, b_3d_dot) + np.cross(self.b_2d, b_3d_dot2)
        (b2d_x_b3d, b2d_x_b3d_dot, b2d_x_b3d_dot2) = utils.get_unit_vector_derivatives(
            b2d_x_b3d_unnormalized, b2d_x_b3d_unnormalized_dot, b2d_x_b3d_unnormalized_dot2)

        self.pose_desired = np.array([b2d_x_b3d,
                                      self.b_2d,
                                      self.b_3d]).T
        self.pose_desired_dot = np.array([b2d_x_b3d_dot,
                                          b_2d_dot,
                                          b_3d_dot]).T
        self.pose_desired_dot2 = np.array([b2d_x_b3d_dot2,
                                           b_2d_dot2,
                                           b_3d_dot2]).T
        omega_desired_hat = self.pose_desired_dot@self.pose_desired.T
        self.omega_desired = utils.get_vee_map(omega_desired_hat)
        omega_desired_dot_hat = self.pose_desired_dot2@self.pose_desired.T + \
            self.pose_desired_dot@self.pose_desired_dot.T
        self.omega_desired_dot = utils.get_vee_map(omega_desired_dot_hat)

    def step_tracking_error(self, sensor_data: simulation.interface.SensorData, ref: trajectory.TrajectoryReference):
        self.e_x = sensor_data.position - ref.x_d
        self.e_v = sensor_data.v - ref.v_d

    def step_attitude_error(self, sensor_data: simulation.interface.SensorData):
        self.e_r = utils.get_vee_map(
            0.5*(self.pose_desired.T@sensor_data.pose - sensor_data.pose.T@self.pose_desired))
        self.e_omega = sensor_data.omega - sensor_data.pose.T@self.omega_desired

    def step_error_function_so3(self, sensor_data: simulation.interface.SensorData):
        '''The error function on SO(3) is defined as the angle between the desired and actual rotation matrix.
        debug purpose only, does not need to use in controller
        '''
        self.psi_r_rd = 0.5*(1 - self.pose_desired[:,0]@sensor_data.pose[:,0] +
                        1 - self.pose_desired[:,1]@sensor_data.pose[:,1] +
                        1 - self.pose_desired[:,2]@sensor_data.pose[:,2])
    
    def step_motor_output(self, sensor_data: simulation.interface.SensorData):
        self.force_motor_desired = self.params.m_wrench_to_thrust@np.hstack((self.f[2], self.torque))

        # debug
        print("check output force: ", self.force_motor_desired)
        # print(self.params.m_thrust_to_wrench)

        if self.is_using_inflow_model:
            # with inflow model
            for i, thrust in enumerate(self.force_motor_desired):
                self.rotation_speed[i] = self.propeller_force_table.get_rotation_speed(
                    sensor_data.rotors.rotors[i].local_wind_velocity,
                    sensor_data.rotors.rotors[i].velocity_inertial_frame,
                    sensor_data.rotors.rotors[i].pose,
                    sensor_data.rotors.rotors[i].rotation_speed,
                    thrust
                )
        else:
            # without inflow model
            for i, thrust in enumerate(self.force_motor_desired):
                self.rotation_speed[i] = self.propeller_force_table.get_rotation_speed(
                    np.zeros(3),
                    np.zeros(3),
                    sensor_data.rotors.rotors[i].pose,
                    sensor_data.rotors.rotors[i].rotation_speed,
                    thrust
                )
        print("desired rotor speed:", self.rotation_speed)

    def get_control_output(self):
        """controller provides yaw torque and rotation speed because rotor yaw torque is not modeled"""
        print("controller output torque: ", self.torque)
        print("controller output speed: ", self.rotation_speed)
        return simulation.interface.ControllerOutput(self.rotation_speed, np.array([0.0, 0.0, self.f[2]]), np.array([0.0, 0.0, self.torque[2]]))

