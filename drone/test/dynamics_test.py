import unittest
import numpy as np

import parameters 
import dynamics
import disturbance_model
import propeller
import dynamics_state as state

class TestDroneDynamics(unittest.TestCase):
    def setUp(self):
        self.drone = parameters.TrackingOnSE3()
        self.position = np.array([0.0, 0.0, 0.0])
        self.v = np.array([1.0, 2.0, 3.0])
        self.pose = np.eye(3)
        self.omega = np.array([np.pi/6, 0.0, 0.0])
        self.dt = 0.01  # Time step
        init_state = state.State(self.position, self.v, self.pose, self.omega)
        self.dynamics = dynamics.DroneDynamics(
            self.drone, propeller.apc_8x6, disturbance_model.Free(), init_state, self.dt
        )
        

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.dynamics.state.position, self.position))
        self.assertTrue(np.array_equal(self.dynamics.state.v, self.v))
        self.assertTrue(np.array_equal(self.dynamics.state.pose, self.pose))
        self.assertTrue(np.array_equal(self.dynamics.state.omega, self.omega))
        self.assertEqual(self.dynamics.dt, self.dt)

    def test_pack_state_vector(self):
        state_vector = self.dynamics.pack_state_vector()
        # Expected state vector based on the initial conditions
        expected_vector = np.array([
            self.position[0],  # x-coordinate of the position
            self.position[1],  # y-coordinate of the position
            self.position[2],  # z-coordinate of the position
            self.v[0],         # x-component of the velocity
            self.v[1],         # y-component of the velocity
            self.v[2],         # z-component of the velocity
            1.0,               # w-component of the quaternion (real part)
            0.0,               # x-component of the quaternion (imaginary part)
            0.0,               # y-component of the quaternion (imaginary part)
            0.0,               # z-component of the quaternion (imaginary part)
            self.omega[0],     # x-component of the angular velocity
            self.omega[1],     # y-component of the angular velocity
            self.omega[2],     # z-component of the angular velocity
            self.dynamics.disturbance.f_implicit[0],  # x-component of the disturbance force
            self.dynamics.disturbance.f_implicit[1],  # y-component of the disturbance force
            self.dynamics.disturbance.f_implicit[2],  # z-component of the disturbance force
            self.dynamics.disturbance.t_implicit[0],  # x-component of the disturbance torque
            self.dynamics.disturbance.t_implicit[1],  # y-component of the disturbance torque
            self.dynamics.disturbance.t_implicit[2]   # z-component of the disturbance torque
        ])
        np.testing.assert_almost_equal(state_vector, expected_vector, decimal=5)

    def test_unpack_state_vector(self):
        state_vector = self.dynamics.pack_state_vector()
        self.dynamics.unpack_state_vector(state_vector)
        self.assertTrue(np.array_equal(self.dynamics.state.position, self.position))
        self.assertTrue(np.array_equal(self.dynamics.state.v, self.v))
        self.assertTrue(np.array_equal(self.dynamics.state.omega, self.omega))

    def test_step_dynamics_0(self):
        """one step integration for 1sec"""
        dt = 1.0
        self.dynamics.dt = 1.0
        self.dynamics.step(0.0)
        pose_answer = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                                [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]])
        position_answer = np.array([self.v[0]*dt, self.v[1]*dt, self.v[2]*dt + 0.5*parameters.Environment.g*dt*dt])
        np.testing.assert_array_almost_equal(self.dynamics.state.pose, pose_answer)
        np.testing.assert_array_almost_equal(self.dynamics.state.position, position_answer)

    def test_step_dynamics_1(self):
        """fine integration for 1sec"""
        self.dynamics.dt = 0.01
        for i in range(100):
            self.dynamics.step(self.dt*i)
        pose_answer = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                                [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]])
        dt = 1.0
        position_answer = np.array([self.v[0]*dt, self.v[1]*dt, self.v[2]*dt + 0.5*parameters.Environment.g*dt*dt])
        np.testing.assert_array_almost_equal(self.dynamics.state.pose, pose_answer, decimal=5)
        np.testing.assert_array_almost_equal(self.dynamics.state.position, position_answer, decimal=5)

    def test_step_dynamics_2(self):
        """test torque"""
        self.dynamics.torque = np.array([1.0, 0.0, 0.0])
        self.dynamics.step(self.dt*100)
        expected_omega_dot = self.drone.inertia_inv@self.dynamics.torque
        np.testing.assert_array_almost_equal(self.dynamics.omega_dot, expected_omega_dot)


if __name__ == '__main__':
    unittest.main()
