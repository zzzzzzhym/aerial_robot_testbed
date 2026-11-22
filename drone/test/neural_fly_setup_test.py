import unittest
import numpy as np

import drone.parameters as params
import drone.trajectory as trajectory


class TestMockNeuralFlyByP600(unittest.TestCase):
    def setUp(self):
        self.p600 = params.P600()
        self.mock = params.MockNeuralFlyByP600()

    def test_thrust_to_weight_ratio(self):
        ratio = 4*self.mock.f_motor_max/(self.mock.m*params.Environment.g)
        self.assertAlmostEqual(ratio, 2.2, places=6)

    def test_mass(self):
        self.assertAlmostEqual(self.mock.m, 4.85, places=2)

    def test_geometry_inherited_from_p600(self):
        np.testing.assert_array_almost_equal(self.mock.inertia, self.p600.inertia)
        np.testing.assert_array_almost_equal(self.mock.m_thrust_to_wrench, self.p600.m_thrust_to_wrench)
        self.assertEqual(self.mock.is_ccw_blade, self.p600.is_ccw_blade)

    def test_hover_thrust_per_rotor_below_max(self):
        f_hover = self.mock.m*params.Environment.g/4
        self.assertLess(f_hover, self.mock.f_motor_max)


class TestNeuralFlyFigure8(unittest.TestCase):
    def test_position_matches_spec(self):
        traj = trajectory.NeuralFlyFigure8()
        for t in np.linspace(0, 10, 37):
            traj.step_reference_state(t)
            np.testing.assert_array_almost_equal(
                traj.x_d, [1.25 + 1.25*np.sin(t), 0.0, 0.75*np.sin(2*t)])
            np.testing.assert_array_almost_equal(
                traj.v_d, [1.25*np.cos(t), 0.0, 1.5*np.cos(2*t)])
            np.testing.assert_array_almost_equal(
                traj.x_d_dot2, [-1.25*np.sin(t), 0.0, -3.0*np.sin(2*t)])

    def test_derivatives_consistent_by_finite_difference(self):
        traj = trajectory.NeuralFlyFigure8()
        h = 1e-5
        for t in [0.0, 0.7, 2.1, 4.4]:
            traj.step_reference_state(t + h)
            plus = [traj.x_d, traj.v_d, traj.x_d_dot2, traj.x_d_dot3]
            traj.step_reference_state(t - h)
            minus = [traj.x_d, traj.v_d, traj.x_d_dot2, traj.x_d_dot3]
            traj.step_reference_state(t)
            derivs = [traj.v_d, traj.x_d_dot2, traj.x_d_dot3, traj.x_d_dot4]
            for p, m, d in zip(plus, minus, derivs):
                np.testing.assert_allclose((p - m)/(2*h), d, atol=1e-5)

    def test_limits_from_spec(self):
        traj = trajectory.NeuralFlyFigure8()
        t_span = np.linspace(0, traj.lap_period, 20001)
        pos, spd, acc = [], [], []
        for t in t_span:
            traj.step_reference_state(t)
            pos.append(traj.x_d.copy())
            spd.append(np.linalg.norm(traj.v_d))
            acc.append(np.linalg.norm(traj.x_d_dot2))
        pos = np.array(pos)
        self.assertAlmostEqual(pos[:, 0].min(), 0.0, places=4)
        self.assertAlmostEqual(pos[:, 0].max(), 2.5, places=4)
        self.assertAlmostEqual(pos[:, 2].max(), 0.75, places=4)
        self.assertAlmostEqual(max(spd), 1.953, places=2)
        self.assertAlmostEqual(max(acc), 3.14, places=1)

    def test_init_state_and_protocol_time(self):
        traj = trajectory.NeuralFlyFigure8()
        traj.set_init_state()
        np.testing.assert_array_almost_equal(traj.init_x, [1.25, 0.0, 0.0])
        np.testing.assert_array_almost_equal(traj.init_v, [1.25, 0.0, 1.5])
        self.assertAlmostEqual(traj.get_total_time(), 7*2*np.pi)
        self.assertAlmostEqual(traj.get_eval_start_time(), 2*np.pi)


if __name__ == '__main__':
    unittest.main()
