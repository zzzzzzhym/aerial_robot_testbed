import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


# Stub all external dependencies before importing main_process
_stub_module("arcpy", Controller=MagicMock)
_sim_iface = _stub_module("simulation.interface",
                           DynamicsOutput=MagicMock,
                           ExtendedWorldPerception=MagicMock,
                           ControllerOutput=MagicMock)
_sim_scenario = _stub_module("simulation.scenario", Dynamics=object)
_stub_module("simulation", interface=_sim_iface, scenario=_sim_scenario)
_stub_module("drone.utils", convert_quaternion_to_rotation_matrix=MagicMock(return_value=np.eye(3)))
_stub_module("drone.propeller", Propeller=MagicMock)
_stub_module("drone.parameters", Multicopter=MagicMock)
_stub_module("drone.dynamics_state", State=MagicMock)
_stub_module("drone.disturbance_model", WindEffectNearWall=MagicMock)
_stub_module("drone.rotor", RotorSet=MagicMock)
_drone = _stub_module("drone")
for _attr in ("parameters", "propeller", "dynamics_state", "disturbance_model", "rotor", "utils"):
    setattr(_drone, _attr, sys.modules[f"drone.{_attr}"])
_stub_module("external_sim.cfd_wind_field_lookup.vtk_reader")
_stub_module("external_sim.cfd_wind_field_lookup")

import external_sim.main_process as _mp  # noqa: E402

# Unbound references to the methods under test — works regardless of how P600 was resolved
_get_rotor_forces_body_frame = _mp.P600.get_rotor_forces_body_frame
_save_dynamics_state = _mp.P600.save_dynamics_state


def _make_self(extra_attrs=None):
    """Minimal namespace that stands in for `self` when calling unbound P600 methods."""
    import types
    ns = types.SimpleNamespace(
        rotor_force_getters=[object(), object(), object(), object()],
    )
    if extra_attrs:
        for k, v in extra_attrs.items():
            setattr(ns, k, v)
    return ns


class TestGetRotorForcesBodyFrame(unittest.TestCase):

    def test_returns_3d_slice_per_getter(self):
        self_ = _make_self()
        full_output = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        reply = MagicMock()
        reply.get_output_of.return_value = full_output

        forces = _get_rotor_forces_body_frame(self_, reply)

        self.assertEqual(len(forces), 4)
        for f in forces:
            np.testing.assert_array_equal(f, [1.0, 2.0, 3.0])

    def test_each_getter_queried_once(self):
        self_ = _make_self()
        reply = MagicMock()
        reply.get_output_of.return_value = [0.0, 0.0, 5.0]

        _get_rotor_forces_body_frame(self_, reply)

        self.assertEqual(reply.get_output_of.call_count, 4)
        for getter, call in zip(self_.rotor_force_getters, reply.get_output_of.call_args_list):
            self.assertIs(call.args[0], getter)


class TestSaveDynamicsStateRotorForce(unittest.TestCase):

    def _make_body_state(self):
        # 19-element body state: pos(0:3) quat(3:7) vel(7:10) omega(10:13) acc(13:16) alpha(16:19)
        data = np.zeros(19)
        data[3] = 1.0  # quaternion w=1 (identity)
        return data

    def test_set_force_from_body_frame_called_for_each_rotor(self):
        mock_rotor_set = MagicMock()
        mock_rotor_set.rotors = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        self_ = _make_self({
            "rotors": mock_rotor_set,
            "background_wind_reader": MagicMock(**{"get_velocity_at.return_value": np.zeros(3)}),
        })

        body_state_data = self._make_body_state()
        flow_speeds = [np.zeros(3)] * 4
        rotation_speed = np.zeros(4)
        f_bodies = [np.array([0.1, 0.2, float(i)]) for i in range(4)]

        _save_dynamics_state(self_, body_state_data, flow_speeds, rotation_speed, f_bodies)

        for rotor, f_body in zip(mock_rotor_set.rotors, f_bodies):
            rotor.set_force_from_body_frame.assert_called_once()
            np.testing.assert_array_equal(
                rotor.set_force_from_body_frame.call_args.args[0], f_body
            )


if __name__ == "__main__":
    unittest.main()
