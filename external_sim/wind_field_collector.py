import arcpy as ar  # external simulation API
import numpy as np
from pathlib import Path

import simulation.interface
import simulation.scenario
import drone.propeller
import drone.parameters
import drone.dynamics_state
import drone.disturbance_model
import drone.rotor

import external_sim.cfd_wind_field_lookup.vtk_reader as vtk_reader

class WindFieldCollector(simulation.scenario.Dynamics):
    """
    Use the same config (art) as the drone task. 
    But disable the drone and just take data from the background wind field.
    Everything setter / reader under drone dir should be removed. 
    """
    def __init__(
        self,
        drone_params: drone.parameters.Multicopter,
        propeller: drone.propeller.Propeller,
        disturbance: drone.disturbance_model.WindEffectNearWall,
        init_state: drone.dynamics_state.State,
        dt: float = 0.01,
    ):
        print("-----------WindFieldCollector init-----------")
        self.t_end = 10.0
        self.dt = dt

        self.main_api = ar.Controller()

        wind_speed = disturbance.u_free
        print("wind_speed: ", wind_speed)

        wind_folder = vtk_reader.get_wind_velocity_folder_name(wind_speed)
        global_id = 0  # ID=0 corresponds to 'Global'
        self.main_api.set_object_property_string(global_id, 'export.folder_name', str(Path("export") / wind_folder)) 
        self.main_api.set_object_property_uint(global_id, 'export.auto_export', 1) 
        self.main_api.set_object_property_float(global_id, 'export.auto_export_target_fps', 1) 
        self.main_api.set_object_property_float(global_id, 'export.auto_export_start_time', 3) 


        drone_body = self.main_api.get_object('p600')
        self.main_api.set_object_property_uint(drone_body, 'enabled', 0)    # disable the drone body for a clean wind field

        block_fluid = self.main_api.get_system_by_name('Blocks Fluid')
        self.main_api.set_object_property_float(block_fluid, 'fluid.initial_velocity.x', wind_speed[0])
        self.main_api.set_object_property_float(block_fluid, 'fluid.initial_velocity.y', wind_speed[1])
        self.main_api.set_object_property_float(block_fluid, 'fluid.initial_velocity.z', wind_speed[2])

        wall = self.main_api.get_object('floor/tracker')
        self.main_api.set_object_property_float(wall, 'plane_origin.x', disturbance.wind_field_model.wall_origin[0])

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

    def get_dynamics_output(self) -> simulation.interface.DynamicsOutput:
        return self.state

    def get_extended_world_perception(self) -> simulation.interface.ExtendedWorldPerception:
        return self.contact_force

    def step(self, t, command: simulation.interface.ControllerOutput) -> simulation.interface.DynamicsOutput:

        world_intput = {}

        print("-----------WindFieldCollector-----------")
        print(self.i)
        print("t=", t + self.dt) 
        reply = self.main_api.simulate_until(
            t + self.dt,  
            world_intput, 
            [])

        if reply.is_failed():
            raise RuntimeError('Simulation failed!')
        else:
            # flow_field = self.main_api.export(self.i)    # vtk for paraview will be saved in the same path to art file
            self.i += 1 # todo: convert to time-based index and better in string format

    def shutdown(self):
        self.main_api.clear()
        self.main_api.close()
        print("Done.")
