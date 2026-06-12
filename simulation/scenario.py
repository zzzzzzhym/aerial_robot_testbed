from abc import ABC, abstractmethod

import drone.trajectory
import simulation.interface as interface

class Scenario:
    def __init__(self):
        self.dynamics: Dynamics = None    # assemble drone type, disturbance model, dynamics model
        self.controller: Controller = None  # assemble drone type, controller and disturbance estimator
        self.trajectory: drone.trajectory.TrajectoryReference = None  
        self.sensor: Sensor = None

class Dynamics(ABC):
    @abstractmethod
    def step(self, t: float, controller_output: interface.ControllerOutput) -> None:
        pass
    
    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def get_dynamics_output(self):
        pass

    @abstractmethod
    def get_extended_world_perception(self):
        pass



class Controller(ABC):
    @abstractmethod
    def step(self, sensor_data: interface.SensorData, ref: interface.TrajectoryReference):
        pass

class Sensor(ABC):
    @abstractmethod
    def get_sensor_data(self, state: Dynamics, t: float) -> interface.SensorData:
        pass

class Planner(ABC):
    @abstractmethod
    def get_reference(self, t: float):
        pass
    
