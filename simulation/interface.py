import numpy as np
    
import drone.dynamics_state
import drone.rotor

class DynamicsOutput(drone.dynamics_state.State):
    """Define what output the dynamics module should provide to the controller and what it get from controller input
    serve as a data class"""
    def __init__(self, 
                 position: np.ndarray, 
                 v: np.ndarray,
                 pose: np.ndarray, 
                 omega: np.ndarray,
                 v_dot: np.ndarray,
                 rotors: drone.rotor.RotorSet,
                 omega_dot: np.ndarray,
                ) -> None:
        super().__init__(position, v, pose, omega)
        self.v_dot = v_dot  # in inertial frame
        self.omega_dot = omega_dot   # in body fix frame
        self.rotors = rotors

    def get_v_dot_in(self, frame: str):
        frame_id = self.select_frame(frame)
        if frame_id == self.Frame.BODY:
            return self.convert_to_body_frame(self.v_dot)
        return self.v_dot
    
    def get_omega_dot_in(self, frame: str):
        frame_id = self.select_frame(frame)
        if frame_id == self.Frame.BODY:
            return self.omega_dot
        return self.convert_to_inertial_frame(self.omega_dot)

class ExtendedWorldPerception:
    """Other none drone state related info from the world"""
    def __init__(self,
                 contact_force: np.ndarray,
                 tip_position: np.ndarray):
        self.contact_force = contact_force    # in body fix frame
        self.tip_position = tip_position

class ControllerOutput:
    """Interface between controll and the plant"""
    def __init__(self, rotation_speed: np.ndarray, f: np.ndarray, torque: np.ndarray) -> None:
        self.rotation_speed = rotation_speed
        self.f = f
        self.torque = torque

class SensorData(DynamicsOutput):
    """Container for all sensor data, follow FLU convention."""
    def __init__(self, 
                 position: np.ndarray, 
                 v: np.ndarray,
                 pose: np.ndarray, 
                 omega: np.ndarray,
                 v_dot: np.ndarray,
                 rotors: drone.rotor.RotorSet,
                 omega_dot: np.ndarray) -> None:
        
        # does not have difference with DynamicsOutput for now
        super().__init__(position, v, pose, omega, v_dot, rotors, omega_dot)

class TrajectoryReference:
    pass