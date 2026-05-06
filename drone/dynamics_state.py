import enum
import numpy as np

from drone import utils


class State:
    """This class manage the state of the drone. The default convention is in FLU frame.
    """
    class Frame(enum.Enum):
        INERTIAL = 0
        BODY = 1

    def __init__(self, position: np.ndarray=np.array([0.0, 0.0, 0.0]), 
                 v: np.ndarray=np.array([0.0, 0.0, 0.0]),
                 pose: np.ndarray=np.eye(3), 
                 omega: np.ndarray=np.array([0.0, 0.0, 0.0])) -> None:
        self.position = position    # in inertial frame
        self.pose = pose    # rotation matrix in inertial frame
        self.v = v          # in inertial frame
        self.omega = omega  # omega in body fix frame
        self.q = utils.convert_rotation_matrix_to_quaternion(self.pose)

    def convert_to_body_frame(self, input):
        return self.pose.T@input

    def convert_to_inertial_frame(self, input):
        return self.pose@input

    def select_frame(self, frame: str):
        if frame == "body":
            return self.Frame.BODY
        elif frame == "inertial":
            return self.Frame.INERTIAL
        else:
            raise ValueError(f"Invalid frame selection: {frame}, should be either 'body' or 'inertial'")

    def get_position_in(self, frame: str):
        frame_id = self.select_frame(frame)
        if frame_id == self.Frame.BODY:
            return np.zeros(3)
        return self.position
    
    def get_velocity_in(self, frame: str):
        frame_id = self.select_frame(frame)
        if frame_id == self.Frame.BODY:
            return self.convert_to_body_frame(self.v)
        return self.v
    
    def get_omega_in(self, frame: str):
        frame_id = self.select_frame(frame)
        if frame_id == self.Frame.BODY:
            return self.omega
        return self.convert_to_inertial_frame(self.omega)

    def get_pose_in(self, frame: str):
        frame_id = self.select_frame(frame)
        if frame_id == self.Frame.BODY:
            return np.eye(3)
        return self.pose
