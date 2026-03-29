import numpy as np
import waypoint

import numpy as np

class TrajectoryConfig:
    def __init__(
        self,
        order_of_polynomial=7,
        initial_velocity=None,
        initial_acceleration=None,
        initial_jerk=None,
        terminal_velocity=None,
        terminal_acceleration=None,
        terminal_jerk=None,
    ):
        self.order_of_polynomial = order_of_polynomial

        self.initial_velocity = self.default_zeros(initial_velocity)
        self.initial_acceleration = self.default_zeros(initial_acceleration)
        self.initial_jerk = self.default_zeros(initial_jerk)

        self.terminal_velocity = self.default_zeros(terminal_velocity)
        self.terminal_acceleration = self.default_zeros(terminal_acceleration)
        self.terminal_jerk = self.default_zeros(terminal_jerk)

    @staticmethod
    def default_zeros(x):
        if x is None:
            return np.zeros(3, dtype=float)
        return x.copy()

if __name__ == "__main__":
    config = TrajectoryConfig()
    for attr, value in config.__dict__.items():
        print(f"{attr}: {value}")