import numpy as np
import sim_logger
import scenario
import log_adapter

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
        dynamics_output = self.scenario.dynamics.get_dynamics_output()
        self._append_log(logger, log_adapter.log_data_from_sensor(sensor_data))
        self._append_log(logger, log_adapter.log_data_from_dynamics(dynamics_output))
        self._append_log(logger, log_adapter.log_data_from_perception(
            self.scenario.dynamics.get_extended_world_perception()))

    def run_simulation(self, logger: sim_logger.Logger, t_end):
        self.t_span = np.arange(0.0, t_end + self.dt_log, self.dt_log)
        for t in self.t_span:
            self.step_simulation(t, logger)
            self.log_states(logger)
        self.shutdown()

    def shutdown(self):
        self.scenario.dynamics.shutdown()

    def _append_log(self, logger: sim_logger.Logger, data: dict):
        for key, val in data.items():
            logger.buffer[key].append(val)

    def log_states(self, logger: sim_logger.Logger):
        self._append_log(logger, self.scenario.controller.get_log_data())
        self._append_log(logger, self.scenario.trajectory.get_log_data())
        if hasattr(self.scenario.dynamics, 'disturbance'):
            self._append_log(logger, log_adapter.log_data_from_disturbance(self.scenario.dynamics.disturbance))
        if hasattr(self.scenario.dynamics, 'get_log_data'):
            self._append_log(logger, self.scenario.dynamics.get_log_data())




