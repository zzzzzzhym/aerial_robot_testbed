import plotter
import numpy as np

import scenario_factory
import engine
import sim_logger
import drone.trajectory
import drone.propeller
import drone.disturbance_model
import drone.parameters

class Manager:
    """this module calls scenario maker, run engine, log and plot result"""
    def __init__(self) -> None:
        self.logger = sim_logger.Logger()
        self.scenario: scenario_factory.scenario = None
        self.engine: engine = None
        self.result = None

    def set_up(self, 
            #    trajectory=drone.trajectory.RandomWaypoints(300, True, True),    # when test training effect, we may want to use random subtrajectory set
            #    trajectory=drone.trajectory.RandomWaypointsInConstrainedSpace(200, False), 
            #    trajectory=drone.trajectory.RandomWaypoints(300, True), 
               trajectory= drone.trajectory.CircleYZ(),
            #    trajectory= drone.trajectory.Hover(),
            #    drone_params=drone.parameters.PennStateARILab550(), 
               drone_params=drone.parameters.P600(), 
               propeller_params=drone.propeller.apc_8x6, 
               disturbance_model=drone.disturbance_model.WallContact(),
               dt_dynamics=0.005) -> None:
        self.scenario = scenario_factory.Factory.make_scenario(
            trajectory, 
            drone_params, 
            propeller_params, 
            disturbance_model,
            dt_dynamics)
        self.engine = engine.Engine(self.scenario)        

    def run(self, t_end: float) -> None:
        self.engine.run_simulation(self.logger, t_end)
        self.logger.convert_buffer_to_output()
        self.result = plotter.Plotter(self.engine.t_span, self.engine.dt_log)

    def save_result_as_csv(self, file_name: str) -> None:
        self.logger.log_sim_result(file_name, 'csv')

    def save_result_as_pkl(self, file_name: str) -> None:
        self.logger.log_sim_result(file_name, 'pkl')

    def plot(self) -> None:
        self.result.make_plots(self.logger.output)
        