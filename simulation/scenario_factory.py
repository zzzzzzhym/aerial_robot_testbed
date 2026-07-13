import drone.trajectory
import drone.dynamics
import drone.dynamics_state
import drone.controller
import drone.propeller
import drone.disturbance_model
import drone.parameters
import drone.sensor
import scenario


class Factory:
    @staticmethod
    def make_scenario(trajectory: drone.trajectory.TrajectoryReference, 
                      drone_params: drone.parameters.Multicopter, 
                      propeller_params: drone.propeller.Propeller, 
                      disturbance_model: drone.disturbance_model.DisturbanceForce,
                      dt_dynamics: float) -> scenario.Scenario:
        product = scenario.Scenario()
        trajectory.set_init_state()
        init_state = drone.dynamics_state.State(
            trajectory.init_x,
            trajectory.init_v,
            trajectory.init_pose,
            trajectory.init_omega)
        
        product.dynamics = Factory.make_dynamics_assembly(
            drone_params, 
            propeller_params, 
            disturbance_model, 
            init_state, 
            dt_dynamics)
        product.controller = Factory.make_controller_assembly(drone_params)
        product.trajectory = trajectory
        product.sensor = drone.sensor.Sensor()  # can pass from input to customize different sensor model in the future
        return product

    @staticmethod
    def make_dynamics_assembly(drone_params: drone.parameters.Multicopter, 
                               propeller_params: drone.propeller.Propeller, 
                               disturbance_model: drone.disturbance_model.DisturbanceForce, 
                               init_state: drone.dynamics_state.State, 
                               dt_dynamics: float) -> drone.dynamics.DroneDynamics:
        # assembly = drone.dynamics.DroneDynamics(
        #     drone_params,
        #     propeller_params,
        #     disturbance_model,
        #     init_state,
        #     dt_dynamics)

        import external_sim.main_process
        assembly = external_sim.main_process.P600(
            drone_params,
            propeller_params,
            disturbance_model,
            init_state,
            dt_dynamics)

        # import external_sim.wind_field_collector
        # assembly = external_sim.wind_field_collector.WindFieldCollector(
        #     drone_params,
        #     propeller_params,
        #     disturbance_model,
        #     init_state,
        #     dt_dynamics)
        return assembly
    
    @staticmethod
    def make_controller_assembly(drone_params: drone.parameters.Multicopter) -> drone.controller.DroneController:
        assembly = drone.controller.DroneController(drone_params)
        return assembly
    