import numpy as np
import matplotlib.pyplot as plt
import torch

from learning import model

class UnitDisturbance:
    """disturbance in a single dimension
    According to the paper, "We found that the three components of the wind-effect force, 
    fx fy fz, are highly correlated and sharing common features, 
    so we use theta as the basis function for all the component. ", this class can be applied to fx fy fz. 
    Irrationally, we're also using it to estimate torque—even though torque isn't part of the network's training.
    """
    def __init__(self, num_of_kernels, dt: float, mean=0.0, scale=1.0) -> None:
        self.disturbance = 0
        self.a_hat = np.zeros([num_of_kernels, 1])   # estimated a matrix
        self.p_cov = np.eye(num_of_kernels)     # covariance matrix, p_cov.shape = dim_of_a_hat*dim_of_a_hat
        self.lambda_reg = np.eye(num_of_kernels)*0.01
        self.dt = dt
        self.r = np.eye(1)*20  # measurement noise, r.shape = (measured disturbance dimension)^2
        self.q = np.eye(num_of_kernels)    # process noise, q.shape = p_cov.shape   
        self.scale = scale # normalization factor for the model labels, (raw - mean)*scale = normalized
        self.mean = mean # normalization factor for the model labels, (raw - mean)*scale = normalized
        if self.scale < 1e-6:
            raise ValueError("scale should be larger than 1e-6")

    @staticmethod
    def update_adaptation(a_hat: np.ndarray, 
                            p: np.ndarray, 
                            phi_vec: np.ndarray, 
                            y_vec: float, 
                            s_vec: float, 
                            r: np.ndarray, 
                            q: np.ndarray, 
                            lambda_val: np.ndarray, 
                            dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Updates the linear parameter estimate (a_hat) and the covariance matrix (P)
        based on the composite adaptation law.

        Args:
            a_hat (numpy.ndarray): 1D. Current estimate of the linear parameter vector.
            p (numpy.ndarray): 2D. Current covariance matrix for the parameter estimate.
            phi (numpy.ndarray): 1D. Learned basis function representing the aerodynamic effect.
            y (float): Measured aerodynamic residual force.
            s (float): Composite tracking error, defined as s = dot(q) - dot(q_r).
            r (numpy.ndarray): 2D. Positive definite gain matrix for prediction error weighting.
            q (numpy.ndarray): 2D. Positive definite gain matrix for covariance update.
            lambda_val (np.ndarray): 2D. Damping gain for regularization.
            dt (float): Time step size for discretization.

        Returns:
            tuple:
                a_hat_new (numpy.ndarray): Updated linear parameter estimate.
                P_new (numpy.ndarray): Updated covariance matrix.

        Notes:
            - The method implements Equations 8 and 9 from the composite adaptation law.
            - Equation 8 updates the parameter estimate (a_hat) based on the prediction error,
            tracking error (s), and a regularization term.
            - Equation 9 updates the covariance matrix (P), accounting for process noise (Q)
            and measurement noise (R).
        """        
        # Equation 8 - Update for a_hat
        phi = np.atleast_2d(phi_vec)    # treat this as C matrix in y = Cx + D, do not need to transpose
        s = np.atleast_2d(s_vec).T
        y = np.atleast_2d(y_vec).T
        prediction_error = phi @ a_hat - y

        # The continuous time version has a stability issue:
        # a_hat_dot = -lambda_val @ a_hat - p @ phi.T @ np.linalg.inv(r) @ prediction_error + p @ phi.T @ s
        # a_hat_new = a_hat + a_hat_dot * dt  # source of instability

        # # Equation 9 - Update for P
        # p_dot = -2 * lambda_val * p + q - p @ phi.T @ np.linalg.inv(r) @ phi @ p
        # p_new = p + p_dot * dt  # source of instability

        # use discrete Kalman filter to update a_hat and p
        s_covaraince = phi @ p @ phi.T + r  # Innovation covariance (the covariance of prediction_error, not the tracking error)
        k = p @ phi.T @ np.linalg.inv(s_covaraince)

        # Update a_hat
        prediction_error = y - phi @ a_hat
        a_hat_new = a_hat - lambda_val @ a_hat * dt + k @ prediction_error + p @ phi.T @ s * dt

        # Update p (Joseph form for stability)
        i_p = np.eye(p.shape[0])    # identity matrix of the same size as p
        p_new = (i_p - k @ phi) @ p @ (i_p - k @ phi).T + k @ r @ k.T + q * dt  # add Q over dt

        return a_hat_new, p_new     

    def step_estimator(self, kernel: np.ndarray, measured_disturbance: np.ndarray, tracking_error: np.ndarray) -> None:
        """
        Updates the disturbance estimator with the given kernel, measured disturbance, and tracking error.

        Args:
            kernel (np.ndarray): Kernel output.
            measured_disturbance (np.ndarray): The measured aerodynamic residual force.
            tracking_error (np.ndarray): The composite tracking error, defined as s = dot(q) - dot(q_r).
        """
        self.a_hat, self.p_cov = self.update_adaptation(self.a_hat, self.p_cov, kernel, self.normalize(measured_disturbance), tracking_error, self.r, self.q, self.lambda_reg, self.dt)
        self.disturbance = self.get_disturbance(kernel)

    def get_disturbance(self, kernel: np.ndarray):
        disturbance = np.atleast_2d(kernel)@self.a_hat*(1/self.scale) + self.mean
        disturbance = self.denormalize(np.atleast_2d(kernel)@self.a_hat)
        return disturbance[0, 0]
    
    def normalize(self, original: np.ndarray) -> np.ndarray:
        """normalize the disturbance"""
        return (original - self.mean) * self.scale
    
    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """denormalize the output to get disturbance"""
        return normalized / self.scale + self.mean

class DisturbanceEstimator:
    def __init__(self, model_name: str, dt: float) -> None:
        self.phi, h = model.load_daiml_model(model_name)
        self.phi.eval()
        self.num_of_kernals = self.phi.dim_of_output
        self.dof_of_disturbance = 6
        self.dt = dt
        self.f_x = UnitDisturbance(self.num_of_kernals, self.dt, self.phi.output_mean[0].numpy(), self.phi.output_scale[0].numpy())
        self.f_y = UnitDisturbance(self.num_of_kernals, self.dt, self.phi.output_mean[1].numpy(), self.phi.output_scale[1].numpy())
        self.f_z = UnitDisturbance(self.num_of_kernals, self.dt, self.phi.output_mean[2].numpy(), self.phi.output_scale[2].numpy())
        self.tq_x = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_y = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_z = UnitDisturbance(self.num_of_kernals, self.dt)

    def update_kernel(self, position: np.ndarray, v: np.ndarray, q: np.ndarray, omega: np.ndarray, f: np.ndarray, tq: np.ndarray, rotor_spd: np.ndarray) -> np.ndarray:
        nn_input = np.hstack([position, q, v, omega, f, tq, rotor_spd])
        with torch.no_grad():
            kernel = self.phi(torch.from_numpy(nn_input))
        return kernel.numpy()
    
    def step_disturbance(self, position: np.array, v: np.ndarray, q: np.ndarray, omega: np.ndarray, f: np.ndarray, tq: np.ndarray, rotor_spd: np.ndarray, measured_disturbance: np.ndarray, tracking_error: np.ndarray) -> None:
        self.kernel = self.update_kernel(position, v, q, omega, f, tq, rotor_spd)
        self.f_x.step_estimator(self.kernel, measured_disturbance[0], tracking_error[0])
        self.f_y.step_estimator(self.kernel, measured_disturbance[1], tracking_error[1])
        self.f_z.step_estimator(self.kernel, measured_disturbance[2], tracking_error[2])
        self.tq_x.step_estimator(self.kernel, measured_disturbance[3], tracking_error[3])
        self.tq_y.step_estimator(self.kernel, measured_disturbance[4], tracking_error[4])
        self.tq_z.step_estimator(self.kernel, measured_disturbance[5], tracking_error[5])

    def get_disturbance_force(self) -> np.ndarray:
        return np.array([self.f_x.disturbance, self.f_y.disturbance, self.f_z.disturbance])
    
    def get_disturbance_torque(self) -> np.ndarray:
        return np.array([self.tq_x.disturbance, self.tq_y.disturbance, self.tq_z.disturbance])


class BemtFittedDisturbanceEstimatorV0(DisturbanceEstimator):
    def __init__(self, model_name: str, dt: float) -> None:
        self.nn = model.load_daiml_model(model_name)
        self.nn.eval()
        self.num_of_kernals = 2
        self.dof_of_disturbance = 6
        self.dt = dt
        self.f_x = UnitDisturbance(self.num_of_kernals, self.dt, self.nn.output_mean[0].numpy(), self.nn.output_scale[0].numpy())
        self.f_y = UnitDisturbance(self.num_of_kernals, self.dt, self.nn.output_mean[1].numpy(), self.nn.output_scale[1].numpy())
        self.f_z = UnitDisturbance(self.num_of_kernals, self.dt, self.nn.output_mean[2].numpy(), self.nn.output_scale[2].numpy())
        self.tq_x = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_y = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_z = UnitDisturbance(self.num_of_kernals, self.dt)

    def update_kernel(
        self,
        v: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray,
        rotor_0_local_wind_velocity: np.ndarray,
        rotor_1_local_wind_velocity: np.ndarray,
        rotor_2_local_wind_velocity: np.ndarray,
        rotor_3_local_wind_velocity: np.ndarray,
        rotor_spd: np.ndarray
    ) -> np.ndarray:
        nn_input = np.hstack([
            q,
            v,
            omega,
            rotor_0_local_wind_velocity,
            rotor_1_local_wind_velocity,
            rotor_2_local_wind_velocity,
            rotor_3_local_wind_velocity,
            rotor_spd
        ])
        with torch.no_grad():
            kernel = self.nn(torch.from_numpy(nn_input))
        return np.hstack([kernel.numpy(), 1.0])  # add bias term to capture any unmodeled disturbance
    

# make another baseline disturbance estimator that don't take in the kernel
class BaselineDisturbanceEstimator:
    """Baseline disturbance estimator: equivalent to using a kernel of 1, the a_hat will be the disturbance.
    This is equivalent to Neural-Fly constant example in the paper. This is a pure adaptive disturbance estimator that takes control error as estimation input and assumes disturbance is constant.
    """
    def __init__(self, dt: float) -> None:
        self.dof_of_disturbance = 6
        self.dt = dt
        # equivalent to using a kernel of 1, the a_hat will be the disturbance
        self.f_x = UnitDisturbance(1, self.dt)
        self.f_y = UnitDisturbance(1, self.dt)
        self.f_z = UnitDisturbance(1, self.dt)
        self.tq_x = UnitDisturbance(1, self.dt)
        self.tq_y = UnitDisturbance(1, self.dt)
        self.tq_z = UnitDisturbance(1, self.dt)

    def step_disturbance(self, measured_disturbance: np.ndarray, tracking_error: np.ndarray) -> None:
        self.f_x.step_estimator(np.array([[1]]), measured_disturbance[0], tracking_error[0])
        self.f_y.step_estimator(np.array([[1]]), measured_disturbance[1], tracking_error[1])
        self.f_z.step_estimator(np.array([[1]]), measured_disturbance[2], tracking_error[2])
        self.tq_x.step_estimator(np.array([[1]]), measured_disturbance[3], tracking_error[3])
        self.tq_y.step_estimator(np.array([[1]]), measured_disturbance[4], tracking_error[4])
        self.tq_z.step_estimator(np.array([[1]]), measured_disturbance[5], tracking_error[5])

    def get_disturbance_force(self) -> np.ndarray:
        return np.array([self.f_x.disturbance, self.f_y.disturbance, self.f_z.disturbance])
    
    def get_disturbance_torque(self) -> np.ndarray:
        return np.array([self.tq_x.disturbance, self.tq_y.disturbance, self.tq_z.disturbance])

    
class BemtFittedDisturbanceEstimatorV1(BaselineDisturbanceEstimator):
    def __init__(self, model_name: str, dt: float) -> None:
        self.nn = model.load_simple_model(model_name)   # TODO: make abstract factory to select the right model factory
        self.nn.eval()
        self.dof_of_disturbance = 6
        self.dt = dt
        self.f_x = UnitDisturbance(1, self.dt)
        self.f_y = UnitDisturbance(1, self.dt)
        self.f_z = UnitDisturbance(1, self.dt)
        self.tq_x = UnitDisturbance(1, self.dt)
        self.tq_y = UnitDisturbance(1, self.dt)
        self.tq_z = UnitDisturbance(1, self.dt)

        # tune down measurement noise to suppress noising simplenet output
        self.f_x.r = self.f_x.r*0.1
        self.f_y.r = self.f_y.r*0.1
        self.f_z.r = self.f_z.r*0.1
        self.tq_x.r = self.tq_x.r*0.1
        self.tq_y.r = self.tq_y.r*0.1
        self.tq_z.r = self.tq_z.r*0.1
        self.predicted_disturbance = np.zeros(3)

    def predict_disturbance(
        self,
        v: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray,
        rotor_0_local_wind_velocity: np.ndarray,
        rotor_1_local_wind_velocity: np.ndarray,
        rotor_2_local_wind_velocity: np.ndarray,
        rotor_3_local_wind_velocity: np.ndarray,
        rotor_spd: np.ndarray,
        bemt_predicted_force: np.ndarray,
        bemt_predicted_torque: np.ndarray
    ) -> np.ndarray:
        nn_input = np.hstack([
            q,
            v,
            omega,
            rotor_0_local_wind_velocity,
            rotor_1_local_wind_velocity,
            rotor_2_local_wind_velocity,
            rotor_3_local_wind_velocity,
            rotor_spd
        ])
        with torch.no_grad():
            prediction = self.nn(torch.from_numpy(nn_input))
        return np.hstack([prediction.numpy() + bemt_predicted_force, bemt_predicted_torque])

    def step_disturbance(
        self,
        v: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray,
        rotor_0_local_wind_velocity: np.ndarray,
        rotor_1_local_wind_velocity: np.ndarray,
        rotor_2_local_wind_velocity: np.ndarray,
        rotor_3_local_wind_velocity: np.ndarray,
        rotor_spd: np.ndarray,
        measured_disturbance: np.ndarray,
        tracking_error: np.ndarray,
        bemt_predicted_force: np.ndarray,
        bemt_predicted_torque: np.ndarray
    ) -> None:
        self.predicted_disturbance = self.predict_disturbance(
            v,
            q,
            omega,
            rotor_0_local_wind_velocity,
            rotor_1_local_wind_velocity,
            rotor_2_local_wind_velocity,
            rotor_3_local_wind_velocity,
            rotor_spd,
            bemt_predicted_force,
            bemt_predicted_torque
        )
        self.f_x.step_estimator(np.array([[1]]), measured_disturbance[0] - self.predicted_disturbance[0], tracking_error[0])
        self.f_y.step_estimator(np.array([[1]]), measured_disturbance[1] - self.predicted_disturbance[1], tracking_error[1])
        self.f_z.step_estimator(np.array([[1]]), measured_disturbance[2] - self.predicted_disturbance[2], tracking_error[2])
        self.tq_x.step_estimator(np.array([[1]]), measured_disturbance[3] - self.predicted_disturbance[3], tracking_error[3])
        self.tq_y.step_estimator(np.array([[1]]), measured_disturbance[4] - self.predicted_disturbance[4], tracking_error[4])
        self.tq_z.step_estimator(np.array([[1]]), measured_disturbance[5] - self.predicted_disturbance[5], tracking_error[5])

    def get_disturbance_force(self) -> np.ndarray:
        return np.array([self.f_x.disturbance, self.f_y.disturbance, self.f_z.disturbance]) + self.predicted_disturbance[0:3]
    
    def get_disturbance_torque(self) -> np.ndarray:
        return np.array([self.tq_x.disturbance, self.tq_y.disturbance, self.tq_z.disturbance]) + self.predicted_disturbance[3:6]

if __name__ == "__main__":
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    f = np.array([1.0, 1.0, 1.0, 1.0])
    f_x = []
    f_y = []
    f_z = []
    tq_x = []
    tq_y = []
    tq_z = []
    t = np.arange(0, 10, 0.01)
    # dist = DisturbanceEstimator("test", 0.01)
    # for _ in t:
    #     dist.step_disturbance(v, q, f, y, s)
    #     f_x.append(dist.f_x.disturbance)
    #     f_y.append(dist.f_y.disturbance)
    #     f_z.append(dist.f_z.disturbance)
    #     tq_x.append(dist.tq_x.disturbance)
    #     tq_y.append(dist.tq_y.disturbance)
    #     tq_z.append(dist.tq_z.disturbance)
    dist = BaselineDisturbanceEstimator(0.01)
    for _ in t:
        dist.step_disturbance(y, s)
        f_x.append(dist.f_x.disturbance)
        f_y.append(dist.f_y.disturbance)
        f_z.append(dist.f_z.disturbance)
        tq_x.append(dist.tq_x.disturbance)
        tq_y.append(dist.tq_y.disturbance)
        tq_z.append(dist.tq_z.disturbance)

    fig, axs = plt.subplots(6,1,sharex=True)
    axs[0].plot(t, f_x)
    axs[1].plot(t, f_y)
    axs[2].plot(t, f_z)
    axs[3].plot(t, tq_x)
    axs[4].plot(t, tq_y)
    axs[5].plot(t, tq_z)
    plt.show()
    

