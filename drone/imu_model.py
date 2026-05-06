import numpy as np

from drone import parameters as params


class BandedHarmonics:
    def __init__(self, f_min: float, f_max: float, num_of_samples: int, rms: float):
        self.num_of_samples = num_of_samples
        self.frequencies = np.linspace(f_min, f_max, num_of_samples)    # Hz
        self.amplitude = rms * np.sqrt(2) / np.sqrt(num_of_samples) # RMS^2 = sum(A_i^2)/2 where A_i is the amplitude of each sine wave
        rng = np.random.default_rng(0)
        self.phases = rng.uniform(0, 2*np.pi, size=num_of_samples)

    def get_signal(self, t: float):
        signal = 0.0
        for i in range(self.num_of_samples):
            signal += self.amplitude * np.sin(2*np.pi*self.frequencies[i]*t + self.phases[i])
        return signal
    

class ImuModel:
    """Using M-G365PDC1/PDF1 (https://www.horustech.com.tw/uploads/images/f5bc58f0-04c0-4599-b146-b708c08a5dca.pdf)"""
    def __init__(self):
        # bias
        self.gyro_turn_on_bias = 0.1*np.pi/180.0    # rad/s
        self.accel_turn_on_bias = 3*0.001*params.Environment.g  # m/s2
        self.omega_bias = np.array([1.0, -1.0, 1.0])*self.gyro_turn_on_bias
        self.accel_bias = np.array([1.0, -1.0, 1.0])*self.accel_turn_on_bias

        # noise
        self.gyro_noise_density = 0.002*np.pi/180.0 # rad/s/sqrt(Hz)
        self.accel_noise_density = 48*1e-6*params.Environment.g # m/s2/sqrt(Hz)
        self.gyro_sample_frequency = 100    # Hz
        self.accel_sample_frequency = 100   # Hz
        self.gyro_sigma = self.gyro_noise_density*np.sqrt(self.gyro_sample_frequency*0.5)
        self.accel_sigma = self.accel_noise_density*np.sqrt(self.accel_sample_frequency*0.5)

        # installation misalignment
        self.roll_misalignment = 1.0*np.pi/180.0    # rad
        self.pitch_misalignment = -1.0*np.pi/180.0    # rad
        self.yaw_misalignment = 1.0*np.pi/180.0    # rad
        self.misalignment_matrix = ImuModel.get_misalignment_matrix(
            self.roll_misalignment, self.pitch_misalignment, self.yaw_misalignment
        )

        # vibration induced harmonics
        f_min = 10.0  # Hz
        f_max = 15.0 # Hz
        num_of_samples = 100
        gyro_rms = 0.005    # rad/s
        a_z_rms = 0.5     # m/s2
        a_x_rms = 0.25    # m/s2
        a_y_rms = 0.25    # m/s2
        self.gyro_harmonics = BandedHarmonics(f_min, f_max, num_of_samples, gyro_rms)
        self.accel_x_harmonics = BandedHarmonics(f_min, f_max, num_of_samples, a_x_rms)
        self.accel_y_harmonics = BandedHarmonics(f_min, f_max, num_of_samples, a_y_rms)
        self.accel_z_harmonics = BandedHarmonics(f_min, f_max, num_of_samples, a_z_rms)

    @staticmethod
    def get_misalignment_matrix(roll_misalignment, pitch_misalignment, yaw_misalignment):
        """IMU_perceived_vector = misalignment_matrix @ ground_truth_vector"""
        roll_matrix = ImuModel.get_roll_rotation_matrix(roll_misalignment)
        pitch_matrix = ImuModel.get_pitch_rotation_matrix(pitch_misalignment)
        yaw_matrix = ImuModel.get_yaw_rotation_matrix(yaw_misalignment)
        return roll_matrix.T @ pitch_matrix.T @ yaw_matrix.T

    @staticmethod
    def get_roll_rotation_matrix(roll):
        """column of the matrix is the body frame basis vectors of the misaligned IMU"""
        return np.array([
            [1, 0,            0           ],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

    @staticmethod
    def get_pitch_rotation_matrix(pitch):
        """column of the matrix is the body frame basis vectors of the misaligned IMU"""
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch),0, np.cos(pitch)]
        ])

    @staticmethod
    def get_yaw_rotation_matrix(yaw):
        """column of the matrix is the body frame basis vectors of the misaligned IMU"""
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

    def inject_bias(self, ground_truth: np.ndarray, bias: np.ndarray):
        return (ground_truth + bias)

    def inject_white_noise(self, ground_truth: np.ndarray, sigma: float):
        return (ground_truth + np.random.normal(0, sigma, size=ground_truth.shape))

    def inject_installation_misalignment(self, ground_truth_vector: np.array):
        return self.misalignment_matrix @ ground_truth_vector

    def create_noisified_omega(self, omega: np.ndarray, t: float):
        noisidifed_omega = self.inject_installation_misalignment(omega)
        noisidifed_omega = self.inject_bias(noisidifed_omega, self.omega_bias)
        noisidifed_omega = self.inject_white_noise(noisidifed_omega, self.gyro_sigma)
        noisidifed_omega += self.gyro_harmonics.get_signal(t)
        return noisidifed_omega

    def create_noisified_accel(self, accel: np.ndarray, t: float):
        noisidifed_accel = self.inject_installation_misalignment(accel)
        noisidifed_accel = self.inject_bias(noisidifed_accel, self.accel_bias)
        noisidifed_accel = self.inject_white_noise(noisidifed_accel, self.accel_sigma)
        harmonics = np.array([
            self.accel_x_harmonics.get_signal(t),
            self.accel_y_harmonics.get_signal(t),
            self.accel_z_harmonics.get_signal(t)
        ])
        noisidifed_accel += harmonics
        return noisidifed_accel
    



