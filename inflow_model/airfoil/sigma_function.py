import numpy as np

class SigmaFunction:
    def __init__(self, alpha_0=np.radians(25), M=25):
        """
        Initialize the SigmaFunction with parameters and caps.

        Parameters:
        - alpha0: Stall angle (radians)
        - M: Sharpness parameter
        - alpha_cap: Tuple of (min_alpha, max_alpha) for capping alpha values
        - M_cap: Tuple of (min_M, max_M) for capping M values
        """
        self.alpha_0 = np.clip(alpha_0, -30, 30)
        self.M = np.clip(M, 10, 50)
        self.alpha_cap = np.pi/2

    def compute(self, alpha):
        """
        Compute the sigma function for a given angle of attack (alpha).

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - Sigma value (float)
        """

        alpha = np.clip(alpha, -self.alpha_cap, self.alpha_cap)
        # Compute sigma
        exp1 = np.exp(-self.M * (alpha - self.alpha_0))
        exp2 = np.exp(self.M * (alpha + self.alpha_0))
        numerator = 1 + exp1 + exp2
        denominator = (1 + exp1) * (1 + exp2)
        return numerator / denominator

