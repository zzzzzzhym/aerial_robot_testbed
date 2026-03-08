import numpy as np

import sigma_function
from air import Air



class Coeffecients:
    def __init__(self, cl_1=5.3, cl_2=1.7, alpha_0=np.radians(20.6), cd=1.8, cd_0=0.01, cp=1.328):
        """
        Initialize the Coeffecients with parameters.

        Parameters:
        - cl_1: Lift coefficient parameter
        - cl_2: Lift coefficient parameter
        - cd: Drag coefficient parameter
        - alpha0: Stall angle in radians
        """
        self.cl_1 = cl_1
        self.cl_2 = cl_2
        self.alpha_0 = alpha_0
        self.cd = cd
        self.cd_0 = cd_0
        self.cp = cp
        self.sigma = sigma_function.SigmaFunction(alpha_0=self.alpha_0)


    def get_cl(self, alpha):
        """
        Compute the lift coefficient for a given angle of attack (alpha).

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - CL value (float)
        """
        CL = (1 - self.sigma.compute(alpha))*self.cl_1 * alpha + self.sigma.compute(alpha)*self.cl_2 * np.sin(alpha) * np.cos(alpha)

        return CL
    
    def get_cd(self, alpha, u, chord):
        """
        Compute the drag coefficient for a given angle of attack (alpha).

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - CD value (float)
        """
        rn_clamped = np.maximum(Coeffecients.get_reynolds_number(u, chord), 1)
        CD = self.cd * np.sin(alpha)**2 + 2*1.02*self.cp / np.sqrt(rn_clamped) + self.cd_0
        return CD
    
    @staticmethod
    def get_reynolds_number(u, chord):
        """
        Compute the Reynolds number for a given velocity and chord length.

        Parameters:
        - U: Velocity (m/s)
        - chord: Chord length (m)

        Returns:
        - Reynolds number (float)
        """
        rn = Air.rho * u * chord / Air.mu
        return rn
    

class NeuroBEMCoefficients:
    """
    Coefficient model used in the NeuroBEM MATLAB BEM code (ParameterID.m).

    Nonlinear (final) mode:
        cl(alpha) = a * sin(alpha) * cos(alpha)
        cd(alpha) = d * sin(alpha)^2

    Linear (initialization) mode:
        cl(alpha) = a * alpha
        cd(alpha) = d   (constant)

    Notes:
    - alpha is in radians.
    - u and chord are accepted for interface compatibility but not used.
    """

    def __init__(self, a=15.20569, d=13.53063, linear_mode=False):
        self.a = a
        self.d = d
        self.linear_mode = linear_mode

    def get_cl(self, alpha):
        if self.linear_mode:
            return self.a * alpha
        return self.a * np.sin(alpha) * np.cos(alpha)

    def get_cd(self, alpha, u=None, chord=None):
        if self.linear_mode:
            # constant drag in linearMode branch of their MATLAB
            return np.full_like(alpha, self.d, dtype=float) if np.ndim(alpha) else float(self.d)
        return self.d * (np.sin(alpha) ** 2)