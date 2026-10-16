from abc import ABC, abstractmethod

import numpy as np

import sigma_function
from air import Air


class AeroCoefficients(ABC):
    """Abstract interface for an airfoil's lift and drag coefficients.

    Blade Element Theory depends only on this interface (the get_cl and get_cd methods).
    Concrete coefficient models are implemented as subclasses and attached to a blade
    (see Blade.get_aero_coeff in blade_params).
    """

    @abstractmethod
    def get_cl(self, alpha):
        """Compute the lift coefficient.

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - Lift coefficient (float)
        """

    @abstractmethod
    def get_cd(self, alpha, u, chord):
        """Compute the drag coefficient.

        Parameters:
        - alpha: Angle of attack (radians)
        - u: Local flow speed (m/s)
        - chord: Chord length (m)

        Returns:
        - Drag coefficient (float)
        """

    @staticmethod
    def get_reynolds_number(u, chord):
        """Compute the Reynolds number.

        Parameters:
        - u: Flow speed (m/s)
        - chord: Chord length (m)

        Returns:
        - Reynolds number (float)
        """
        return Air.rho * u * chord / Air.mu

    @abstractmethod
    def to_dict(self):
        """Return the coefficient parameters as a plain, YAML-serializable dict.

        For reference/logging only (e.g. recorded in a lookup-table YAML); not used in
        the force computation.
        """

    def describe(self):
        """Human-readable multi-line listing of the coefficient parameters."""
        return "\n".join(f"{key} = {value}" for key, value in self.to_dict().items())


class ClassicalAeroCoefficients(AeroCoefficients):
    """Closed-form CL/CD profile (the original analytic model)."""

    def __init__(
        self,
        cl_1=5.3,
        cl_2=1.7,
        alpha_0=np.radians(20.6),
        cd=1.8,
        cd_0=0.01,
        cp=1.328,
        alpha_zero_lift=0.0,
        alpha_d_min=0.0,
    ):
        """
        Parameters:
        - cl_1, cl_2: lift coefficient parameters
        - cd: drag coefficient parameter
        - alpha_0: stall angle in radians
        - cd_0, cp: profile-drag constants
        - alpha_zero_lift: zero-lift angle of attack in radians
        - alpha_d_min: minimum-drag angle of attack in radians
        """
        self.cl_1 = cl_1
        self.cl_2 = cl_2
        self.alpha_0 = alpha_0
        self.cd = cd
        self.cd_0 = cd_0
        self.cp = cp
        self.alpha_zero_lift = alpha_zero_lift
        self.alpha_d_min = alpha_d_min
        self.sigma = sigma_function.SigmaFunction(alpha_0=self.alpha_0)

    def update_fit_params(self, cl_1, cl_2, cd, alpha_0, alpha_zero_lift=None, alpha_d_min=None):
        """Update the fitted coefficients in place (cd_0 / cp are kept).

        alpha_zero_lift / alpha_d_min default to the current values when omitted (the
        full-vehicle fit only optimizes the first four).  Rebuilds the stall transition
        since it depends on alpha_0.
        """
        self.cl_1 = cl_1
        self.cl_2 = cl_2
        self.cd = cd
        self.alpha_0 = alpha_0
        if alpha_zero_lift is not None:
            self.alpha_zero_lift = alpha_zero_lift
        if alpha_d_min is not None:
            self.alpha_d_min = alpha_d_min
        self.sigma = sigma_function.SigmaFunction(alpha_0=self.alpha_0)

    def get_cl(self, alpha):
        alpha_attached = alpha - self.alpha_zero_lift
        sigma = self.sigma.compute(alpha)
        return (
            (1 - sigma) * self.cl_1 * alpha_attached
            + sigma * self.cl_2 * np.sin(alpha) * np.cos(alpha)
        )

    def get_cd(self, alpha, u, chord):
        alpha_drag = alpha - self.alpha_d_min
        rn_clamped = np.maximum(self.get_reynolds_number(u, chord), 1)
        return (
            self.cd * np.sin(alpha_drag) ** 2
            + 2 * 1.02 * self.cp / np.sqrt(rn_clamped)
            + self.cd_0
        )

    def to_dict(self):
        return {
            "type": "classical",
            "cl_1": float(self.cl_1),
            "cl_2": float(self.cl_2),
            "cd": float(self.cd),
            "cd_0": float(self.cd_0),
            "cp": float(self.cp),
            "alpha_0": float(self.alpha_0),
            "alpha_zero_lift": float(self.alpha_zero_lift),
            "alpha_d_min": float(self.alpha_d_min),
        }


# Default angle-of-attack knots (radians) for the banded correction (TASK step 4/5):
# anchors at +-10, +-45, +-90, -180 deg plus 0. The interpolation is periodic over 2*pi,
# so -180 deg also covers +180 deg (no separate +180 knot).
_DEFAULT_CORRECTION_KNOTS = np.radians(
    np.array([-180.0, -90.0, -45.0, -10.0, 0.0, 10.0, 45.0, 90.0])
)

_CORRECTION_PERIOD = 2 * np.pi  # angle of attack is periodic; ties +180 deg to -180 deg


class BandCorrectedAeroCoefficients(ClassicalAeroCoefficients):
    """Classical profile plus a bounded piecewise-linear correction band (TASK step 4/5).

    Adds a per-angle-of-attack offset, linearly interpolated between fixed knots, on top of
    the classical CL and CD. Because a linear interpolation of knot offsets that each lie
    within the interval [-band, +band] stays within that interval at every angle of attack,
    the band is enforced simply by bounding the offsets (done later in the fitting layer).
    Zero offsets, the default, reproduce the classical profile exactly.

    The interpolation is periodic over 2*pi, so the correction at +180 deg equals the
    correction at -180 deg (angle of attack wraps); knot sets must not list both +-180.
    """

    def __init__(
        self,
        cl_1=5.3,
        cl_2=1.7,
        alpha_0=np.radians(20.6),
        cd=1.8,
        cd_0=0.01,
        cp=1.328,
        alpha_zero_lift=0.0,
        alpha_d_min=0.0,
        cl_knots=None,
        cl_offsets=None,
        cd_knots=None,
        cd_offsets=None,
    ):
        super().__init__(
            cl_1=cl_1,
            cl_2=cl_2,
            alpha_0=alpha_0,
            cd=cd,
            cd_0=cd_0,
            cp=cp,
            alpha_zero_lift=alpha_zero_lift,
            alpha_d_min=alpha_d_min,
        )
        self.cl_knots = np.asarray(_DEFAULT_CORRECTION_KNOTS if cl_knots is None else cl_knots, dtype=float)
        self.cd_knots = np.asarray(_DEFAULT_CORRECTION_KNOTS if cd_knots is None else cd_knots, dtype=float)
        self.cl_offsets = (np.zeros_like(self.cl_knots) if cl_offsets is None
                           else np.asarray(cl_offsets, dtype=float))
        self.cd_offsets = (np.zeros_like(self.cd_knots) if cd_offsets is None
                           else np.asarray(cd_offsets, dtype=float))

    def get_cl(self, alpha):
        return super().get_cl(alpha) + np.interp(alpha, self.cl_knots, self.cl_offsets,
                                                 period=_CORRECTION_PERIOD)

    def get_cd(self, alpha, u, chord):
        cd = super().get_cd(alpha, u, chord) + np.interp(alpha, self.cd_knots, self.cd_offsets,
                                                         period=_CORRECTION_PERIOD)
        return np.maximum(cd, 0.0)  # drag stays physical after correction

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "band_corrected"
        # .tolist() converts the numpy arrays to native Python floats (YAML-serializable)
        d["cl_knots"] = self.cl_knots.tolist()
        d["cl_offsets"] = self.cl_offsets.tolist()
        d["cd_knots"] = self.cd_knots.tolist()
        d["cd_offsets"] = self.cd_offsets.tolist()
        return d


class NeuroBEMCoefficients(AeroCoefficients):
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

    def to_dict(self):
        return {"type": "neurobem", "a": float(self.a), "d": float(self.d),
                "linear_mode": bool(self.linear_mode)}
