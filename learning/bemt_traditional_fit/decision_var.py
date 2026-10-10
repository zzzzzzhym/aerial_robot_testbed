"""Decision variables for the BEMT fitting pipelines.

Owns the optimizer's search space for each fitting mode: parameter names, bounds, which
parameters are angles (for display), and the map between the search vector and the
physical blade coefficients.  Centralizing this here keeps the models, objective, solver
and engine free of search-space details — they talk to a ``DecisionVar`` instead.

Most modes search directly in physical coefficients (identity conversions).  The
single-rotor mode reparameterizes to enforce the CL-shape constraint; see
``OffsetAugmentedAeroCoeffDecisionVar``.
"""
import numpy as np


class DecisionVar:
    """A fitting mode's search space and its map to physical coefficients.

    Default conversions are identity (the search vector *is* the physical vector).
    Subclasses override :meth:`to_physical` / :meth:`to_search` to reparameterize.
    """

    def __init__(self, search_names, physical_names, bounds, angle_params=()):
        self.search_names = tuple(search_names)      # search-space parameter names
        self.physical_names = tuple(physical_names)  # physical coefficient names
        self.bounds = list(bounds)                   # per-search-parameter (lo, hi)
        self.angle_params = tuple(angle_params)      # physical names shown in degrees

    def to_physical(self, x):
        """Search-space vector -> physical coefficient vector."""
        return np.asarray(x, dtype=float)

    def to_search(self, physical):
        """Physical coefficient vector -> a valid search-space seed."""
        return np.asarray(physical, dtype=float)

    def clip(self, x):
        """Clip a search-space vector into ``bounds``."""
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return np.clip(np.asarray(x, dtype=float), lo, hi)

    def format(self, values):
        """Format a search-space vector for display, always in physical coefficients."""
        parts = []
        for name, value in zip(self.physical_names, self.to_physical(values)):
            if name in self.angle_params:
                parts.append(f"{name}={np.degrees(value):.3f}deg")
            else:
                parts.append(f"{name}={value:.3f}")
        return "  ".join(parts)


_OFFSET_AUGMENTED_AERO_COEFF_BOUNDS = [
    (2.0, 50.0),                            # cl_1
    (0.0, 0.99),                            # k_cl_2        (0 <= k_cl_2 <= 0.99)
    (0.0, 5.0),                             # cd
    (np.radians(1.0), np.radians(40.0)),    # delta_alpha   (> 0)
    (np.radians(-10.0), np.radians(10.0)),  # alpha_zero_lift
    (np.radians(-10.0), np.radians(10.0)),  # alpha_d_min
]


class OffsetAugmentedAeroCoeffDecisionVar(DecisionVar):
    """Single-rotor search space, reparameterized to constrain the CL curve (goal 2).

    The search replaces alpha_0 and cl_2 with delta_alpha = alpha_0 - alpha_zero_lift
    (the angular width of the linear/attached lift region, > 0) and k_cl_2 via
    cl_2 = 2*k_cl_2*cl_1*delta_alpha.  This turns the CL-shape constraint
    cl_2 < 2*cl_1*(alpha_0 - alpha_zero_lift) — which stops a spurious second lift peak
    in the stalled region — plus the ordering alpha_0 > alpha_zero_lift into plain box
    bounds the optimizer can enforce directly.
    """

    def __init__(self):
        super().__init__(
            search_names=("cl_1", "k_cl_2", "cd", "delta_alpha", "alpha_zero_lift", "alpha_d_min"),
            physical_names=("cl_1", "cl_2", "cd", "alpha_0", "alpha_zero_lift", "alpha_d_min"),
            bounds=_OFFSET_AUGMENTED_AERO_COEFF_BOUNDS,
            angle_params=("alpha_0", "alpha_zero_lift", "alpha_d_min"),
        )

    def to_physical(self, x):
        cl_1, k_cl_2, cd, delta_alpha, alpha_zero_lift, alpha_d_min = x
        alpha_0 = alpha_zero_lift + delta_alpha
        cl_2 = 2.0 * k_cl_2 * cl_1 * delta_alpha
        return np.array([cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min])

    def to_search(self, physical):
        """Inverse of :meth:`to_physical`, clipped into ``bounds`` for use as a seed.
        """
        cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min = physical
        delta_alpha = alpha_0 - alpha_zero_lift
        denominator_guard = 0.01    # avoid devided by zero
        k_cl_2 = cl_2 / (2.0 * cl_1 * max(delta_alpha, denominator_guard))
        return self.clip([cl_1, k_cl_2, cd, delta_alpha, alpha_zero_lift, alpha_d_min])


def full_vehicle():
    """Full four-rotor fit: physical search over (cl_1, cl_2, cd, alpha_0, k_body_drag)."""
    return DecisionVar(
        search_names=("cl_1", "cl_2", "cd", "alpha_0", "k_body_drag"),
        physical_names=("cl_1", "cl_2", "cd", "alpha_0", "k_body_drag"),
        bounds=[
            (2.0, 50.0),                        # cl_1
            (0.0, 50.0),                        # cl_2
            (0.0, 5.0),                         # cd
            (np.radians(10), np.radians(40)),   # alpha_0
            (0.0, 10.0),                        # k_body_drag
        ],
        angle_params=("alpha_0",),
    )


def body_drag():
    """Body-drag-only fit: physical search over the single parameter k_body_drag."""
    return DecisionVar(
        search_names=("k_body_drag",),
        physical_names=("k_body_drag",),
        bounds=[(0.0, 10.0)],
        angle_params=(),
    )
