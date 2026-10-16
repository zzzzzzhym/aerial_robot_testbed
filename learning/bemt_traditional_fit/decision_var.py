"""Decision variables for the BEMT fitting pipelines.

Owns the optimizer's search space for each fitting mode: parameter names, bounds, which
parameters are angles (for display), and the map between the search vector and the
physical blade coefficients.  Centralizing this here keeps the models, objective, solver
and engine free of search-space details — they talk to a DecisionVar instead.

Most modes search directly in physical coefficients (identity conversions). The
single-rotor mode reparameterizes to enforce the CL-shape constraint; see
OffsetAugmentedAeroCoeffDecisionVar.
"""
import numpy as np


class DecisionVar:
    """A fitting mode's search space and its map to physical coefficients.

    Default conversions are identity (the search vector is the physical vector).
    Subclasses override to_physical / to_search to reparameterize.
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
        """Clip a search-space vector into the per-parameter bounds."""
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return np.clip(np.asarray(x, dtype=float), lo, hi)

    _FORMAT_ITEMS_PER_LINE = 6  # wrap long parameter lists onto several lines

    def _format_named(self, names, vec):
        items = []
        for name, value in zip(names, vec):
            if name in self.angle_params:
                items.append(f"{name}={np.degrees(value):.3f}deg")
            else:
                items.append(f"{name}={value:.3f}")
        per = self._FORMAT_ITEMS_PER_LINE
        return "\n    ".join("  ".join(items[i:i + per]) for i in range(0, len(items), per))

    def format(self, values):
        """Format a search-space vector for display, wrapped onto several lines.

        Identity modes print the physical coefficients (the search vars are those coeffs).
        Reparameterized modes print the physical coefficients, then a ``[decision: ...]``
        line showing only the parameters the reparameterization actually changes (those
        whose search name differs from the physical name) — pass-through params and
        correction offsets, being identical, are not repeated.
        """
        values = np.asarray(values, dtype=float)
        physical_str = self._format_named(self.physical_names, self.to_physical(values))
        if self.search_names == self.physical_names:
            return physical_str
        reparam = [(n, v) for n, v, pn in zip(self.search_names, values, self.physical_names) if n != pn]
        decision_str = self._format_named([n for n, _ in reparam], [v for _, v in reparam])
        return f"{physical_str}\n    [decision: {decision_str}]"


_OFFSET_AUGMENTED_AERO_COEFF_BOUNDS = [
    (2.0, 20.0),                            # cl_1
    (0.0, 0.6),                            # k_cl_2        (0 <= k_cl_2 <= 0.99)
    (0.0, 3.0),                             # cd
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
            angle_params=("alpha_0", "delta_alpha", "alpha_zero_lift", "alpha_d_min"),
        )

    def to_physical(self, x):
        cl_1, k_cl_2, cd, delta_alpha, alpha_zero_lift, alpha_d_min = x
        alpha_0 = alpha_zero_lift + delta_alpha
        cl_2 = 2.0 * k_cl_2 * cl_1 * delta_alpha
        return np.array([cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min])

    def to_search(self, physical):
        """Inverse of to_physical, clipped into the bounds for use as a seed."""
        cl_1, cl_2, cd, alpha_0, alpha_zero_lift, alpha_d_min = physical
        delta_alpha = alpha_0 - alpha_zero_lift
        denominator_guard = 0.01    # avoid devided by zero
        k_cl_2 = cl_2 / (2.0 * cl_1 * max(delta_alpha, denominator_guard))
        return self.clip([cl_1, k_cl_2, cd, delta_alpha, alpha_zero_lift, alpha_d_min])


# Fitted correction knots (radians) across the full angle-of-attack range. The correction
# interpolation is periodic (see BandCorrectedAeroCoefficients), so -180 deg also covers
# +180 deg and no separate +180 knot is listed.
BAND_CORRECTION_KNOTS = np.radians([-180.0, -90.0, -45.0, -10.0, 10.0, 45.0, 90.0])
BAND_CORRECTION_CL_LIMIT = 0.5   # +- band on each CL offset
BAND_CORRECTION_CD_LIMIT = 0.2   # +- band on each CD offset


class BandCorrectedDecisionVar(DecisionVar):
    """Offset-augmented single-rotor search plus a bounded CL/CD correction band.

    The first six search parameters are the offset-augmented aero coefficients (see
    OffsetAugmentedAeroCoeffDecisionVar). The remaining parameters are the per-knot CL
    offsets followed by the per-knot CD offsets. Each offset is box-bounded to its amplitude
    band; because a linear interpolation of in-band knot offsets stays in-band at every angle
    of attack, that box bound enforces the whole correction band (no extra constraint needed).
    The offsets are physical already, so only the first six parameters are reparameterized.
    """

    def __init__(self, cl_knots=BAND_CORRECTION_KNOTS, cd_knots=BAND_CORRECTION_KNOTS,
                 cl_limit=BAND_CORRECTION_CL_LIMIT, cd_limit=BAND_CORRECTION_CD_LIMIT):
        self._base = OffsetAugmentedAeroCoeffDecisionVar()
        self.cl_knots = np.asarray(cl_knots, dtype=float)
        self.cd_knots = np.asarray(cd_knots, dtype=float)
        self.n_base = len(self._base.search_names)
        self.n_cl = len(self.cl_knots)
        self.n_cd = len(self.cd_knots)
        cl_names = tuple(f"dcl_{i}" for i in range(self.n_cl))
        cd_names = tuple(f"dcd_{i}" for i in range(self.n_cd))
        super().__init__(
            search_names=self._base.search_names + cl_names + cd_names,
            physical_names=self._base.physical_names + cl_names + cd_names,
            bounds=(self._base.bounds
                    + [(-cl_limit, cl_limit)] * self.n_cl
                    + [(-cd_limit, cd_limit)] * self.n_cd),
            angle_params=self._base.angle_params,
        )

    def to_physical(self, x):
        x = np.asarray(x, dtype=float)
        base_physical = self._base.to_physical(x[:self.n_base])
        return np.concatenate([base_physical, x[self.n_base:]])

    def to_search(self, physical):
        physical = np.asarray(physical, dtype=float)
        base_search = self._base.to_search(physical[:self.n_base])
        return self.clip(np.concatenate([base_search, physical[self.n_base:]]))


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
