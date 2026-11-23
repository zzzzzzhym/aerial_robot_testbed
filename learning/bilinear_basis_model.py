"""Deployable BILINEAR-BASIS disturbance handler (STEP 7): isolate the wind MAGNITUDE so it is
never a deployed-network input, and recover its effect ON-THE-FLY.

    f = [phi(state); 1] @ a

  - phi(state): a flattened-window ResMLP basis, DEPLOYED. Its inputs are ONLY in-domain,
    magnitude-FREE features -- quaternion, ego velocity v, sensed body rate, rotor speeds,
    rotor INDUCED velocity v_i = sensed_wind - local_wind + rotor_velocity (in-domain even at an
    OOD background wind), and the flow DIRECTION unit(R^T(wind - v)) with NO magnitude. Because
    the network never sees the wind magnitude, it never extrapolates -> no OOD collapse.
  - a in R^{(K+1)x3}: the per-axis coefficients that carry ALL the wind dependence. A separate
    hypernetwork (WindNet) produced `a` during training so phi would become a wind-invariant
    basis; that hypernetwork is DISCARDED at deployment. Online, `a` is recovered by causal
    recursive-least-squares on the measured disturbance -- exactly the knob that lets the handler
    track an out-of-domain wind a plain feed-forward net cannot predict.

This is the self-contained inference counterpart of claude/learning/experiments/step7_windnet.py
(best OOD result to date: ~1.11 N on the x=-9 run vs direct GRU 3.0-3.7, adaptive est 1.26, prior
NFG invariant-basis 1.68). Checkpoints are produced by
claude/learning/experiments/step7_train_deploy.py.

Streaming protocol (causal, no leakage): each tick assemble the magnitude-free feature frame,
push it on the window buffer, compute phi, PREDICT with the current `a`, then UPDATE `a` (and the
shared covariance P) from the measured disturbance via RLS with a forgetting factor.
"""
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# Feature construction (shared contract between training and deployment)
# ----------------------------------------------------------------------------------------------
# Per-frame magnitude-free feature layout (29 dims), in this fixed order:
#   q(4) + v(3) + sensed_omega(3) + rotor_rotation_spd(4) + v_i(4 rotors x 3) + flow_direction(3)
BASIS_FRAME_DIM = 29
ROTORS = 4


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Rotation matrix body->world from a [w,x,y,z] quaternion. q: [...,4] -> R: [...,3,3]."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def basis_frame_from_fields(field_values: dict) -> np.ndarray:
    """Build ONE magnitude-free feature frame [29] from a dict of raw controller fields.

    Required keys (value = array-like):
      q(4), v(3), sensed_omega(3),
      rotor_{r}_rotation_spd (scalar), rotor_{r}_sensed_wind_velocity(3),
      rotor_{r}_local_wind_velocity(3), rotor_{r}_velocity(3)   for r in 0..3
    The background-wind VECTOR (rotor_0_local_wind_velocity) enters only through its DIRECTION;
    its magnitude never reaches the feature vector.
    """
    def g(name):
        return np.atleast_1d(np.asarray(field_values[name], dtype=float)).reshape(-1)

    q = g("q"); v = g("v"); om = g("sensed_omega")
    spd = np.array([g(f"rotor_{r}_rotation_spd")[0] for r in range(ROTORS)])
    vi = []
    for r in range(ROTORS):
        sw = g(f"rotor_{r}_sensed_wind_velocity")
        lw = g(f"rotor_{r}_local_wind_velocity")
        rv = g(f"rotor_{r}_velocity")
        vi.append(sw - lw + rv)                       # induced velocity (in-domain at OOD wind)
    vi = np.concatenate(vi)                            # [12]
    wind = g("rotor_0_local_wind_velocity")           # background wind (world)
    R = quat_to_R(q)
    rel = R.T @ (wind - v)                             # body-frame relative airspeed
    direction = rel / (np.linalg.norm(rel) + 1e-6)    # DIRECTION only, magnitude dropped
    return np.concatenate([q, v, om, spd, vi, direction])


# ----------------------------------------------------------------------------------------------
# Deployed basis network (flattened window -> K-dim basis with an appended bias column)
# ----------------------------------------------------------------------------------------------
class _ResMLP(nn.Module):
    """Residual-block MLP (LayerNorm + SiLU). Mirrors the training architecture
    (claude/learning/experiments/models.py:ResMLP) so a checkpoint loads exactly."""
    def __init__(self, din, dout, width=256, blocks=3):
        super().__init__()
        self.inp = nn.Linear(din, width)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict({
                "ln": nn.LayerNorm(width),
                "fc1": nn.Linear(width, width),
                "fc2": nn.Linear(width, width),
                "drop": nn.Dropout(0.0),
            }))
        self.out_ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, dout)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            z = b["ln"](h)
            z = F.silu(b["fc1"](z))
            z = b["drop"](z)
            z = b["fc2"](z)
            h = h + z
        return self.out(self.out_ln(h))


class BilinearBasisNet(nn.Module):
    """Flattened-window ResMLP -> phi(state) in R^K with a bias column appended -> [N, K+1]."""
    def __init__(self, in_dim: int, k: int, width: int = 256, blocks: int = 3):
        super().__init__()
        self.net = _ResMLP(in_dim, k, width=width, blocks=blocks)
        self.k = k

    def forward(self, x):                             # [N, window*frame_dim] -> [N, K+1]
        phi = self.net(x)
        return torch.cat([phi, torch.ones(len(x), 1, device=x.device, dtype=x.dtype)], dim=1)


# ----------------------------------------------------------------------------------------------
# Streaming estimator
# ----------------------------------------------------------------------------------------------
class BilinearBasisEstimator:
    """Streaming bilinear-basis + online-RLS disturbance estimator (physical-unit force N).

    lam_ff : RLS forgetting factor in (0,1]; <1 tracks a drifting (OOD) wind, =1 is pure RLS.
    p0     : initial covariance scale (P0 = I*p0). Larger p0 = weaker trust in the prior a0.

    The RLS here uses a single covariance P shared across the 3 force axes with a per-axis
    coefficient column a[:,axis]. Because phi, lam and P0 are identical across axes, this is
    numerically identical to running three independent per-axis RLS filters (the experiment form).

    Torque reuses the same basis: tau = [phi(state); 1] @ a_torque, a_torque in R^{(K+1)x3} (N m).
    The basis was trained on force only, so a_torque has no trained prior (starts at zero) and is
    recovered online by the same RLS. P (and the gain) do not depend on the measurement, so the
    torque columns share P with the force columns and the force estimate is unchanged.
    """
    def __init__(self, ckpt_path: str, lam_ff: float = 0.98, p0: float = 10.0,
                 use_prior: bool = True):
        pkg = torch.load(ckpt_path, map_location="cpu")
        self.window = int(pkg["window"]); self.K = int(pkg["K"])
        self.frame_dim = int(pkg["frame_dim"])
        self.in_dim = self.window * self.frame_dim
        self.net = BilinearBasisNet(self.in_dim, self.K,
                                    width=int(pkg.get("width", 256)),
                                    blocks=int(pkg.get("blocks", 3)))
        self.net.load_state_dict(pkg["net"]); self.net.float(); self.net.eval()
        self.x_mean = np.asarray(pkg["x_mean"], float)     # [window*frame_dim]
        self.x_scale = np.asarray(pkg["x_scale"], float)   # multiplicative (1/std)
        self.a0 = np.asarray(pkg["a0"], float)             # [K+1, 3] physical-unit prior
        self.lam_ff = float(lam_ff); self.p0 = float(p0)
        self.use_prior = bool(use_prior)
        self.reset()

    def reset(self):
        self.a = self.a0.copy() if self.use_prior else np.zeros_like(self.a0)
        self.a_torque = np.zeros((self.K + 1, 3))      # no trained torque prior
        self.P = np.eye(self.K + 1) * self.p0
        self.buffer = collections.deque(maxlen=self.window)
        self.pred = np.zeros(3)
        self.pred_torque = np.zeros(3)
        self._phi_cur = None

    def _phi(self, field_values: dict) -> np.ndarray:
        frame = basis_frame_from_fields(field_values)
        if len(self.buffer) == 0:
            for _ in range(self.window):               # left-pad at startup
                self.buffer.append(frame)
        else:
            self.buffer.append(frame)
        win = np.concatenate(list(self.buffer))        # [window*frame_dim], trained-frame order
        xn = ((win - self.x_mean) * self.x_scale)[None]
        with torch.no_grad():
            return self.net(torch.from_numpy(xn).float()).numpy()[0]   # [K+1]

    def predict(self, field_values: dict) -> np.ndarray:
        """Compute phi for this tick and return the physical disturbance-force prediction
        (the torque prediction is computed alongside, see get_disturbance_torque)."""
        self._phi_cur = self._phi(field_values)
        self.pred = self._phi_cur @ self.a            # a is already in physical units
        self.pred_torque = self._phi_cur @ self.a_torque
        return self.pred

    def adapt(self, measured_disturbance_force, measured_disturbance_torque=None) -> None:
        """RLS update of a (and shared P) from the measured disturbance for the last tick.
        If measured_disturbance_torque is given, a_torque is updated with the same gain."""
        phi = self._phi_cur
        y = np.asarray(measured_disturbance_force, float).reshape(-1)[:3]
        Pphi = self.P @ phi
        g = Pphi / (self.lam_ff + phi @ Pphi)
        err = y - phi @ self.a                         # [3]
        self.a = self.a + np.outer(g, err)
        if measured_disturbance_torque is not None:
            y_torque = np.asarray(measured_disturbance_torque, float).reshape(-1)[:3]
            self.a_torque = self.a_torque + np.outer(g, y_torque - phi @ self.a_torque)
        self.P = (self.P - np.outer(g, Pphi)) / self.lam_ff

    def step(self, field_values: dict, measured_disturbance_force=None,
             measured_disturbance_torque=None) -> np.ndarray:
        """Causal step: predict with current a, then (if a measurement is given) adapt."""
        self.predict(field_values)
        if measured_disturbance_force is not None:
            self.adapt(measured_disturbance_force, measured_disturbance_torque)
        return self.pred

    def get_disturbance_force(self) -> np.ndarray:
        return self.pred

    def get_disturbance_torque(self) -> np.ndarray:
        return self.pred_torque
