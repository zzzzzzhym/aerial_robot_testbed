"""Training for the bilinear-basis disturbance model (phi/a split) -- production.

Builds the two kernel groups and fits them jointly, then saves a checkpoint consumable by
learning/bilinear_basis_model.py:BilinearBasisEstimator (which runs the phi basis + online RLS):

  phi-net (DEPLOYED)  : BilinearBasisNet -- flattened-window ResMLP over magnitude-FREE features
                        (q, v, sensed_omega, rotor speed, kinematic induced velocity v_i, flow
                        direction). Shared with inference via bilinear_basis_model.basis_frame_from_fields.
  a-net   (TRAINING)  : CoefficientNet -- a hypernetwork mapping the wind MAGNITUDE (the single OOD
                        degree of freedom) to the coefficient matrix a in R^{(K+1)x3}. It forces phi
                        to become wind-invariant, then is DISCARDED: at inference a is recovered
                        on-the-fly by RLS. (Wind DIRECTION already lives in phi, so a-net input is 1-D.)

f = [phi(state); 1] @ a.  Checkpoint saves ONLY phi + normalization + an a0 prior (mean per-condition
least-squares a), since the a-net is not deployed.

Run (from repo root):
    python -m learning.bilinear_basis_trainer            # trains on the default our_proposal data
"""
import glob
import os
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from learning.bilinear_basis_model import BilinearBasisNet, BASIS_FRAME_DIM, quat_to_R

torch.set_default_dtype(torch.float32)

WINDOW, K, WIDTH, BLOCKS = 10, 16, 256, 3
DEV = torch.device("cpu")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRAIN = os.path.join(REPO, "data", "training", "wind_free_space_training_cfd_our_proposal")
DEFAULT_OUT = os.path.join(REPO, "learning", "model", "wind_free_space_our_proposal_bilinear_basis.pth")


def _col(df, n):
    v = df[n]
    return np.array(v.apply(literal_eval).to_list()) if isinstance(v.iloc[0], str) else np.array(v.to_list())


def basis_features(df: pd.DataFrame) -> np.ndarray:
    """VECTORIZED magnitude-free per-frame features, identical layout to
    bilinear_basis_model.basis_frame_from_fields:  q,v,omega,spd,v_i(4x3),direction(3) = 29.

    v_i is the KINEMATIC induced velocity (sensed_wind - local_wind + rotor_velocity): in-domain even
    at an OOD background wind and computable online with no lookup table. The flow DIRECTION is the
    unit body-frame relative airspeed (magnitude dropped)."""
    q = _col(df, "q"); v = _col(df, "v"); om = _col(df, "sensed_omega")
    spd = np.stack([_col(df, f"rotor_{r}_rotation_spd").reshape(-1) for r in range(4)], 1)
    vi = np.concatenate([_col(df, f"rotor_{r}_sensed_wind_velocity")
                         - _col(df, f"rotor_{r}_local_wind_velocity")
                         + _col(df, f"rotor_{r}_velocity") for r in range(4)], 1)      # [N,12]
    wind = _col(df, "rotor_0_local_wind_velocity")
    Rt = np.transpose(quat_to_R(q), (0, 2, 1))
    rel = np.einsum("nij,nj->ni", Rt, wind - v)
    direction = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-6)
    X = np.concatenate([q, v, om, spd, vi, direction], 1)
    assert X.shape[1] == BASIS_FRAME_DIM, (X.shape[1], BASIS_FRAME_DIM)
    return X


def wind_magnitude(df: pd.DataFrame) -> np.ndarray:
    """a-net input: |background wind| (the single out-of-domain degree of freedom)."""
    return np.linalg.norm(_col(df, "rotor_0_local_wind_velocity"), axis=1, keepdims=True)


class CoefficientNet(nn.Module):
    """a-net: wind magnitude -> coefficient matrix a in R^{(K+1)x3}. Discarded at inference."""
    def __init__(self, cin: int = 1, k: int = K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, (k + 1) * 3))
        self.k = k

    def forward(self, c):
        return self.net(c).reshape(len(c), self.k + 1, 3)


class _Std:
    def __init__(s, X): s.m = X.mean(0); d = X.std(0); d[d < 1e-6] = 1; s.s = 1 / d
    def f(s, X): return (X - s.m) * s.s


def _windowize(X, Y, w):
    idx = np.arange(w - 1, len(X))
    return np.stack([X[i - w + 1:i + 1].reshape(-1) for i in idx], 0), Y[idx], idx


def _load(files):
    Xs, Cs, Ys, spans = [], [], [], []
    off = 0
    for f in files:
        df = pd.read_csv(f)
        X = basis_features(df); Y = _col(df, "f_disturb_sensed_raw"); C = wind_magnitude(df)
        Xw, Yw, idx = _windowize(X, Y, WINDOW); Cw = C[idx]
        spans.append((off, len(Xw))); off += len(Xw)
        Xs.append(Xw); Cs.append(Cw); Ys.append(Yw)
    return np.concatenate(Xs), np.concatenate(Cs), np.concatenate(Ys), spans


def train_and_save(train_dir: str = DEFAULT_TRAIN, out_path: str = DEFAULT_OUT,
                   seeds=(0, 1, 2), epochs: int = 250, verbose: bool = True):
    """Train the phi/a split on every CSV in train_dir; save the best-val phi basis checkpoint."""
    files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no training CSVs in {train_dir}")
    Xtr, Ctr, Ytr, spans = _load(files)
    xs = _Std(Xtr); ys_sc = Ytr.std(0); ys_sc[ys_sc < 1e-6] = 1
    cs = _Std(Ctr)
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    rng = np.random.default_rng(0); perm = rng.permutation(len(Xtr)); nv = int(0.15 * len(Xtr))
    va, tr = perm[:nv], perm[nv:]
    Xt, Ct, Yt = T(xs.f(Xtr[tr])), T(cs.f(Ctr[tr])), T(Ytr[tr])
    Xv, Cv, Yv = T(xs.f(Xtr[va])), T(cs.f(Ctr[va])), Ytr[va]
    Xall = T(xs.f(Xtr)); fsc = T(ys_sc)
    in_dim = WINDOW * BASIS_FRAME_DIM

    best_basis, best_val = None, 1e9
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        basis = BilinearBasisNet(in_dim, K, width=WIDTH, blocks=BLOCKS).to(DEV)
        anet = CoefficientNet(Ct.shape[1], K).to(DEV)
        opt = torch.optim.Adam(list(basis.parameters()) + list(anet.parameters()), 5e-4, weight_decay=1e-5)
        crit = nn.MSELoss(); best = 1e9; bstate = None; bad = 0
        for ep in range(epochs):
            basis.train(); anet.train(); pm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                b = pm[i:i + 1024]; opt.zero_grad()
                fpred = torch.einsum("bk,bka->ba", basis(Xt[b]), anet(Ct[b]))
                loss = crit(fpred / fsc, Yt[b] / fsc); loss.backward(); opt.step()
            basis.eval(); anet.eval()
            with torch.no_grad():
                fv = torch.einsum("bk,bka->ba", basis(Xv), anet(Cv)).numpy()
            r = float(np.sqrt(np.mean((fv - Yv) ** 2)))
            if r < best - 1e-4:
                best = r; bstate = {k_: v_.clone() for k_, v_ in basis.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad > 30:
                    break
        basis.load_state_dict(bstate); basis.eval()
        if verbose:
            print(f"  seed {sd}: in-dom joint val = {best:.3f}", flush=True)
        if best < best_val:
            best_val, best_basis = best, basis

    # a0 prior = mean per-condition least-squares a on the saved basis (physical units)
    with torch.no_grad():
        Phi_all = best_basis(Xall).numpy()
    a0 = np.mean([np.linalg.lstsq(Phi_all[s:s + n], Ytr[s:s + n], rcond=None)[0] for s, n in spans], 0)

    pkg = {
        "net": best_basis.state_dict(),
        "x_mean": xs.m.astype(np.float32), "x_scale": xs.s.astype(np.float32),
        "a0": a0.astype(np.float32),
        "window": WINDOW, "K": K, "frame_dim": BASIS_FRAME_DIM, "width": WIDTH, "blocks": BLOCKS,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(pkg, out_path)
    if verbose:
        print(f"saved {out_path}  (best-val seed, val={best_val:.3f}, K={K}, window={WINDOW})")
    return out_path


if __name__ == "__main__":
    train_and_save()
