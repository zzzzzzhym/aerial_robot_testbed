"""Unit tests for the deployable bilinear-basis disturbance handler (STEP 7).

Covers the correctness-critical pieces of learning/bilinear_basis_model.py:
  1. quat_to_R: known quaternion + orthonormality.
  2. Feature CONTRACT: the per-frame dict feature builder (basis_frame_from_fields, used at
     deployment) must exactly match the vectorized training feature builder
     (bilinear_basis_trainer.basis_features) on real CSV rows -- otherwise train/deploy diverge.
  3. The flow-DIRECTION block is a unit vector and carries no magnitude (scaling the background
     wind relative vector leaves it unchanged); the wind magnitude never enters the frame.
  4. Streaming estimator: predict/adapt shapes, and RLS converges on a synthetic linear target.
  5. Shared-P RLS == three independent per-axis RLS filters (the equivalence the module relies on).
  6. Torque on the same basis: starts at zero, converges on a linear target, leaves force untouched.

Run from the repo root:
    python -m unittest learning.test.bilinear_basis_test
or directly:
    python learning/test/bilinear_basis_test.py
"""
import os
import sys
import glob
import tempfile
import unittest

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from learning.bilinear_basis_model import (
    quat_to_R, basis_frame_from_fields, BilinearBasisNet, BilinearBasisEstimator,
    BASIS_FRAME_DIM,
)

TRAIN = os.path.join(REPO, "data", "training", "wind_free_space_training_cfd_our_proposal")


class TestQuatToR(unittest.TestCase):
    def test_identity(self):
        np.testing.assert_allclose(quat_to_R(np.array([1.0, 0, 0, 0])), np.eye(3), atol=1e-12)

    def test_90deg_z(self):
        # +90 deg about world/body z: body-x -> world-y
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        R = quat_to_R(q)
        np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), np.array([0, 1, 0]), atol=1e-7)

    def test_orthonormal(self):
        q = np.array([0.5, 0.5, -0.5, 0.5]); q = q / np.linalg.norm(q)
        R = quat_to_R(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)


class TestFeatureContract(unittest.TestCase):
    """The deploy-time per-frame builder must match the train-time vectorized builder exactly."""
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from learning.bilinear_basis_trainer import basis_features  # vectorized (training path)
        cls.basis_features = staticmethod(basis_features)
        files = sorted(glob.glob(os.path.join(TRAIN, "*.csv")))
        if not files:
            raise unittest.SkipTest("training CSVs not present")
        cls.df = pd.read_csv(files[0])
        from ast import literal_eval
        cls.literal_eval = staticmethod(literal_eval)

    def _row_fields(self, i):
        import numpy as np
        le = self.literal_eval
        def a(n):
            col = self.df[n]
            return np.array(col.apply(le).to_list()) if isinstance(col.iloc[0], str) \
                else np.array(col.to_list())
        fv = {"q": a("q")[i], "v": a("v")[i], "sensed_omega": a("sensed_omega")[i]}
        for r in range(4):
            fv[f"rotor_{r}_rotation_spd"] = a(f"rotor_{r}_rotation_spd")[i]
            fv[f"rotor_{r}_sensed_wind_velocity"] = a(f"rotor_{r}_sensed_wind_velocity")[i]
            fv[f"rotor_{r}_local_wind_velocity"] = a(f"rotor_{r}_local_wind_velocity")[i]
            fv[f"rotor_{r}_velocity"] = a(f"rotor_{r}_velocity")[i]
        return fv

    def test_frame_dim(self):
        f = basis_frame_from_fields(self._row_fields(0))
        self.assertEqual(f.shape, (BASIS_FRAME_DIM,))

    def test_matches_vectorized(self):
        Xvec = self.basis_features(self.df)              # [N, 29]
        for i in [0, 1, 37, 250, len(self.df) - 1]:
            got = basis_frame_from_fields(self._row_fields(i))
            np.testing.assert_allclose(got, Xvec[i], atol=1e-9,
                                       err_msg=f"frame mismatch at row {i}")

    def test_direction_is_unit_and_magnitude_free(self):
        fv = self._row_fields(100)
        f1 = basis_frame_from_fields(fv)
        direction = f1[-3:]
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=5)
        # scaling the background-wind RELATIVE vector (wind-v) by a positive factor must not
        # change the direction block -> magnitude of the relative airspeed never enters the frame.
        v = np.asarray(fv["v"], float)
        for r in range(4):
            key = f"rotor_{r}_local_wind_velocity"
            fv[key] = v + 3.0 * (np.asarray(fv[key], float) - v)  # scale (wind-v) about v, x3
        f2 = basis_frame_from_fields(fv)
        np.testing.assert_allclose(f2[-3:], direction, atol=1e-5)


def _make_ckpt(path, window=4, K=6, frame_dim=BASIS_FRAME_DIM, width=32, blocks=2):
    torch.manual_seed(0)
    net = BilinearBasisNet(window * frame_dim, K, width=width, blocks=blocks)
    in_dim = window * frame_dim
    pkg = {
        "net": net.state_dict(),
        "x_mean": np.zeros(in_dim, np.float32), "x_scale": np.ones(in_dim, np.float32),
        "a0": np.zeros((K + 1, 3), np.float32),
        "window": window, "K": K, "frame_dim": frame_dim, "width": width, "blocks": blocks,
    }
    torch.save(pkg, path)


def _fields(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    fv = {"q": q, "v": rng.normal(size=3), "sensed_omega": rng.normal(size=3)}
    for r in range(4):
        fv[f"rotor_{r}_rotation_spd"] = float(300 + rng.normal())
        fv[f"rotor_{r}_sensed_wind_velocity"] = rng.normal(size=3)
        fv[f"rotor_{r}_local_wind_velocity"] = rng.normal(size=3)
        fv[f"rotor_{r}_velocity"] = rng.normal(size=3)
    return fv


class TestEstimator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ckpt = os.path.join(self.tmp, "ckpt.pth")
        _make_ckpt(self.ckpt)

    def test_predict_adapt_shapes(self):
        est = BilinearBasisEstimator(self.ckpt, lam_ff=0.99, p0=10.0)
        p = est.step(_fields(1), measured_disturbance_force=np.array([1.0, 2.0, 3.0]))
        self.assertEqual(p.shape, (3,))
        np.testing.assert_array_equal(est.get_disturbance_force(), p)

    def test_rls_converges_on_linear_target(self):
        """With a fixed basis, a constant true a* makes f = phi @ a*; RLS must recover it."""
        est = BilinearBasisEstimator(self.ckpt, lam_ff=1.0, p0=1e3)   # pure RLS, diffuse prior
        rng = np.random.default_rng(0)
        a_true = rng.normal(size=(est.K + 1, 3))
        errs = []
        for t in range(400):
            fv = _fields(t)
            est.predict(fv)
            phi = est._phi_cur
            y = phi @ a_true                      # noiseless linear measurement
            errs.append(np.linalg.norm(est.pred - y))
            est.adapt(y)
        # threshold accommodates the float32 basis (weights are stored/run in float32)
        self.assertLess(np.mean(errs[-20:]), 5e-3,
                        f"RLS did not converge (tail err {np.mean(errs[-20:]):.2e})")

    def test_shared_P_equals_per_axis(self):
        """Shared-P matrix RLS == three independent per-axis RLS (module's efficiency claim)."""
        est = BilinearBasisEstimator(self.ckpt, lam_ff=0.97, p0=5.0)
        # independent per-axis reference
        d = est.K + 1
        a_ax = [np.zeros(d) for _ in range(3)]
        P_ax = [np.eye(d) * 5.0 for _ in range(3)]
        lam = 0.97
        rng = np.random.default_rng(3)
        for t in range(50):
            fv = _fields(t + 100)
            est.predict(fv); phi = est._phi_cur
            y = rng.normal(size=3)
            for ax in range(3):
                Pph = P_ax[ax] @ phi
                k = Pph / (lam + phi @ Pph)
                a_ax[ax] = a_ax[ax] + k * (y[ax] - phi @ a_ax[ax])
                P_ax[ax] = (P_ax[ax] - np.outer(k, Pph)) / lam
            est.adapt(y)
        ref = np.stack(a_ax, axis=1)              # [d, 3]
        np.testing.assert_allclose(est.a, ref, atol=1e-9)


class TestEstimatorTorque(unittest.TestCase):
    """Torque reuses the force basis: tau = phi @ a_torque, a_torque by the same (shared-P) RLS."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ckpt = os.path.join(self.tmp, "ckpt.pth")
        _make_ckpt(self.ckpt)

    def test_torque_starts_at_zero(self):
        est = BilinearBasisEstimator(self.ckpt)
        est.predict(_fields(0))
        np.testing.assert_array_equal(est.get_disturbance_torque(), np.zeros(3))

    def test_force_unchanged_by_torque_adaptation(self):
        with_torque = BilinearBasisEstimator(self.ckpt, lam_ff=0.97, p0=5.0)
        force_only = BilinearBasisEstimator(self.ckpt, lam_ff=0.97, p0=5.0)
        rng = np.random.default_rng(7)
        for t in range(50):
            fv = _fields(t)
            f, tq = rng.normal(size=3), rng.normal(size=3)
            with_torque.step(fv, measured_disturbance_force=f, measured_disturbance_torque=tq)
            force_only.step(fv, measured_disturbance_force=f)
        np.testing.assert_array_equal(with_torque.a, force_only.a)
        np.testing.assert_array_equal(with_torque.P, force_only.P)
        np.testing.assert_array_equal(force_only.a_torque, np.zeros_like(force_only.a_torque))

    def test_torque_rls_converges_on_linear_target(self):
        est = BilinearBasisEstimator(self.ckpt, lam_ff=1.0, p0=1e3)
        rng = np.random.default_rng(1)
        a_true = rng.normal(size=(est.K + 1, 3))
        errs = []
        for t in range(400):
            fv = _fields(t)
            est.predict(fv)
            tq = est._phi_cur @ a_true
            errs.append(np.linalg.norm(est.get_disturbance_torque() - tq))
            est.adapt(np.zeros(3), tq)
        self.assertLess(np.mean(errs[-20:]), 5e-3)


if __name__ == "__main__":
    unittest.main()
