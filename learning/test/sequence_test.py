"""Unit tests for the SEQUENCE_NET additions (models, factory, windowing, save/load).

Run from repo root:
    python -m unittest learning.test.sequence_test
or directly:
    python learning/test/sequence_test.py
"""
import os
import sys
import unittest

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "learning"))

import sequence_model
import data_factory


def make_factory(model_type, feat_dim=21, window=10, dim_of_label=3):
    cfg = {
        "model_type": model_type,
        "ResMLP": {"width": 32, "blocks": 2},
        "GRU": {"hidden": 32, "layers": 2},
        "ResGRU": {"hidden": 32, "layers": 3},
        "TCN": {"channels": 32, "levels": 3, "kernel": 3, "dropout": 0.0},
    }
    return sequence_model.SequenceNetFactory(
        feat_dim, window, dim_of_label,
        np.zeros(feat_dim), np.ones(feat_dim), np.zeros(dim_of_label), np.ones(dim_of_label),
        customized_config=cfg)


class TestSequenceModels(unittest.TestCase):
    def test_output_shape_each_model_type(self):
        B, W, F, O = 8, 10, 21, 3
        x = torch.randn(B, W, F, dtype=torch.float32)
        for mt in sequence_model.SequenceNetFactory.MODEL_TYPES:
            net = make_factory(mt, F, W, O).generate_nets().float()
            y = net(x)
            self.assertEqual(tuple(y.shape), (B, O), f"bad shape for {mt}")

    def test_res_gru_same_param_count_as_gru(self):
        """The 'combined' idea: residual between GRU layers adds no parameters."""
        F, O = 21, 3
        gru = sequence_model.GruNet(F, O, hidden=32, layers=3)
        res_gru = sequence_model.ResGruNet(F, O, hidden=32, layers=3)
        n_gru = sum(p.numel() for p in gru.parameters())
        n_res = sum(p.numel() for p in res_gru.parameters())
        self.assertEqual(n_gru, n_res)

    def test_resmlp_uses_flattened_window(self):
        net = make_factory("resmlp", feat_dim=21, window=10).generate_nets()
        self.assertEqual(net.inp.in_features, 21 * 10)

    def test_factory_roundtrip_reconstructs_same_net(self):
        B, W, F, O = 4, 10, 21, 3
        x = torch.randn(B, W, F, dtype=torch.float32)
        for mt in ("res_gru", "tcn"):
            f1 = make_factory(mt, F, W, O)
            net1 = f1.generate_nets().float().eval()
            cfg = f1.generate_self_config()
            f2 = sequence_model.SequenceNetFactory(
                cfg["feat_dim"], cfg["window"], cfg["dim_of_label"],
                cfg["input_mean"], cfg["input_scale"], cfg["label_mean"], cfg["label_scale"],
                customized_config=cfg["customized_config"])
            net2 = f2.generate_nets().float().eval()
            net2.load_state_dict(net1.state_dict())
            with torch.no_grad():
                np.testing.assert_allclose(net1(x).numpy(), net2(x).numpy(),
                                           rtol=1e-5, atol=1e-6, err_msg=mt)

    def test_tcn_is_causal(self):
        """TCN output at the last step must not depend on any future frame (there are none
        here) and, after chomping, output length matches input -- verify readout only sees
        the window by checking a perturbation of the FIRST frame still changes the output
        (receptive field reaches back) while shapes stay well-formed."""
        F, O, W = 21, 3, 10
        net = make_factory("tcn", F, W, O).generate_nets().float().eval()
        x = torch.randn(1, W, F, dtype=torch.float32)
        with torch.no_grad():
            y0 = net(x)
            x2 = x.clone(); x2[0, 0] += 5.0        # perturb the oldest frame
            y1 = net(x2)
        self.assertEqual(tuple(y0.shape), (1, O))
        self.assertTrue(torch.any(torch.abs(y1 - y0) > 1e-6),
                        "receptive field should reach the first frame")


class TestSequenceWindowing(unittest.TestCase):
    def setUp(self):
        self.factory = data_factory.SequenceDataFactory(
            os.path.join("template", "input_label_map_disturbance_force_label.yaml"))

    def test_build_windows_shape(self):
        n, f, o, w = 100, len(self.factory.input_headers), len(self.factory.label_headers), self.factory.window
        X = np.random.randn(n, f)
        Y = np.random.randn(n, o)
        windows, labels = self.factory.build_windows(X, Y)
        self.assertEqual(windows.shape, (n - w + 1, w, f))
        self.assertEqual(labels.shape, (n - w + 1, o))

    def test_window_contents_and_label_alignment(self):
        f, o, w = len(self.factory.input_headers), len(self.factory.label_headers), self.factory.window
        n = w + 3
        X = np.arange(n * f).reshape(n, f).astype(float)
        Y = np.arange(n * o).reshape(n, o).astype(float)
        windows, labels = self.factory.build_windows(X, Y)
        # first window = frames [0..w-1]; its label = label at the last frame (index w-1)
        np.testing.assert_array_equal(windows[0], X[0:w])
        np.testing.assert_array_equal(labels[0], Y[w - 1])
        # last window ends at the last frame
        np.testing.assert_array_equal(windows[-1], X[n - w:n])
        np.testing.assert_array_equal(labels[-1], Y[n - 1])

    def test_too_short_sequence_yields_empty(self):
        f, o = len(self.factory.input_headers), len(self.factory.label_headers)
        X = np.random.randn(self.factory.window - 1, f)
        Y = np.random.randn(self.factory.window - 1, o)
        windows, labels = self.factory.build_windows(X, Y)
        self.assertEqual(len(windows), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
