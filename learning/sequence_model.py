"""Sequence / windowed disturbance-force models and their factory.

Four architectures selectable by config (`model_type`):
  - "resmlp":  residual MLP on the flattened time window (LayerNorm + SiLU + residual blocks)
  - "gru":     plain stacked GRU over the window
  - "res_gru": stacked GRU with a residual skip BETWEEN layers (same size as "gru"),
               the best configuration found in the exploration (see claude/learning/WORKLOG.md)
  - "tcn":     dilated causal 1D-conv stack over the window (learned FIR filter over time);
               parallel, stateless, inherently regularizing (see claude/learning/WORKLOG.md)

All take an input window of shape [batch, window, feat_dim] and output [batch, dim_of_label].
ResMLP flattens the window internally; TCN transposes to [batch, feat_dim, window]; the data
format fed to the factory is uniform across model types.
"""
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import common_utils


# --------------------------------------------------------------------------------------
# Building block
# --------------------------------------------------------------------------------------
class _ResBlock(nn.Module):
    """Pre-LayerNorm residual block: h -> h + W2 SiLU(W1 LN(h))."""
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h):
        z = self.ln(h)
        z = F.silu(self.fc1(z))
        z = self.fc2(z)
        return h + z


# --------------------------------------------------------------------------------------
# Models (uniform interface: forward([B, window, feat_dim]) -> [B, dim_of_label])
# --------------------------------------------------------------------------------------
class ResMlpNet(nn.Module):
    """Residual MLP on the flattened window."""
    def __init__(self, feat_dim: int, window: int, dim_of_label: int, width: int, blocks: int):
        super().__init__()
        self.inp = nn.Linear(feat_dim * window, width)
        self.blocks = nn.ModuleList([_ResBlock(width) for _ in range(blocks)])
        self.out_ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim_of_label)

    def forward(self, x):  # [B, W, F]
        h = self.inp(x.reshape(x.shape[0], -1))
        for block in self.blocks:
            h = block(h)
        return self.out(self.out_ln(h))


class GruNet(nn.Module):
    """Plain stacked GRU over the window; readout from the last timestep."""
    def __init__(self, feat_dim: int, dim_of_label: int, hidden: int, layers: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.SiLU(), nn.Linear(hidden, dim_of_label))

    def forward(self, x):  # [B, W, F]
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class ResGruNet(nn.Module):
    """Stacked GRU with a residual skip between layers (same parameter count as GruNet).
    First layer keeps the cheap raw-feature input width; subsequent layers run at `hidden`
    width with h <- h + GRU(h). This is the winning 'combined' architecture."""
    def __init__(self, feat_dim: int, dim_of_label: int, hidden: int, layers: int):
        super().__init__()
        self.gru0 = nn.GRU(feat_dim, hidden, batch_first=True)
        self.rest = nn.ModuleList([nn.GRU(hidden, hidden, batch_first=True)
                                   for _ in range(layers - 1)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.SiLU(), nn.Linear(hidden, dim_of_label))

    def forward(self, x):  # [B, W, F]
        h, _ = self.gru0(x)
        for gru in self.rest:
            out, _ = gru(h)
            h = h + out                      # residual skip between GRU layers (0 extra params)
        return self.head(h[:, -1])


class _CausalChomp(nn.Module):
    """Trim the right-side padding a causal dilated conv adds, so output length == input."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):  # [B, C, W + chomp] -> [B, C, W]
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TcnNet(nn.Module):
    """Temporal convolutional network over the window: stacked dilated causal 1D convs
    (a learned FIR filter over time), with the readout taken from the last (most recent)
    timestep. Weight sharing across time and a fixed receptive field make it inherently
    regularizing; it is also parallel and stateless at inference."""
    def __init__(self, feat_dim: int, dim_of_label: int, channels: int, levels: int,
                 kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        c_in = feat_dim
        for i in range(levels):
            dilation = 2 ** i
            pad = (kernel - 1) * dilation
            layers += [nn.Conv1d(c_in, channels, kernel, padding=pad, dilation=dilation),
                       _CausalChomp(pad), nn.SiLU(), nn.Dropout(dropout)]
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(),
                                  nn.Linear(channels, dim_of_label))

    def forward(self, x):  # [B, W, F]
        z = self.net(x.transpose(1, 2))   # [B, C, W]
        return self.head(z[:, :, -1])     # readout from the most recent timestep


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
class SequenceNetFactory:
    MODEL_TYPES = ("resmlp", "gru", "res_gru", "tcn")

    def __init__(self,
                 feat_dim: int,
                 window: int,
                 dim_of_label: int,
                 input_mean,
                 input_scale,
                 label_mean,
                 label_scale,
                 customized_config: dict = None):
        self.config = self.set_up_config(customized_config)
        self.feat_dim = feat_dim
        self.window = window
        self.dim_of_label = dim_of_label
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.label_mean = label_mean
        self.label_scale = label_scale

    def generate_nets(self):
        model_type = self.config["model_type"]
        if model_type == "resmlp":
            c = self.config["ResMLP"]
            return ResMlpNet(self.feat_dim, self.window, self.dim_of_label, c["width"], c["blocks"])
        elif model_type == "gru":
            c = self.config["GRU"]
            return GruNet(self.feat_dim, self.dim_of_label, c["hidden"], c["layers"])
        elif model_type == "res_gru":
            c = self.config["ResGRU"]
            return ResGruNet(self.feat_dim, self.dim_of_label, c["hidden"], c["layers"])
        elif model_type == "tcn":
            c = self.config["TCN"]
            return TcnNet(self.feat_dim, self.dim_of_label, c["channels"], c["levels"],
                          c.get("kernel", 3), c.get("dropout", 0.0))
        raise ValueError(f"unknown model_type '{model_type}', expected one of {self.MODEL_TYPES}")

    @staticmethod
    def set_up_config(customized_config: dict = None):
        if customized_config is None:
            return SequenceNetFactory.load_config(SequenceNetFactory.get_default_config_path())
        return customized_config

    @staticmethod
    def get_default_config_path():
        return common_utils.file_manager.find_path_to_folder(
            ["learning", "config", "sequence_net", "model_config.yaml"])

    @staticmethod
    def load_config(config_path: str):
        print(f"Loading model config from {os.path.relpath(config_path)}...")
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def generate_self_config(self) -> dict:
        return {
            "feat_dim": self.feat_dim,
            "window": self.window,
            "dim_of_label": self.dim_of_label,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "label_mean": self.label_mean,
            "label_scale": self.label_scale,
            "customized_config": self.config,
        }


def save_sequence_model(name, net, model_factory_config: dict, input_label_map: dict) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")
    cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}  # portable across devices
    torch.save({"net": cpu_state,
                "model_factory_init_args": model_factory_config,
                "sequence_net_io_fields": input_label_map},
               file_path)
    print(f"Model saved to {os.path.relpath(file_path, os.getcwd())}")


def load_sequence_model(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")
    package = torch.load(file_path, map_location="cpu")
    args = package["model_factory_init_args"]
    factory = SequenceNetFactory(
        args["feat_dim"], args["window"], args["dim_of_label"],
        args["input_mean"], args["input_scale"], args["label_mean"], args["label_scale"],
        customized_config=args["customized_config"])
    net = factory.generate_nets()
    net.load_state_dict(package["net"])
    net.eval()
    print("sequence net model_type:", factory.config["model_type"], "| window:", args["window"])
    print("sequence net input output fields:")
    for key, value in package["sequence_net_io_fields"].items():
        print(key, value)
    return net
