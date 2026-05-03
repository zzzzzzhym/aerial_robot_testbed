import unittest
import torch.nn as nn

import model

class TestModel(unittest.TestCase):
    def test_generate_nets(self):
        phi_net, h_net = model.DaimlModelFactory.generate_nets(2, 11, 5)
        # --- Test number of layers ---
        self.assertIsInstance(phi_net.layers, nn.ModuleList)
        self.assertIsInstance(h_net.layers, nn.ModuleList)

        config = model.DaimlModelFactory.load_config("model_config.yaml")
        self.assertGreaterEqual(len(phi_net.layers), len(config["PhiNet"]["hidden_layer_dimensions"]) + 1)
        self.assertGreaterEqual(len(h_net.layers), len(config["HNet"]["hidden_layer_dimensions"]) + 1)

if __name__ == '__main__':
    unittest.main()