import torch 
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.DoubleTensor')
import yaml
import os

import model

class ViNet(model.MultilayerNet):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list):
        super().__init__(dim_of_input, dim_of_output, dim_of_layers, can_append_bias=False)   


class ViNetFactory(model.DaimlModelFactory):
    def __init__(self,
                 num_of_conditions,
                 dim_of_input,
                 input_mean,
                 input_scale,
                 label_mean,
                 label_scale,
                 customized_config: dict = None
                 ):
        """consider the following example to predict disturbance forces:
        (v_0 v_1 v_2 q_0 q_1 q_2 q_3 pwm_0 pwm_1 pwm_2 pwm_3)
        velocity (3) + quaternion (4) + input (4) = 11
        dim_of_input: input dimension (11 in this case)
        dim_of_feature: features that phi_net need to capture (not necessarily the same as dim of disturbance forces because it needs to interact with adaptation coefficients)
        From the paper, the feature dimension is 5.
        dim_of_label: dimension of label vector. Because we are trying to compare the disturbance force in x y z direction,
        the label vector is a 3-dim vector because disturbance torque is not estimated in the paper.
        """               
        self.config = self.set_up_config(customized_config)
        self.num_of_conditions = num_of_conditions
        self.dim_of_input = dim_of_input
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.label_mean = label_mean
        self.label_scale = label_scale

    def generate_nets(self):
        vi_net = ViNet(
            self.dim_of_input,
            self.config["dim_of_output"],
            self.config["ViNet"]["hidden_layer_dimensions"],
            self.input_mean,
            self.input_scale,
            self.label_mean,
            self.label_scale
        )
        return vi_net



