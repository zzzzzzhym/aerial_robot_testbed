import torch 
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.DoubleTensor')
import yaml
import os

import common_utils


class MultilayerNet(nn.Module):
    """Takes config that specifies the input, hidden layer and output dimensions and makes the layers accordingly.
    The forward function need to be defined as requested by nn.Module. But it's not shown in this base class."""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, can_append_bias: bool = False):
        super().__init__()
        self.dim_of_input = dim_of_input
        self.dim_of_output = dim_of_output

        self.layers = nn.ModuleList()
        if can_append_bias:
            nn_output = dim_of_output - 1 # append bias appended in forward
        else:
            nn_output = dim_of_output
        if len(dim_of_layers) == 0: # No hidden layers, direct input to output    
            self.layers.append(nn.Linear(dim_of_input, nn_output))
        else:
            self.layers.append(nn.Linear(dim_of_input, dim_of_layers[0]))
            for i in range(1, len(dim_of_layers)):
                self.layers.append(nn.Linear(dim_of_layers[i - 1], dim_of_layers[i]))
            self.layers.append(nn.Linear(dim_of_layers[-1], nn_output))    

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # relu can only range [0, inf], last layer has no relu to prevent limited range
        return x 

class PhiNet(MultilayerNet):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, input_mean, input_scale, output_mean, output_scale):
        super().__init__(dim_of_input, dim_of_output, dim_of_layers, can_append_bias=True)
        # Normalization: use register_buffer to avoid being changed by optimizer
        self.register_buffer("input_mean", torch.tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))

        # output normalization is used as (a*phi(x) - output_mean)*output_scale
        # it's not used when outputing the kernel, but used after applying adaptation coefficients
        self.register_buffer("output_mean", torch.tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input, assuming x is raw input <- should not do that. dataset is already normalized. Todo: check inference implementation
        x = (x - self.input_mean) * self.input_scale

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        if len(x.shape) == 1:
            # single input (batch size == 1)
            one = torch.ones(1, device=x.device)
            return torch.cat([one, x])
        else:
            # batch input for training (assuming that x is at most 2D)
            ones = torch.ones([x.shape[0], 1], device=x.device)
            return torch.cat([ones, x], dim=-1)
        
class HNet(MultilayerNet):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list):
        super().__init__(dim_of_input, dim_of_output, dim_of_layers, can_append_bias=False)        


class SimpleNet(nn.Module):
    """A simple feedforward network without normalization or bias append"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, dim_of_shared_features, head_layer_dimensions: list, input_mean, input_scale, output_mean, output_scale):
        super().__init__()
        # Normalization: use register_buffer to avoid being changed by optimizer
        self.register_buffer("input_mean", torch.tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        self.register_buffer("output_mean", torch.tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float32))

        self.shared_backbone = MultilayerNet(dim_of_input, dim_of_shared_features, dim_of_layers, can_append_bias=False)

        self.heads = nn.ModuleList()
        for _ in range(dim_of_output):
            net = MultilayerNet(dim_of_shared_features, 1, head_layer_dimensions, can_append_bias=False)
            self.heads.append(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) * self.input_scale
        shared_net_output = self.shared_backbone(x) 
        head_outputs = [head(shared_net_output) for head in self.heads]
        output = torch.cat(head_outputs, dim=-1)    # concatenate head outputs to get final output
        return output  # scale up to match the range of disturbance forces


class NaiveSumRotorNet(nn.Module):
    """A per rotor network without normalization or bias append"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, dim_of_shared_features, head_layer_dimensions: list, input_mean, input_scale, output_mean, output_scale):
        super().__init__()
        self.shared_net = SimpleNet(dim_of_input,
          dim_of_output,
          dim_of_layers,
          dim_of_shared_features,
          head_layer_dimensions,
          input_mean, input_scale, output_mean, output_scale)
        
    def forward(self, shared: torch.Tensor, individual_list: list[torch.Tensor]) -> torch.Tensor:
        total = 0        # 0 works as safe accumulator for tensors
        for ind in individual_list:
            each_rotor_input = torch.cat([shared, ind], dim=-1)  # dim = -1 is equivalent to dim = 1 here
            out = self.shared_net(each_rotor_input)               
            total = total + out                                  
        return total


class AttentionBasedRotorNet(nn.Module):
    """Transformer version"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, dim_of_shared_features, head_layer_dimensions: list, input_mean, input_scale, output_mean, output_scale):
        super().__init__()
        d_model = 8 # TODO: put in config
        self.shared_net = SimpleNet(dim_of_input,
          d_model,
          dim_of_layers,
          dim_of_shared_features,
          head_layer_dimensions,
          input_mean, input_scale, output_mean, output_scale)
        
        nhead = 2 # TODO: put in config
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.output_head = nn.Linear(d_model, dim_of_output)

        
    def forward(self, shared: torch.Tensor, individual_list: list[torch.Tensor]) -> torch.Tensor:
        # 1) same as before, but save all per-rotor outputs as a sequence
        outs = []
        for ind in individual_list:
            each_rotor_input = torch.cat([shared, ind], dim=-1)
            out = self.shared_net(each_rotor_input)  # assume [batch_size, dim_of_feature_vector_per_rotor]
            outs.append(out.unsqueeze(1))   # [batch_size, 1, dim_of_feature_vector_per_rotor]

        seq = torch.cat(outs, dim=1)        # [batch_size, num_rotors, dim_of_feature_vector_per_rotor]

        # 2) self-attention over rotors
        context, attn_weights = self.attn(seq, seq, seq)  # [batch_size, num_rotors, dim_of_feature_vector_per_rotor], [batch_size, num_rotors, num_rotors]

        # 3) replace naive sum with pooled attention output
        pooled = context.mean(dim=1)        # [batch_size, dim_of_feature_vector_per_rotor]

        # 4) final head (you can add a Linear here)
        return self.output_head(pooled)

class DaimlModelFactory:
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
        phi_net = PhiNet(
            self.dim_of_input,
            self.config["dim_of_feature"],
            self.config["PhiNet"]["hidden_layer_dimensions"],
            self.input_mean,
            self.input_scale,
            self.label_mean,
            self.label_scale
        )
        h_net = HNet(
            self.config["dim_of_feature"],
            self.num_of_conditions,
            self.config["HNet"]["hidden_layer_dimensions"]
        )
        return phi_net, h_net

    @staticmethod
    def set_up_config(customized_config: dict = None):
        """Set up the model configuration. If customized_config is None, load the default config."""
        if customized_config is None:
            config_path = DaimlModelFactory.get_default_config_path()
            config = DaimlModelFactory.load_config(config_path)
        else:
            config = customized_config  # in case of loading the trained model, use the documented config at generation time
        return config

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "pure_diaml_net", "model_config.yaml"])
        return path

    @staticmethod
    def load_config(config_path: str):
        """Load model configuration from YAML file"""
        print(f"Loading model config from {os.path.relpath(config_path)}...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def generate_self_config(self) -> dict:
        """Generate a self-contained configuration dictionary. 
        Use this as input argument to recreate the same factory and produce the exact model later."""
        return {
            "num_of_conditions": self.num_of_conditions,
            "dim_of_input": self.dim_of_input,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "label_mean": self.label_mean,
            "label_scale": self.label_scale,
            "customized_config": self.config
        }
    
class SimpleNetFactory:
    def __init__(self,
                 dim_of_input,
                 dim_of_label,
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
        self.dim_of_input = dim_of_input
        self.dim_of_label = dim_of_label
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.label_mean = label_mean
        self.label_scale = label_scale        

    def generate_nets(self):
        net = SimpleNet(
            self.dim_of_input,
            self.dim_of_label,
            self.config["SimpleNet"]["hidden_layer_dimensions"],
            self.config["SimpleNet"]["dim_of_shared_features"],
            self.config["SimpleNet"]["head_layer_dimensions"],
            self.input_mean,
            self.input_scale,
            self.label_mean,
            self.label_scale
        )
        return net

    @staticmethod
    def set_up_config(customized_config: dict = None):
        """Set up the model configuration. If customized_config is None, load the default config."""
        if customized_config is None:
            config_path = SimpleNetFactory.get_default_config_path()
            config = SimpleNetFactory.load_config(config_path)
        else:
            config = customized_config  # in case of loading the trained model, use the documented config at generation time
        return config

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "bemt_infused_multihead_simple_net", "model_config.yaml"])
        return path

    @staticmethod
    def load_config(config_path: str):
        """Load model configuration from YAML file"""
        print(f"Loading model config from {os.path.relpath(config_path)}...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def generate_self_config(self) -> dict:
        """Generate a self-contained configuration dictionary. 
        Use this as input argument to recreate the same factory and produce the exact model later."""
        return {
            "dim_of_input": self.dim_of_input,
            "dim_of_label": self.dim_of_label,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "label_mean": self.label_mean,
            "label_scale": self.label_scale,
            "customized_config": self.config
        }


class RotorNetFactory:
    def __init__(self,
                 dim_of_input,
                 dim_of_label,
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
        self.dim_of_input = dim_of_input    # dim_of_shared_input + dim_of_individual_input
        self.dim_of_label = dim_of_label
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.label_mean = label_mean
        self.label_scale = label_scale        
        # temporary solution to hard code input_scale and mean. TODO: do it in data factory
        self.input_scale =torch.ones(self.dim_of_input, requires_grad=False)
        self.input_scale[-1] = 1  # scale down rpm input
        self.input_mean = torch.zeros(self.dim_of_input, requires_grad=False)
        self.input_mean[-1] = 1000

    def generate_nets(self):
        if self.config["has_attention"]:
            net = AttentionBasedRotorNet(
                self.dim_of_input,
                self.dim_of_label,
                self.config["SimpleNet"]["hidden_layer_dimensions"],
                self.config["SimpleNet"]["dim_of_shared_features"],
                self.config["SimpleNet"]["head_layer_dimensions"],
                self.input_mean,
                self.input_scale,
                self.label_mean,
                self.label_scale
            )
        else:
            net = NaiveSumRotorNet(
                self.dim_of_input,
                self.dim_of_label,
                self.config["SimpleNet"]["hidden_layer_dimensions"],
                self.config["SimpleNet"]["dim_of_shared_features"],
                self.config["SimpleNet"]["head_layer_dimensions"],
                self.input_mean,
                self.input_scale,
                self.label_mean,
                self.label_scale
            )
        return net

    @staticmethod
    def set_up_config(customized_config: dict = None):
        """Set up the model configuration. If customized_config is None, load the default config."""
        if customized_config is None:
            config_path = RotorNetFactory.get_default_config_path()
            config = RotorNetFactory.load_config(config_path)
        else:
            config = customized_config  # in case of loading the trained model, use the documented config at generation time
        return config

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "bemt_rotor_net", "model_config.yaml"])
        return path

    @staticmethod
    def load_config(config_path: str):
        """Load model configuration from YAML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def generate_self_config(self) -> dict:
        """Generate a self-contained configuration dictionary. 
        Use this as input argument to recreate the same factory and produce the exact model later."""
        return {
            "dim_of_input": self.dim_of_input,
            "dim_of_label": self.dim_of_label,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "label_mean": self.label_mean,
            "label_scale": self.label_scale,
            "customized_config": self.config
        }


def save_daiml_model(name, phi_net: PhiNet, h_net: HNet, model_factory_config: dict, input_label_map: dict) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "model", name + ".pth")
        torch.save({"phi": phi_net.state_dict(),
                    "h": h_net.state_dict(),
                    "model_factory_init_args": model_factory_config, 
                    "phi_net_io_fields": input_label_map}, 
                    file_path)
        print(f"Model saved to {os.path.relpath(file_path, os.getcwd())}")

def load_daiml_model(name) -> tuple[PhiNet, HNet, dict]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")       
    package = torch.load(file_path)

    # recreate the model factory instance that generated the trained model
    model_factory_instance = DaimlModelFactory(
        package["model_factory_init_args"]["num_of_conditions"],
        package["model_factory_init_args"]["dim_of_input"],
        package["model_factory_init_args"]["input_mean"],
        package["model_factory_init_args"]["input_scale"],
        package["model_factory_init_args"]["label_mean"],
        package["model_factory_init_args"]["label_scale"],
        customized_config=package["model_factory_init_args"]["customized_config"]
    )
    phi, h = model_factory_instance.generate_nets()
    phi.load_state_dict(package["phi"])
    phi.eval()  # set to eval mode
    h.load_state_dict(package["h"])
    h.eval()
    print("phi net input output fields:")
    for key, value in package["phi_net_io_fields"].items():
        print(key, value)
    return phi, h

def save_simple_model(name, simple_net: SimpleNet, model_factory_config: dict, input_label_map: dict) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "model", name + ".pth")
        torch.save({"simple": simple_net.state_dict(),
                    "model_factory_init_args": model_factory_config,
                    "simple_net_io_fields": input_label_map}, 
                    file_path)
        print(f"Model saved to {os.path.relpath(file_path, os.getcwd())}")

def load_simple_model(name) -> SimpleNet:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")       
    package = torch.load(file_path)

    # recreate the model factory instance that generated the trained model
    model_factory_instance = SimpleNetFactory(
        package["model_factory_init_args"]["dim_of_input"],
        package["model_factory_init_args"]["dim_of_label"],
        package["model_factory_init_args"]["input_mean"],
        package["model_factory_init_args"]["input_scale"],
        package["model_factory_init_args"]["label_mean"],
        package["model_factory_init_args"]["label_scale"],
        customized_config=package["model_factory_init_args"]["customized_config"]
    )
    simple_net = model_factory_instance.generate_nets()
    simple_net.load_state_dict(package["simple"])
    simple_net.eval()
    print("simple net input output fields:")
    for key, value in package["simple_net_io_fields"].items():
        print(key, value)
    return simple_net

def save_rotor_net_model(name, rotor_net: NaiveSumRotorNet | AttentionBasedRotorNet, model_factory_config: dict, input_label_map: dict) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "model", name + ".pth")
        torch.save({"rotor": rotor_net.state_dict(),
                    "model_factory_init_args": model_factory_config,
                    "rotor_net_io_fields": input_label_map}, 
                    file_path)
        print(f"Model saved to {os.path.relpath(file_path, os.getcwd())}")

def load_rotor_net_model(name) -> NaiveSumRotorNet | AttentionBasedRotorNet:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")       
    package = torch.load(file_path)

    # recreate the model factory instance that generated the trained model
    model_factory_instance = RotorNetFactory(
        package["model_factory_init_args"]["dim_of_input"],
        package["model_factory_init_args"]["dim_of_label"],
        package["model_factory_init_args"]["input_mean"],
        package["model_factory_init_args"]["input_scale"],
        package["model_factory_init_args"]["label_mean"],
        package["model_factory_init_args"]["label_scale"],
        customized_config=package["model_factory_init_args"]["customized_config"]
    )
    rotor_net = model_factory_instance.generate_nets()
    rotor_net.load_state_dict(package["rotor"])
    rotor_net.eval()
    print("rotor net input output fields:")
    for key, value in package["rotor_net_io_fields"].items():
        print(key, value)
    print("has_attention:", model_factory_instance.config["has_attention"])
    return rotor_net





