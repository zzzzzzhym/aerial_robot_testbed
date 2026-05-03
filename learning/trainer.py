import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import model
import performance_analyzer
import common_utils

class DaimlTrainer:
    def __init__(self,
                 phi_net: model.MultilayerNet,
                 h_net: model.MultilayerNet,
                 loaderset_phi: list[DataLoader],
                 loaderset_a: list[DataLoader],
                 dim_of_label: int,
                 validation_callback: callable,
                 config_file: str = "trainer_config.yaml",
                 ) -> None:
        self.phi_net = phi_net
        self.h_net = h_net
        self.loaderset_phi = loaderset_phi
        self.loaderset_a = loaderset_a
        self.dim_of_label = dim_of_label
        self.num_of_conditions = self.h_net.dim_of_output   # should match the length of loaderset_phi and loaderset_a
        if self.num_of_conditions != len(loaderset_phi) or self.num_of_conditions != len(loaderset_a):
            raise ValueError("Number of conditions does not match the length of loaderset_phi and loaderset_a")
        self.config = DaimlTrainer.load_config(config_file)
        self.criterion = None
        self.criterion_h = None
        self.optimizer_h = None
        self.optimizer_phi = None
        self.validator_callback = validation_callback  

        # initialization
        self.set_optimizers()
        self.set_criterion()
        self.loss_phi_trace = []
        self.loss_h_trace = []
        self.loss_phi_trace_on_validation = []
        self.a_trace = [
            np.zeros((self.config["num_epochs"], self.phi_net.dim_of_output, self.dim_of_label))
            for _ in range(self.num_of_conditions)
        ]   # a list of 3D array, a_trace[condition_ID][iteration] = adapter matrix, a

    def set_optimizers(self):
        """Set optimizers for phi_net and h_net"""
        self.optimizer_h = optim.Adam(self.h_net.parameters(), lr=self.config["learning_rate_h"])
        self.optimizer_phi = optim.Adam(self.phi_net.parameters(), lr=self.config["learning_rate_phi"])
        self.optimizer_phi.zero_grad()
        self.optimizer_h.zero_grad()

    def set_criterion(self):
        self.criterion = nn.MSELoss()
        self.criterion_h = nn.CrossEntropyLoss()

    @staticmethod
    def load_config(config_file: str):
        """Load model configuration from YAML file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "pure_diaml_net", "trainer_config.yaml"])
        print(f"Loading trainer config from {os.path.relpath(path)}...")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_optimal_a(self, phi: torch.Tensor, ground_truth, is_greedy: bool=False) -> torch.Tensor:
        """Get the optimal adapter matrix and decide whether to use greedy residual fitting.
        When greedy residual fitting is used, the adapter matrix is computed column by column so that the earlier columns of adapter can take more responsibility to fit the ground truth.
        """
        if not is_greedy:
            a = self.get_least_square_of_a(phi, ground_truth)
            a = self.normalize_a(a)
        else:
            num_of_phi_rows, num_of_phi_columns = phi.shape
            # num_of_y_columns = ground_truth.shape[1]
            residual = ground_truth.clone()

            # use list to avoid in-place writes to prevent breaking the autograd graph
            a_rows = []

            for column in range(num_of_phi_columns):
                phi_column = phi[:, column].unsqueeze(1)  # shape (num_of_phi_rows, 1)
                a_row = self.get_least_square_of_a(phi_column, residual).squeeze(0)  # shape (num_of_y_columns,)
                a_rows.append(a_row)
                residual = residual - torch.mm(phi_column, a_row.unsqueeze(0))  # shape (num_of_phi_rows, num_of_y_columns)

            a = torch.stack(a_rows, dim=0)  # shape (num_of_phi_columns, num_of_y_columns)
            a = self.normalize_a(a)
        return a

    def get_least_square_of_a(self, phi: torch.Tensor, ground_truth):
        """
        phi*a = labels
        a = inv(phi_t*phi)*phi_t*labels
        
        Args:
            phi (_type_): network output
            labels (_type_): ground truth

        Returns:
            _type_: _description_
        """
        phi_t = phi.transpose(0,1)
        phi_t_phi_inv = torch.inverse(torch.mm(phi_t, phi))
        a = torch.mm(phi_t_phi_inv, torch.mm(phi_t, ground_truth.to(phi.device)))
        return a

    def normalize_a(self, a: torch.Tensor) -> torch.Tensor:
        """Normalize the adapter matrix a to have Frobenius norm of gamma"""
        if torch.norm(a, 'fro') < 0.00001:
            raise ValueError(f"a is too small, a = {a}")
        else:
            return a / torch.norm(a, 'fro') * self.config["gamma"]

    # def normalize_a(self, a: torch.Tensor) -> torch.Tensor:
    #     """Normalize all but the first row of adapter matrix a to have Frobenius norm gamma."""
    #     # Extract rows to be normalized
    #     a_rest = a[1:]  # shape: (M-1, D)
    #     norm = torch.norm(a_rest, p='fro')
    #     if norm < 1e-5:
    #         raise ValueError(f"a (excluding first row) is too small, a = {a}")
        
    #     # Normalize only rows 1 to end
    #     a_normalized = a.clone()
    #     a_normalized[1:] = a_rest / norm * self.config["gamma"]
    #     return a_normalized
    
    @staticmethod
    def get_prediction(phi, a):
        return torch.mm(phi, a)
    
    def can_train_h_net(self) -> bool:
        """randomly insert h_net training"""
        can_do = np.random.rand() <= 1.0 / self.config["frequency_h"]
        return can_do

    def step_training(self, epoch: int) -> tuple[float, float]:
        # Randomize the order in which we train over the subdatasets
        randomized_cases = np.arange(self.num_of_conditions)
        np.random.shuffle(randomized_cases)
        
        loss_phi_sum = 0.0
        loss_h_sum = 0.0
        a_trace_per_step = [
            np.zeros((0, self.a_trace[0].shape[1], self.a_trace[0].shape[2]))
            for _ in range(len(self.a_trace))
        ]
        alpha = 0.0
        if epoch > self.config["warmup_epoch"]:
            alpha = self.config["alpha"]
        for case_num in randomized_cases:
            with torch.no_grad():
                batch_phi = next(iter(self.loaderset_phi[case_num]))
                batch_a = next(iter(self.loaderset_a[case_num]))
            self.optimizer_phi.zero_grad()   # reset gradient before each training section starts
            prediction_on_adapter_data = self.phi_net(batch_a['input'])
            if self.config["is_dynamic_environment"]:
                # happy path to Neural Fly training process
                a = self.get_optimal_a(prediction_on_adapter_data, batch_a['output'], self.config["use_greedy_residual_fitting"])
            else:
                a = torch.ones(self.a_trace[case_num].shape[1], self.a_trace[case_num].shape[2])
                a[-1,:] = torch.zeros(self.a_trace[case_num].shape[2])  # last row is bias row, do not use
            phi_output = self.phi_net(batch_phi['input'])
            prediction = self.get_prediction(phi_output, a)
            loss_h = self.criterion_h(self.h_net(phi_output), batch_phi['c'])
            loss_phi = self.criterion(prediction, batch_phi['output']) - alpha*loss_h
            # trial code -> 
            # loss_mse = self.criterion(prediction, batch_phi['output'])
            # self.optimizer_phi.zero_grad()
            # loss_mse.backward(retain_graph=True)
            # mse_grad_norm = self.compute_grad_norm(self.phi_net)
            # self.optimizer_phi.zero_grad()
            # loss_h.backward(retain_graph=True)
            # ce_grad_norm = self.compute_grad_norm(self.phi_net)
            # epsilon = 1e-8
            # alpha = float(mse_grad_norm / (ce_grad_norm + epsilon))*0.002
            # alpha = max(0.0000001, min(alpha, 0.0000002))  # Clamp to safe range
            # self.optimizer_phi.zero_grad()
            # loss_phi = loss_mse - alpha * loss_h
            # trial code <-
            loss_phi.backward()
            self.optimizer_phi.step()

            if self.can_train_h_net() or self.config["fine_tune_epoch_h"] < epoch:
                self.optimizer_h.zero_grad() # remove h_net gradient gained from loss_phi 
                phi_output = self.phi_net(batch_phi['input']) # get output again after optimizer step
                loss_h = self.criterion_h(self.h_net(phi_output), batch_phi['c'])
                loss_h.backward()
                self.optimizer_h.step()

            '''
            Spectral normalization
            '''
            if self.config["spectral_norm"] > 0:
                for param in self.phi_net.parameters():
                    M = param.detach().numpy()
                    if M.shape[0] > 0:  # only normalize if the layter takes input (pure bias layer will have shape (0, n))
                        s = np.linalg.norm(M, 2)
                        if s > self.config["spectral_norm"]:
                            param.data = param / s * self.config["spectral_norm"]
            '''
            record loss trace
            '''
            loss_phi_sum += loss_phi.item()
            loss_h_sum += loss_h.item()
            a_np = np.copy(a.detach().numpy())
            a_trace_per_step[case_num] = np.concatenate((a_trace_per_step[case_num], np.array([a_np])), axis=0)
        return loss_phi_sum, loss_h_sum, a_trace_per_step

    def compute_grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train_model(self):
        self.loss_phi_trace = []
        self.loss_h_trace = []
        self.loss_phi_trace_on_validation = []
        for epoch in range(self.config["num_epochs"]):
            # set the model to training mode in case validation mode change it to eval mode from previous epoch
            self.phi_net.train()
            self.h_net.train()
            loss_phi, loss_h, a_trace = self.step_training(epoch)
            self.loss_phi_trace.append(loss_phi/self.num_of_conditions)
            self.loss_h_trace.append(loss_h/self.num_of_conditions)
            self.loss_phi_trace_on_validation.append(self.validator_callback())
            for case_num in range(len(a_trace)):
                self.a_trace[case_num][epoch] = a_trace[case_num]
            if epoch % 100 == 0:
                print(
                    f"[{epoch + 1}] "
                    f"loss_phi: {self.loss_phi_trace[-1]:.2f} "
                    f"loss_h: {self.loss_h_trace[-1]:.2f} "
                    f"loss_validation: {self.loss_phi_trace_on_validation[-1]:.2f}"
                    f"\na_trace: {self.a_trace[0][epoch]}"
                )

    def plot_prediction_error(self, error, groundtruth, prediction, title):
        """assumes the output is 3d (fx,fy,fz)"""
        abs_error = np.abs(error.detach().cpu().numpy()) if isinstance(error, torch.Tensor) else np.abs(error)
        p90 = np.percentile(abs_error, 90, axis=0)
        rms_per_dim = np.sqrt(np.mean(abs_error**2, axis=0)) 
        print(f"RMS: {rms_per_dim}")

        fig, axs = plt.subplots(3, 2, figsize=(10, 6))
        fig.suptitle(f"{title}\n P90 Error: "
                    f"fx={p90[0]:.3f} N, fy={p90[1]:.3f} N, fz={p90[2]:.3f} N", fontsize=12)

        # Plot ground truth vs prediction
        axs[0, 0].plot(groundtruth[:, 0])
        axs[1, 0].plot(groundtruth[:, 1])
        axs[2, 0].plot(groundtruth[:, 2])
        axs[0, 0].plot(prediction[:, 0])
        axs[1, 0].plot(prediction[:, 1])
        axs[2, 0].plot(prediction[:, 2])
        axs[0, 0].legend(["groundtruth", "prediction"]) 
        axs[0, 0].set_ylabel('f_disturb_x')
        axs[1, 0].set_ylabel('f_disturb_y')
        axs[2, 0].set_ylabel('f_disturb_z')

        # Plot errors
        axs[0, 1].plot(error[:, 0])
        axs[1, 1].plot(error[:, 1])
        axs[2, 1].plot(error[:, 2])
        axs[0, 1].legend(["error"])
        axs[0, 1].set_ylabel('error_x')
        axs[1, 1].set_ylabel('error_y')
        axs[2, 1].set_ylabel('error_z')

        # Add P90 text annotation to each error plot
        axs[0, 1].text(0.95, 0.95, f"P90={p90[0]:.3f}", ha='right', va='top', transform=axs[0, 1].transAxes)
        axs[1, 1].text(0.95, 0.95, f"P90={p90[1]:.3f}", ha='right', va='top', transform=axs[1, 1].transAxes)
        axs[2, 1].text(0.95, 0.95, f"P90={p90[2]:.3f}", ha='right', va='top', transform=axs[2, 1].transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle        

    def plot_loss(self):
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.loss_phi_trace)
        axs[0].set_ylabel('loss_phi_trace [N]')
        axs[1].plot(self.loss_h_trace)
        axs[1].set_ylabel('loss_h_trace')
        axs[2].plot(self.loss_phi_trace_on_validation)
        axs[2].set_ylabel('loss_phi_trace_on_validation [N]')
        axs[2].set_xlabel('epoch')

    def plot_tsne_of_a_trace(self):
        if not self.config["is_dynamic_environment"]:
            print("TSNE plot of a_trace is only available when is_dynamic_environment is True.")
        else:
            print("Plotting t-SNE of a_trace...")
            can_print_3d = False
            n = 100
            performance_analyzer.plot_tsne_of_a(self.a_trace, list(range(n, n + 10)))
            if can_print_3d:
                performance_analyzer.plot_tsne_3d_of_a(self.a_trace, list(range(n, n + 10)))

    
class SimpleTrainer:
    def __init__(self,
                 simple_net: model.MultilayerNet,
                 loaderset: list[DataLoader],
                 dim_of_label: int,
                 validation_callback: callable,
                 config_file: str,
                 ) -> None:
        self.simple_net = simple_net
        self.loaderset = loaderset
        self.dim_of_label = dim_of_label
        self.config = SimpleTrainer.load_config(config_file)
        self.criterion = None
        self.optimizer = None
        self.validator_callback = validation_callback
        self.num_of_conditions = len(self.loaderset)

        # initialization
        self.set_optimizers()
        self.set_criterion()
        self.loss_trace = []
        self.loss_trace_on_validation = []
        
    def set_optimizers(self):
        """Set optimizers for simple_net"""
        self.optimizer = optim.Adam(self.simple_net.parameters(), lr=self.config["learning_rate"])
        self.optimizer.zero_grad()

    def set_criterion(self):
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_config(config_file: str):
        """Load model configuration from YAML file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "bemt_infused_multihead_simple_net", "trainer_config.yaml"])
        print(f"Loading trainer config from {os.path.relpath(path)}...")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def step_training(self, epoch: int) -> float:
        # Randomize the order in which we train over the subdatasets
        randomized_cases = np.arange(self.num_of_conditions)
        np.random.shuffle(randomized_cases)
        
        loss_sum = 0.0
        for case_num in randomized_cases:
            with torch.no_grad():
                batch = next(iter(self.loaderset[case_num]))
            self.optimizer.zero_grad()   # reset gradient before each training section starts
            prediction = self.simple_net(batch['input'])
            loss = self.criterion(prediction, batch['output'])
            loss.backward()
            self.optimizer.step()

            '''
            Spectral normalization
            '''
            if self.config["spectral_norm"] > 0:
                for param in self.simple_net.parameters():
                    M = param.detach().numpy()
                    if M.shape[0] > 0:  # only normalize if the layter takes input (pure bias layer will have shape (0, n))
                        s = np.linalg.norm(M, 2)
                        if s > self.config["spectral_norm"]:
                            param.data = param / s * self.config["spectral_norm"]
            '''
            record loss trace
            '''
            loss_sum += loss.item()
        return loss_sum
    
    def train_model(self):
        self.loss_trace = []
        self.loss_trace_on_validation = []
        for epoch in range(self.config["num_epochs"]):
            # set the model to training mode in case validation mode change it to eval mode from previous epoch
            self.simple_net.train()
            loss = self.step_training(epoch)
            self.loss_trace.append(loss/self.num_of_conditions)
            self.loss_trace_on_validation.append(self.validator_callback())
            if epoch % 100 == 0:
                print(
                    f"[{epoch + 1}] "
                    f"loss: {self.loss_trace[-1]:.2f} "
                    f"loss_validation: {self.loss_trace_on_validation[-1]:.2f}"
                )    

    def plot_loss(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.loss_trace)
        axs[0].set_ylabel('loss_trace [N]')
        axs[1].plot(self.loss_trace_on_validation)
        axs[1].set_ylabel('loss_trace_on_validation [N]')
        axs[1].set_xlabel('epoch')
        # np.savez(
        #     f"zerowind.npz",
        #     loss_trace=self.loss_trace,
        #     loss_trace_on_validation=self.loss_trace_on_validation,
        # )
    
class RotorNetTrainer:
    def __init__(self,
                 rotor_net: model.NaiveSumRotorNet | model.AttentionBasedRotorNet,
                 loaderset: list[DataLoader],
                 dim_of_label: int,
                 validation_callback: callable,
                 config_file: str = "trainer_config.yaml",
                 ) -> None:
        self.rotor_net = rotor_net
        self.loaderset = loaderset
        self.dim_of_label = dim_of_label
        self.config = RotorNetTrainer.load_config(config_file)
        self.criterion = None
        self.optimizer = None
        self.validator_callback = validation_callback
        self.num_of_conditions = len(self.loaderset)

        # initialization
        self.set_optimizers()
        self.set_criterion()
        self.loss_trace = []
        self.loss_trace_on_validation = []
        
    def set_optimizers(self):
        """Set optimizers for simple_net"""
        self.optimizer = optim.Adam(self.rotor_net.parameters(), lr=self.config["learning_rate"])
        self.optimizer.zero_grad()

    def set_criterion(self):
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_config(config_file: str):
        """Load model configuration from YAML file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "bemt_rotor_net", config_file])
        print(f"Loading trainer config from {os.path.relpath(path)}...")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def step_training(self, epoch: int) -> float:
        # Randomize the order in which we train over the subdatasets
        randomized_cases = np.arange(self.num_of_conditions)
        np.random.shuffle(randomized_cases)
        
        loss_sum = 0.0
        for case_num in randomized_cases:
            with torch.no_grad():
                batch = next(iter(self.loaderset[case_num]))
            self.optimizer.zero_grad()   # reset gradient before each training section starts
            prediction = self.rotor_net(batch['shared_input'], batch['individual_input'])
            loss = self.criterion(prediction, batch['output'])
            loss.backward()
            self.optimizer.step()

            '''
            Spectral normalization
            '''
            if self.config["spectral_norm"] > 0:
                for param in self.rotor_net.parameters():
                    M = param.detach().numpy()
                    if M.shape[0] > 0:  # only normalize if the layter takes input (pure bias layer will have shape (0, n))
                        s = np.linalg.norm(M, 2)
                        if s > self.config["spectral_norm"]:
                            param.data = param / s * self.config["spectral_norm"]
            '''
            record loss trace
            '''
            loss_sum += loss.item()
        return loss_sum
    
    def train_model(self):
        self.loss_trace = []
        self.loss_trace_on_validation = []
        for epoch in range(self.config["num_epochs"]):
            # set the model to training mode in case validation mode change it to eval mode from previous epoch
            self.rotor_net.train()
            loss = self.step_training(epoch)
            self.loss_trace.append(loss/self.num_of_conditions)
            self.loss_trace_on_validation.append(self.validator_callback())
            if epoch % 100 == 0:
                print(
                    f"[{epoch + 1}] "
                    f"loss: {self.loss_trace[-1]:.2f} "
                    f"loss_validation: {self.loss_trace_on_validation[-1]:.2f}"
                )    

    def plot_loss(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.loss_trace)
        axs[0].set_ylabel('loss_trace [N]')
        axs[1].plot(self.loss_trace_on_validation)
        axs[1].set_ylabel('loss_trace_on_validation [N]')
        axs[1].set_xlabel('epoch')