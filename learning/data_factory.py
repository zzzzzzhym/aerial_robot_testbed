from dataclasses import field
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pickle
import pandas as pd
from ast import literal_eval

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split

import normalization
import common_utils

class DaimlDataset(Dataset):

    def __init__(self,
        input: np.ndarray,
        output: np.ndarray,
        condition_id: int,
        input_mean_vector,
        input_scale_vector,
        label_mean_vector,
        label_scale_vector,
        source_file: str) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
            condition_id (np.ndarray): an ID number to label the condition of the data
        """
        self.input = input
        self.output = output
        self.c = condition_id
        self.source_file = source_file
        self.input_mean_vector = input_mean_vector
        self.input_scale_vector = input_scale_vector
        self.label_mean_vector = label_mean_vector
        self.label_scale_vector = label_scale_vector

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx, :]
        output = self.output[idx, :]
        sample = {'input': torch.tensor(input), 
                  'output': torch.tensor(output), 
                  'c': torch.tensor(self.c)}
        return sample

class SimpleDataset(Dataset):

    def __init__(self,
        input: np.ndarray,
        output: np.ndarray,
        input_mean_vector,
        input_scale_vector,
        label_mean_vector,
        label_scale_vector,
        source_file: str) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
        """
        self.input = input
        self.output = output
        self.source_file = source_file
        self.input_mean_vector = input_mean_vector
        self.input_scale_vector = input_scale_vector
        self.label_mean_vector = label_mean_vector
        self.label_scale_vector = label_scale_vector

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx, :]
        output = self.output[idx, :]
        sample = {'input': torch.tensor(input), 
                  'output': torch.tensor(output)}
        return sample
    
class RotorNetDataset(Dataset):

    def __init__(self,
        shared_input: np.ndarray,
        individual_input: list[np.ndarray],
        output: np.ndarray,
        input_mean_vector,
        input_scale_vector,
        label_mean_vector,
        label_scale_vector,
        source_file: str) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
        """
        self.shared_input = shared_input
        self.individual_input = individual_input
        self.output = output
        self.source_file = source_file
        self.input_mean_vector = input_mean_vector
        self.input_scale_vector = input_scale_vector
        self.label_mean_vector = label_mean_vector
        self.label_scale_vector = label_scale_vector

    def __len__(self):
        return len(self.shared_input)

    def __getitem__(self, idx):
        shared_input = torch.tensor(self.shared_input[idx, :])
        individual_input = [torch.tensor(each[idx, :]) for each in self.individual_input]
        output = torch.tensor(self.output[idx, :])
        sample = {'shared_input': shared_input, 
                  'individual_input': individual_input,
                  'output': output}
        return sample


class FittingDataset:
    """Dataset for fitting the model parameters"""
    def __init__(self, df: pd.DataFrame, path_to_data_file: str) -> None:
        self.path_to_data_file = path_to_data_file

        self.u_free_0 = np.array(df["rotor_0_local_wind_velocity"].to_list())
        self.v_forward_0 = np.array(df["rotor_0_velocity"].to_list())
        self.omega_0 = np.array(df["rotor_0_rotation_spd"].to_list())

        self.u_free_1 = np.array(df["rotor_1_local_wind_velocity"].to_list())
        self.v_forward_1 = np.array(df["rotor_1_velocity"].to_list())
        self.omega_1 = np.array(df["rotor_1_rotation_spd"].to_list())

        self.u_free_2 = np.array(df["rotor_2_local_wind_velocity"].to_list())
        self.v_forward_2 = np.array(df["rotor_2_velocity"].to_list())
        self.omega_2 = np.array(df["rotor_2_rotation_spd"].to_list())

        self.u_free_3 = np.array(df["rotor_3_local_wind_velocity"].to_list())
        self.v_forward_3 = np.array(df["rotor_3_velocity"].to_list())
        self.omega_3 = np.array(df["rotor_3_rotation_spd"].to_list())
        
        self.shared_r_disk = np.array(df["shared_r_disk"].to_list())

        self.rotor_0_f_rotor_inertial_frame = np.array(df["rotor_0_f_rotor_inertial_frame"].to_list())
        self.rotor_1_f_rotor_inertial_frame = np.array(df["rotor_1_f_rotor_inertial_frame"].to_list())
        self.rotor_2_f_rotor_inertial_frame = np.array(df["rotor_2_f_rotor_inertial_frame"].to_list())
        self.rotor_3_f_rotor_inertial_frame = np.array(df["rotor_3_f_rotor_inertial_frame"].to_list())

        # self.dv = np.array(df["dv"].to_list())    # ground truth data
        # self.omega = np.array(df["omega"].to_list())  # ground truth data
        self.dv = np.array(df["sensed_dv"].to_list())
        self.omega = np.array(df["sensed_omega"].to_list())
        # self.omega_dot = np.array(df["omega_dot"].to_list())
        # self.f_disturb = np.array(df["f_disturb"].to_list())
        # self.torque_disturb = np.array(df["torque_disturb"].to_list())

        self.f_residual = None

    def attach_residual_force(self, f_residual):
        if (len(f_residual) == len(self.omega)):
            print("f_residual mismatch the length with the other data")
        else:
            self.f_residual = f_residual

    def is_ready_for_second_training(self):
        is_ready = False
        if self.f_residual is None:
            print("f_residual is not attached")
        else:
            is_ready = True
        return is_ready

    def __len__(self):
        return len(self.u_free_0)


class DataFactory:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_path_to_data_file(file_name: str) -> str:
        """Get the path to the data file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(current_dir)
        file_path = os.path.join(upper_dir, "data", "training", file_name)
        return file_path  
    
    @staticmethod
    def process_raw_data_frame(df):
        for header in df.columns:
            if isinstance(df[header][0], str):
                df[header] = df[header].apply(literal_eval)

    @staticmethod
    def get_path_to_data_file(file_name: str) -> str:
        """Get the path to the data file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(current_dir)
        file_path = os.path.join(upper_dir, "data", "training", file_name)
        return file_path

    @staticmethod
    def make_data_frame_from_csv(file: str, selected_columns: list[str]=None) -> pd.DataFrame:
        file_path = DataFactory.get_path_to_data_file(file)
        if selected_columns is None:
            data = pd.read_csv(file_path)
        else:
            data = pd.read_csv(file_path, usecols=selected_columns)
        DataFactory.process_raw_data_frame(data)
        return data

class FittingFactory(DataFactory):
    def __init__(self) -> None:
        super().__init__()

    def prepare_datasets(self, data_menu: list) -> FittingDataset:
        """Prepare the datasets from the data menu
        Each file in the data menu is a different condition, the length of the dataset is the number of conditions"""
        datasets = []
        for file in data_menu:
            file_path = DataFactory.get_path_to_data_file(file)
            data = DataFactory.make_data_frame_from_csv(file)
            dataset = FittingDataset(data, file_path)
            datasets.append(dataset)
        return datasets
    
class TrainingDataFactory(DataFactory):
    def __init__(self, input_label_map_file: str) -> None:
        super().__init__()
        self.input_label_map = TrainingDataFactory.get_map(input_label_map_file)
        self.input_headers = TrainingDataFactory.extract_header_from_input_label_map(self.input_label_map["input"])
        self.label_headers = TrainingDataFactory.extract_header_from_input_label_map(self.input_label_map["label"])
        self.input_normalization, self.label_normalization = self.initialize_normalization_dict()
        self.normalization_params_file_path = TrainingDataFactory.find_path_to_normalization_params_file(input_label_map_file)
        self.config = TrainingDataFactory.load_config(self.get_default_config_path())        

    @staticmethod
    def get_map(map_file: str) -> dict:
        map_file_path = DataFactory.get_path_to_data_file(map_file)
        with open(map_file_path, 'r') as file:
            map = yaml.safe_load(file)
        return map

    @staticmethod
    def extract_header_from_input_label_map(column_names: dict[list]) -> list[str]:
        """Extract the header names from the input_label_map

        Args:
            column_names (dict[list]): column_names[name] = None if the column is 1D, otherwise column_names[name] = [0, 1, 2] for a 3D column

        Returns:
            list[str]: List of header names, with dimension suffixes where applicable
        """
        headers = []
        for column, dimensions in column_names.items():
            if dimensions is None:
                headers.append(column)
            else:
                headers.extend([f"{column}_{i}" for i in dimensions])
        return headers

    @staticmethod
    def extract_data_from_data_frame(column_names: dict[list], df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract data from the DataFrame based on the specified column names."""
        data = []
        for column, dimensions in column_names.items():
            col_data = np.array(df[column].to_list())
            if dimensions is None:
                data.append(col_data)
            else:
                data.append(col_data[:, dimensions])
        data = np.column_stack(data)
        return data

    @staticmethod
    def find_path_to_normalization_params_file(input_label_map_file: str) -> str:
        input_label_map_file_path = DataFactory.get_path_to_data_file(input_label_map_file)
        input_label_map_dir = os.path.dirname(input_label_map_file_path)
        normalization_params_file_path = os.path.join(input_label_map_dir, "normalization_params.yaml")
        return normalization_params_file_path 

    @staticmethod
    def get_default_config_path():
        raise NotImplementedError("Set default config path for specific data factory implementation")

    @staticmethod
    def load_config(config_path: str):
        """Load model configuration from YAML file"""
        print(f"Loading model config from {os.path.relpath(config_path)}...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def find_input_label_column(input_label_map: dict):
        """use the config files to find the input and label columns"""
        input_columns = []
        for key, value in input_label_map["input"].items():
            input_columns.append(key)
        label_columns = []
        for key, value in input_label_map["label"].items():
            label_columns.append(key)
        return input_columns, label_columns

    def load_sim_data(self, file_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the csv file and find the columns of interest."""
        input_columns, label_columns = TrainingDataFactory.find_input_label_column(self.input_label_map)
        df = DataFactory.make_data_frame_from_csv(file_name, input_columns + label_columns)
        input_data = TrainingDataFactory.extract_data_from_data_frame(self.input_label_map["input"], df)
        label_data = TrainingDataFactory.extract_data_from_data_frame(self.input_label_map["label"], df)
        return input_data, label_data

    def initialize_normalization_dict(self):
        """When normalization applies to each dataset, this template can be copied as a starting point."""
        input_normalization: dict[str, normalization.Normalization] = {}
        label_normalization: dict[str, normalization.Normalization] = {}
        for field, dimensions in self.input_label_map["input"].items():
            if dimensions is None:
                input_normalization[field] = normalization.Normalization()
            else:
                for dim in dimensions:
                    input_normalization[f"{field}_{dim}"] = normalization.Normalization()
        for field, dimensions in self.input_label_map["label"].items():
            if dimensions is None:
                label_normalization[field] = normalization.Normalization()
            else:
                for dim in dimensions:
                    label_normalization[f"{field}_{dim}"] = normalization.Normalization()
        return input_normalization, label_normalization
    
    def normalize_data(self, data_menu: list) -> None:
        """Normalize the data using the mean and standard deviation"""
        # traverse the datasets
        for file in data_menu:
            input_data, label_data = TrainingDataFactory.load_sim_data(file)
            for column in self.input_headers:
                self.input_normalization[column].add_batch(input_data)
            for column in self.label_headers:
                self.label_normalization[column].add_batch(label_data)

    def generate_normalization_params_file(self, output_path: str) -> None:
        """Generate the normalization parameters for input and output"""
        content = {"input": {}, "output": {}}
        for column in self.input_headers:
            mean, scale = self.input_normalization[column].get_normalization_params()
            content["input"][column] = {"mean": mean, "scale": scale}
        for column in self.label_headers:
            mean, scale = self.label_normalization[column].get_normalization_params()
            content["output"][column] = {"mean": mean, "scale": scale}
        with open(output_path, 'w') as file:
            yaml.dump(content, file)

    def make_normalization_params(self, data_menu: list) -> None:
        # if the normalization params file is not generated, generate it
        if not os.path.exists(self.normalization_params_file_path):
            print("Normalization params file not found, generating it and saving it to\n" + os.path.relpath(self.normalization_params_file_path))
            self.normalize_data(data_menu)
            self.generate_normalization_params_file(self.normalization_params_file_path)
            print("Normalization params file generated")
        # load the normalization params file
        print("Loading normalization params file from\n" + os.path.relpath(self.normalization_params_file_path))
        with open(self.normalization_params_file_path, 'r') as file:
            normalization_params = yaml.safe_load(file)

        # make normalization matrix
        mean = []
        scale = []
        if normalization_params["input"] is None: 
            input_mean_vector = np.zeros(len(self.input_headers))
            input_scale_vector = np.ones(len(self.input_headers))
        for field in self.input_label_map["input"]:
            if field in normalization_params["input"].keys():
                if self.input_label_map["input"][field] is None:
                    mean.append(normalization_params["input"][field]["mean"])
                    scale.append(normalization_params["input"][field]["scale"])
                else:
                    for dim in self.input_label_map["input"][field]:
                        mean.append(normalization_params["input"][field][dim]["mean"])
                        scale.append(normalization_params["input"][field][dim]["scale"])
            else:
                if self.input_label_map["input"][field] is None:
                    mean.append(0.0)
                    scale.append(1.0)
                else:
                    for dim in self.input_label_map["input"][field]:
                        mean.append(0.0)
                        scale.append(1.0)
            input_mean_vector = np.array(mean)
            input_scale_vector = np.array(scale)
        mean = []
        scale = []
        if normalization_params["label"] is None: 
            label_mean_vector = np.zeros(len(self.label_headers))
            label_scale_vector = np.ones(len(self.label_headers))
        else:
            for field in self.input_label_map["label"]:
                if field in normalization_params["label"].keys():
                    if self.input_label_map["label"][field] is None:
                        mean.append(normalization_params["label"][field]["mean"])
                        scale.append(normalization_params["label"][field]["scale"])
                    else:
                        for dim in self.input_label_map["label"][field]:
                            mean.append(normalization_params["label"][field][dim]["mean"])
                            scale.append(normalization_params["label"][field][dim]["scale"])
                else:
                    if self.input_label_map["label"][field] is None:
                        mean.append(0.0)
                        scale.append(1.0)
                    else:
                        for dim in self.input_label_map["label"][field]:
                            mean.append(0.0)
                            scale.append(1.0)
            label_mean_vector = np.array(mean)
            label_scale_vector = np.array(scale)
        self.input_mean_vector = input_mean_vector
        self.input_scale_vector = input_scale_vector
        self.label_mean_vector = label_mean_vector
        self.label_scale_vector = label_scale_vector

    @staticmethod
    def plot_dataset_distribution_grid(data: np.ndarray, sub_titles: list[str], title: str, bins=50, figsize=(20, 10), cols=7):
        num_features = data.shape[1]
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        for i in range(num_features):
            sns.histplot(data[:, i], bins=bins, kde=True, ax=axes[i])
            axes[i].set_title(sub_titles[i])
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True)
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title)

class DaimlDataFactory(TrainingDataFactory):
    """Data manager is responsible for:
    1. defining the list of data to be used to train
    2. check which columns are input and which are labels
    3. generate LearningDataset and then LoaderSets for trainer
    4. normalize the data"""
    def __init__(self, input_label_map_file: str) -> None:
        super().__init__(input_label_map_file)
        self.num_of_conditions = 0

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "pure_diaml_net", "data_factory_config.yaml"])
        return path

    def set_num_of_conditions(self, data_menu: list) -> None:
        self.num_of_conditions = len(data_menu)

    def convert_sim_to_training_data(self, input_data: np.ndarray, label_data: np.ndarray, condition_id: int, file: str) -> DaimlDataset:
        # normalize data before putting it into the dataset
        result = DaimlDataset(input_data,   # input normalization will be done in the network
                                 (label_data - self.label_mean_vector)*self.label_scale_vector,
                                 condition_id,
                                 self.input_mean_vector,
                                 self.input_scale_vector,
                                 self.label_mean_vector,
                                 self.label_scale_vector,
                                 file)
        return result

    def prepare_datasets(self, data_menu: list, can_inspect_data: bool=False) -> list[DaimlDataset]:
        """Prepare the datasets from the data menu
        Each file in the data menu is a different condition, the length of the dataset is the number of conditions"""
        datasets = []
        condition_id = 0    # ID value doesn't matter, but each different condition should have different ID
        for file in data_menu:
            input_data, label_data = self.load_sim_data(file)
            if can_inspect_data:
                TrainingDataFactory.plot_dataset_distribution_grid(input_data, self.input_headers, file + " input")  # inspect the data
                TrainingDataFactory.plot_dataset_distribution_grid(label_data, self.label_headers, file + " label")  # inspect the data
            dataset = self.convert_sim_to_training_data(input_data, label_data, condition_id, file)
            condition_id += 1
            datasets.append(dataset)
        return datasets

    def get_data_loaders(self, training_data: DaimlDataset) -> tuple[DataLoader, DataLoader]:
        """Separate the dataset into two parts: part1 for phi NN and part2 for adaptation coeffecients, 
        generate a data loader for each part"""
        length = int(len(training_data)*self.config["data_usage_ratio"])
        print(f"Using {length} samples from the dataset for training", f"from source file: {training_data.source_file}")
        ratio = self.config["phi_shot"]/(self.config["phi_shot"] + self.config["a_shot"])
        part_phi = int(length*ratio)
        part_a = length - part_phi
        if self.config["can_shuffle"]:
            phi_set, a_set, _ = random_split(training_data, [part_phi, part_a, len(training_data) - part_phi - part_a])
        else:
            phi_set = Subset(training_data, range(part_phi))
            a_set = Subset(training_data, range(part_phi, part_phi + part_a))
        phi_loader = torch.utils.data.DataLoader(phi_set, batch_size=self.config["phi_shot"], shuffle=self.config["can_shuffle"])
        a_loader = torch.utils.data.DataLoader(a_set, batch_size=self.config["a_shot"], shuffle=True)
        return phi_loader, a_loader

    def prepare_loadersets(self, datasets: list[DaimlDataset]) -> tuple[list[DataLoader], list[DataLoader]]:
        """Generate a list of data loaders for phi and a respectively"""
        phi_set = []
        a_set = []
        for data in datasets:
            phi_loader, a_loader = self.get_data_loaders(data)
            phi_set.append(phi_loader)
            a_set.append(a_loader)
        return phi_set, a_set

    def get_data(self, data_menu: list, can_inspect_data: bool=False) -> tuple[list[DataLoader], list[DataLoader]]:
        """Main API of DataFactory"""
        datasets = self.prepare_datasets(data_menu, can_inspect_data)
        phi_set, a_set = self.prepare_loadersets(datasets)
        return phi_set, a_set


class SimpleDataFactory(TrainingDataFactory):
    def __init__(self, input_label_map_file: str) -> None:
        super().__init__(input_label_map_file)

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "bemt_infused_multihead_simple_net", "data_factory_config.yaml"])
        return path

    def convert_sim_to_training_data(self, input_data: np.ndarray, label_data: np.ndarray, file: str) -> SimpleDataset:
        # normalize data before putting it into the dataset
        result = SimpleDataset(input_data,   # input normalization will be done in the network
                               (label_data - self.label_mean_vector)*self.label_scale_vector,
                               self.input_mean_vector,
                               self.input_scale_vector,
                               self.label_mean_vector,
                               self.label_scale_vector,
                               file)
        return result

    def prepare_datasets(self, data_menu: list, can_inspect_data: bool=False) -> list[SimpleDataset]:
        """Prepare the datasets from the data menu
        Each file in the data menu is a different condition, the length of the dataset is the number of conditions"""
        datasets = []
        for file in data_menu:
            input_data, label_data = self.load_sim_data(file)
            if can_inspect_data:
                TrainingDataFactory.plot_dataset_distribution_grid(input_data, self.input_headers, file + " input")  # inspect the data
                TrainingDataFactory.plot_dataset_distribution_grid(label_data, self.label_headers, file + " label")  # inspect the data
            dataset = self.convert_sim_to_training_data(input_data, label_data, file)
            datasets.append(dataset)
        return datasets
    
    def get_data_loaders(self, training_data: SimpleDataset) -> DataLoader:
        """Generate a data loader for the dataset"""
        length = int(len(training_data)*self.config["data_usage_ratio"])
        print(f"Using {length} samples from the dataset for training", f"from source file: {training_data.source_file}")
        subset = Subset(training_data, range(length))   # use only the beginning subset of the data for training
        loader = torch.utils.data.DataLoader(subset, batch_size=self.config["simple_shot"], shuffle=self.config["can_shuffle"])
        return loader

    def prepare_loaderset(self, datasets: list[SimpleDataset]) -> list[DataLoader]:
        """Generate a list of data loaders"""
        loaderset = []
        for data in datasets:
            single_loader = self.get_data_loaders(data)
            loaderset.append(single_loader)
        return loaderset

    def get_data(self, data_menu: list, can_inspect_data: bool=False) -> list[DataLoader]:
        """Main API of DataFactory"""
        datasets = self.prepare_datasets(data_menu, can_inspect_data)
        loaderset = self.prepare_loaderset(datasets)
        return loaderset
    

class RotorNetDataFactory(TrainingDataFactory):
    def __init__(self, input_label_map_file: str) -> None:
        self.input_label_map = TrainingDataFactory.get_map(input_label_map_file)
        self.shared_input_headers = TrainingDataFactory.extract_header_from_input_label_map(self.input_label_map["shared_input"])
        self.individual_input_headers = []
        for member in self.input_label_map["individual_input"]:
            self.individual_input_headers.append(TrainingDataFactory.extract_header_from_input_label_map(member))

        self.label_headers = TrainingDataFactory.extract_header_from_input_label_map(self.input_label_map["label"])
        self.input_normalization, self.label_normalization = self.initialize_normalization_dict()
        self.normalization_params_file_path = TrainingDataFactory.find_path_to_normalization_params_file(input_label_map_file)
        self.config = TrainingDataFactory.load_config(self.get_default_config_path())     

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        path = common_utils.file_manager.find_path_to_folder(["learning", "config", "bemt_rotor_net", "data_factory_config.yaml"])
        return path
    
    def convert_sim_to_training_data(self, shared_input_data: np.ndarray, individual_input_data: list[np.ndarray], label_data: np.ndarray, file: str) -> SimpleDataset:
        # normalize data before putting it into the dataset
        result = RotorNetDataset(shared_input_data,   # input normalization will be done in the network
                                 individual_input_data,
                               (label_data - self.label_mean_vector)*self.label_scale_vector,
                               self.input_mean_vector,
                               self.input_scale_vector,
                               self.label_mean_vector,
                               self.label_scale_vector,
                               file)
        return result

    def load_sim_data(self, file_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the csv file and find the columns of interest."""
        input_columns, label_columns = TrainingDataFactory.find_input_label_column(self.input_label_map)
        df = DataFactory.make_data_frame_from_csv(file_name, input_columns + label_columns)
        shared_input_data = TrainingDataFactory.extract_data_from_data_frame(self.input_label_map["shared_input"], df)
        individual_input_data = []
        for member in self.input_label_map["individual_input"]:
            individual_input_data.append(TrainingDataFactory.extract_data_from_data_frame(member, df))
        label_data = TrainingDataFactory.extract_data_from_data_frame(self.input_label_map["label"], df)
        return shared_input_data, individual_input_data, label_data

    def prepare_datasets(self, data_menu: list, can_inspect_data: bool=False) -> list[RotorNetDataset]:
        """Prepare the datasets from the data menu
        Each file in the data menu is a different condition, the length of the dataset is the number of conditions"""
        datasets = []
        for file in data_menu:
            shared_input_data, individual_input_data, label_data = self.load_sim_data(file)
            if can_inspect_data:
                TrainingDataFactory.plot_dataset_distribution_grid(shared_input_data, self.shared_input_headers, file + " input")  # inspect the data
                TrainingDataFactory.plot_dataset_distribution_grid(label_data, self.label_headers, file + " label")  # inspect the data
            dataset = self.convert_sim_to_training_data(shared_input_data, individual_input_data, label_data, file)
            datasets.append(dataset)
        return datasets
    
    def get_data_loaders(self, training_data: RotorNetDataset) -> DataLoader:
        """Generate a data loader for the dataset"""
        length = int(len(training_data)*self.config["data_usage_ratio"])
        print(f"Using {length} samples from the dataset for training", f"from source file: {training_data.source_file}")
        subset = Subset(training_data, range(length))   # use only the beginning subset of the data for training
        loader = torch.utils.data.DataLoader(subset, batch_size=self.config["simple_shot"], shuffle=self.config["can_shuffle"])
        return loader

    def prepare_loaderset(self, datasets: list[RotorNetDataset]) -> list[DataLoader]:
        """Generate a list of data loaders"""
        loaderset = []
        for data in datasets:
            single_loader = self.get_data_loaders(data)
            loaderset.append(single_loader)
        return loaderset

    def get_data(self, data_menu: list, can_inspect_data: bool=False) -> list[DataLoader]:
        """Main API of DataFactory"""
        datasets = self.prepare_datasets(data_menu, can_inspect_data)
        loaderset = self.prepare_loaderset(datasets)
        return loaderset
    

def generate_data_list(subfolder, file_extension=".csv") -> list[str]:
    """
    Generate a list of data files in the specified subfolder.
    """
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training", subfolder)
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Filter out files that do not end with the specified file extension
    file_names = [f for f in file_names if f.endswith(file_extension)]
    # add subfolder to the file names
    data_list = [os.path.join(subfolder, f) for f in file_names]
    return data_list

