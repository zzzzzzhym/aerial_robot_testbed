from pipeline_adapters.base import Adapter
from pipeline_adapters.artifacts import DataFactoryArtifacts, ModelFactoryArtifacts, ValidatorArtifacts

import model
import data_factory
import trainer
import validator

class DaimlDataFactoryAdapter(Adapter):
    def set_up(self, sample_data_menu: list, input_label_map_file: str) -> None:
        """The sample data menu is a subset of training data to inspect number of conditions and do normalization"""
        print("Setting up data factory...")
        self.implementation = data_factory.DaimlDataFactory(
            input_label_map_file
        )
        self.implementation.set_num_of_conditions(sample_data_menu)
        self.implementation.make_normalization_params(sample_data_menu)

    def generate_artifacts(self, data_menu: list, can_inspect_data: bool=False) -> DataFactoryArtifacts:
        artifacts = DataFactoryArtifacts()
        artifacts.datasets = self.implementation.prepare_datasets(data_menu, can_inspect_data)
        artifacts.loaderset_phi, artifacts.loaderset_a = self.implementation.prepare_loadersets(artifacts.datasets)
        artifacts.dim_of_input = len(self.implementation.input_headers)
        artifacts.dim_of_label = len(self.implementation.label_headers)
        artifacts.num_of_conditions = self.implementation.num_of_conditions
        artifacts.input_mean_vector = self.implementation.input_mean_vector
        artifacts.input_scale_vector = self.implementation.input_scale_vector
        artifacts.label_mean_vector = self.implementation.label_mean_vector
        artifacts.label_scale_vector = self.implementation.label_scale_vector
        artifacts.input_label_map = self.implementation.input_label_map
        return artifacts

class DaimlModelFactoryAdapter(Adapter):
    def set_up(self, artifacts: DataFactoryArtifacts) -> None:
        print("Setting up model factory...")
        self.implementation = model.DaimlModelFactory(
            artifacts.num_of_conditions,
            artifacts.dim_of_input,
            artifacts.input_mean_vector,
            artifacts.input_scale_vector,
            artifacts.label_mean_vector,
            artifacts.label_scale_vector,
        )

    def generate_artifacts(self) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.phi_net, artifacts.h_net = self.implementation.generate_nets()
        artifacts.config = self.implementation.generate_self_config()
        return artifacts

class DaimlTrainerAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts, artifacts_validator: ValidatorArtifacts) -> None:
        print("Setting up trainer...")
        self.implementation = trainer.DaimlTrainer(
            artifacts_model.phi_net,
            artifacts_model.h_net,
            artifacts_data.loaderset_phi,
            artifacts_data.loaderset_a,
            artifacts_data.dim_of_label,
            artifacts_validator.in_run_validate
        )

    def train(self):
        self.implementation.train_model()
        self.implementation.plot_loss()

    def generate_results(self):
        """Trigger to train the model"""
        self.implementation.plot_loss()
        self.implementation.plot_tsne_of_a_trace()  # only available when is_dynamic_environment is True

class DaimlModelSaver:
    def set_up(self, artifacts_model: ModelFactoryArtifacts, artifacts_data: DataFactoryArtifacts) -> None:
        self.phi_net = artifacts_model.phi_net
        self.h_net = artifacts_model.h_net
        self.model_factory_config = artifacts_model.config
        self.input_label_map = artifacts_data.input_label_map

    def save_model(self, name) -> None:
        model.save_daiml_model(name, self.phi_net, self.h_net, self.model_factory_config, self.input_label_map)

class DaimlModelLoader:
    def generate_artifacts(self, name) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.phi_net, artifacts.h_net = model.load_daiml_model(name)
        return artifacts
        
class DaimlValidatorAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts) -> None:
        print("Setting up validator...")
        self.validator_instance = validator.DaimlEvaluator()
        self.validator_instance.load_model(artifacts_model.phi_net, artifacts_model.h_net)
        self.validator_instance.load_dataset(artifacts_data.datasets)

    def generate_artifacts(self) -> ValidatorArtifacts:
        artifacts = ValidatorArtifacts()
        artifacts.in_run_validate = self.validator_instance.callback_validation
        return artifacts
    
    def test_model(self) -> None:
        self.validator_instance.test_model()