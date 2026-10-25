from pipeline_adapters.base import Adapter
from pipeline_adapters.artifacts import DataFactoryArtifacts, ModelFactoryArtifacts, ValidatorArtifacts

import sequence_model
import data_factory
import trainer
import validator


class SequenceDataFactoryAdapter(Adapter):
    def set_up(self, sample_data_menu: list, input_label_map_file: str) -> None:
        print("Setting up sequence data factory...")
        self.implementation = data_factory.SequenceDataFactory(input_label_map_file)
        self.implementation.make_normalization_params(sample_data_menu)

    def generate_artifacts(self, data_menu: list, can_inspect_data: bool = False) -> DataFactoryArtifacts:
        artifacts = DataFactoryArtifacts()
        artifacts.datasets = self.implementation.prepare_datasets(data_menu, can_inspect_data)
        artifacts.train_loader = self.implementation.prepare_train_loader(artifacts.datasets)
        artifacts.window = self.implementation.window
        artifacts.dim_of_input = len(self.implementation.input_headers)   # feat_dim per frame
        artifacts.dim_of_label = len(self.implementation.label_headers)
        artifacts.input_mean_vector = self.implementation.input_mean_vector
        artifacts.input_scale_vector = self.implementation.input_scale_vector
        artifacts.label_mean_vector = self.implementation.label_mean_vector
        artifacts.label_scale_vector = self.implementation.label_scale_vector
        artifacts.input_label_map = self.implementation.input_label_map
        return artifacts


class SequenceModelFactoryAdapter(Adapter):
    def set_up(self, artifacts: DataFactoryArtifacts) -> None:
        print("Setting up sequence model factory...")
        self.implementation = sequence_model.SequenceNetFactory(
            artifacts.dim_of_input,
            artifacts.window,
            artifacts.dim_of_label,
            artifacts.input_mean_vector,
            artifacts.input_scale_vector,
            artifacts.label_mean_vector,
            artifacts.label_scale_vector,
        )

    def generate_artifacts(self) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.sequence_net = self.implementation.generate_nets()
        artifacts.config = self.implementation.generate_self_config()
        return artifacts


class SequenceTrainerAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts, artifacts_validator: ValidatorArtifacts) -> None:
        print("Setting up sequence trainer...")
        self.implementation = trainer.SequenceTrainer(
            artifacts_model.sequence_net,
            artifacts_data.train_loader,
            artifacts_data.dim_of_label,
            artifacts_validator.in_run_validate,
            "trainer_config.yaml",
        )

    def train(self):
        self.implementation.train_model()
        self.implementation.plot_loss()

    def generate_results(self):
        self.implementation.plot_loss()


class SequenceModelSaver:
    def set_up(self, artifacts_model: ModelFactoryArtifacts, artifacts_data: DataFactoryArtifacts) -> None:
        self.sequence_net = artifacts_model.sequence_net
        self.model_factory_config = artifacts_model.config
        self.input_label_map = artifacts_data.input_label_map

    def save_model(self, name) -> None:
        sequence_model.save_sequence_model(name, self.sequence_net, self.model_factory_config, self.input_label_map)


class SequenceModelLoader:
    def generate_artifacts(self, name) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.sequence_net = sequence_model.load_sequence_model(name)
        return artifacts


class SequenceValidatorAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts) -> None:
        print("Setting up sequence validator...")
        self.validator_instance = validator.SequenceEvaluator()
        self.validator_instance.load_model(artifacts_model.sequence_net)
        self.validator_instance.load_dataset(artifacts_data.datasets)

    def generate_artifacts(self) -> ValidatorArtifacts:
        artifacts = ValidatorArtifacts()
        artifacts.in_run_validate = self.validator_instance.callback_validation
        return artifacts

    def test_model(self) -> None:
        self.validator_instance.test_model()
