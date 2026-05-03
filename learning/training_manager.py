from enum import Enum
import pipeline_adapters

class ModelArchitecture(Enum):
    DAIML = 0
    SIMPLE_NET = 1
    ROTOR_NET = 2

class TrainingPipeline:
    def __init__(
        self,
        data_adapter: pipeline_adapters.Adapter,
        model_adapter: pipeline_adapters.Adapter,
        trainer_adapter: pipeline_adapters.Adapter,
        validator_adapter: pipeline_adapters.Adapter,
        model_saver
    ) -> None:
        self.data_adapter = data_adapter
        self.model_adapter = model_adapter
        self.trainer_adapter = trainer_adapter
        self.validator_adapter = validator_adapter
        self.model_saver = model_saver

    def set_up(self, training_data_menu: list, validation_data_menu: list, input_label_map_file: str, can_inspect_data=False) -> None:
        self.data_adapter.set_up(training_data_menu, input_label_map_file)
        training_data_artifacts = self.data_adapter.generate_artifacts(training_data_menu, can_inspect_data)
        validation_data_artifacts = self.data_adapter.generate_artifacts(validation_data_menu, can_inspect_data)
        self.model_adapter.set_up(training_data_artifacts)
        model_artifacts = self.model_adapter.generate_artifacts()
        self.validator_adapter.set_up(validation_data_artifacts, model_artifacts)
        validator_artifacts = self.validator_adapter.generate_artifacts()
        self.trainer_adapter.set_up(training_data_artifacts, model_artifacts, validator_artifacts)
        self.model_saver.set_up(model_artifacts, training_data_artifacts)

    def train(self) -> None:
        self.trainer_adapter.train()

    def show_result_only(self) -> None:
        self.trainer_adapter.generate_results()

    def save_model(self, name):
        self.model_saver.save_model(name)


class TestPipeline:
    def __init__(
        self,
        data_adapter: pipeline_adapters.Adapter,
        validator_adapter: pipeline_adapters.Adapter,
        model_loader: pipeline_adapters.DaimlModelLoader
    ) -> None:
        self.data_adapter = data_adapter
        self.validator_adapter = validator_adapter
        self.model_loader = model_loader

    def set_up(self, data_menu: list, input_label_map_file: str, model_name: str) -> None:
        self.data_adapter.set_up(data_menu, input_label_map_file)
        data_artifacts = self.data_adapter.generate_artifacts(data_menu)
        model_artifacts = self.model_loader.generate_artifacts(model_name)
        self.validator_adapter.set_up(data_artifacts, model_artifacts)

    def test(self) -> None:
        self.validator_adapter.test_model()
        

class PipelineFactory:
    """An abstract factory selects and instruct the detailed factories to create parts"""
    PIPELINE_BUILDERS = {
        ModelArchitecture.DAIML: {
            "train": lambda: TrainingPipeline(
                pipeline_adapters.DaimlDataFactoryAdapter(),
                pipeline_adapters.DaimlModelFactoryAdapter(),
                pipeline_adapters.DaimlTrainerAdapter(),
                pipeline_adapters.DaimlValidatorAdapter(),
                pipeline_adapters.DaimlModelSaver(),
            ),
            "test": lambda: TestPipeline(
                pipeline_adapters.DaimlDataFactoryAdapter(),
                pipeline_adapters.DaimlValidatorAdapter(),
                pipeline_adapters.DaimlModelLoader(),
            ),
        },

        ModelArchitecture.SIMPLE_NET: {
            "train": lambda: TrainingPipeline(
                pipeline_adapters.SimpleDataFactoryAdapter(),
                pipeline_adapters.SimpleModelFactoryAdapter(),
                pipeline_adapters.SimpleTrainerAdapter(),
                pipeline_adapters.SimpleValidatorAdapter(),
                pipeline_adapters.SimpleModelSaver(),
            ),
            "test": lambda: TestPipeline(
                pipeline_adapters.SimpleDataFactoryAdapter(),
                pipeline_adapters.SimpleValidatorAdapter(),
                pipeline_adapters.SimpleModelLoader(),
            ),
        },

        ModelArchitecture.ROTOR_NET: {
            "train": lambda: TrainingPipeline(
                pipeline_adapters.RotorNetDataFactoryAdapter(),
                pipeline_adapters.RotorNetModelFactoryAdapter(),
                pipeline_adapters.RotorNetTrainerAdapter(),
                pipeline_adapters.RotorNetValidatorAdapter(),
                pipeline_adapters.RotorNetModelSaver(),
            ),
            "test": lambda: TestPipeline(
                pipeline_adapters.RotorNetDataFactoryAdapter(),
                pipeline_adapters.RotorNetValidatorAdapter(),
                pipeline_adapters.RotorNetModelLoader(),
            ),
        },
    }

    def __init__(self, model_type: ModelArchitecture) -> None:
        self.config = self.load_config(model_type)

    def load_config(self, model_type: ModelArchitecture) -> None:
        """Simple config for now, can load from a file later"""
        config = {}
        config["model_type"] = model_type
        return config

    def make_training_pipeline(self) -> TrainingPipeline:
        """API for users to get the training pipeline, the main job is to select the right adapters"""
        return self.PIPELINE_BUILDERS[self.config["model_type"]]["train"]()

    def make_test_pipeline(self) -> TestPipeline:
        """API for users to get the testing pipeline, the main job is to select the right adapters"""
        return self.PIPELINE_BUILDERS[self.config["model_type"]]["test"]()
