
class Adapter:
    """Encapsulates a factory instance and provides a uniform interface to get the package for training manager"""
    def __init__(self) -> None:
        self.implementation = None

    def set_up(self, specs):
        raise NotImplementedError("Adapter subclasses must implement set_up_factory method")

    def generate_artifacts(self) -> object:
        raise NotImplementedError("Adapter subclasses must implement generate_artifacts method")
