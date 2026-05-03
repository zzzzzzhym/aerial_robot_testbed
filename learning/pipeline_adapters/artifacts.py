
class DataFactoryArtifacts:
    def __init__(self) -> None:
        # DAIML specific
        self.loaderset_phi = None 
        self.loaderset_a = None
        self.num_of_conditions = None 
        # simple specific
        self.datasets = None
        self.loaderset = None
        # common
        self.dim_of_input = None 
        self.dim_of_label = None 
        self.input_mean_vector = None
        self.input_scale_vector = None
        self.label_mean_vector = None
        self.label_scale_vector = None
        self.input_label_map = None  # a dict mapping input and label names to column indices in data files

class ModelFactoryArtifacts:
    def __init__(self) -> None:
        # DAIML specific
        self.phi_net = None
        self.h_net = None
        # simple specific
        self.simple_net = None    
        # rotor net specific
        self.rotor_net = None
        # common
        self.config = None


class ValidatorArtifacts:
    def __init__(self) -> None:
        self.in_run_validate = None  # callable function for validation during training

