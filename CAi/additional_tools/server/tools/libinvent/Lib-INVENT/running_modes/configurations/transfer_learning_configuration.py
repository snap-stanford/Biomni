from dataclasses import dataclass

from running_modes.configurations.nested_dataclass import nested_dataclass


@dataclass
class LearningRate:
    start: float = 0.0001
    min: float = 0.000001
    gamma: float = 0.95
    step: int = 1


@dataclass
class TransferLearningConfiguration:
    learning_rate: LearningRate
    model_path: str
    output_path: str
    training_set_path: str
    validation_sets_path: str
    logging_path: str
    decoration_type: str = "single"
    with_weights: bool = False
    sample_size: int = 5000
    save_frequency: int = 1
    epochs: int = 100
    batch_size: int = 128
    clip_gradients: float = 1.0
    collect_stats_frequency: int = 0
