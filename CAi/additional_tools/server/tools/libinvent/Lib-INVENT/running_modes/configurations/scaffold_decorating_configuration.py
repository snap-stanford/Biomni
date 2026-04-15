from dataclasses import dataclass


@dataclass
class ScaffoldDecoratingConfiguration:
    model_path: str
    input_scaffold_path: str
    output_path: str
    logging_path: str
    batch_size: int = 128
    sample_uniquely: bool = True
    number_of_decorations_per_scaffold: int = 32
    randomize: bool = False

