from dataclasses import dataclass


@dataclass
class TuplesLikelihoodComputationConfiguration:
    input_csv_path: str
    output_csv_path: str
    model_path: str
    batch_size: int = 128
    use_gzip: bool = False