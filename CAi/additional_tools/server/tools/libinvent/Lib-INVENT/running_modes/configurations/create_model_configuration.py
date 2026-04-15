from dataclasses import dataclass

@dataclass
class CreateModelConfiguration:
    input_smiles_path: str
    output_model_path: str
    num_layers: int = 3
    layer_size: int = 512
    embedding_layer_size: int = 256
    dropout: float = 0
    layer_normalisation: bool = False
    max_sequence_length: int = 256
