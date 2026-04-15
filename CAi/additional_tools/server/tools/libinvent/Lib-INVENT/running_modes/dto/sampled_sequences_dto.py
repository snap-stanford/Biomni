from dataclasses import dataclass


@dataclass
class SampledSequencesDTO:
    scaffold: str
    decoration: str
    nll: float
