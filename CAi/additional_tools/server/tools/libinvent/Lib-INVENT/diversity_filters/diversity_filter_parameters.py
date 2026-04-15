from dataclasses import dataclass


@dataclass
class DiversityFilterParameters:
    name: str
    minscore: float = 0.4
    bucket_size: int = 25
    minsimilarity: float = 0.4
