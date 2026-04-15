from dataclasses import dataclass


@dataclass
class LearningStrategyConfiguration:
    name: str
    parameters: dict = None
