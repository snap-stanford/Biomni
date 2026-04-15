from dataclasses import dataclass


@dataclass
class ReactionFilterConfiguration:
    type: str
    reactions: dict[str, list[str]]
