from dataclasses import dataclass

from reinvent_scoring import ScoringFuncionParameters

from running_modes.configurations import ReactionFilterConfiguration


@dataclass
class ScoringConfiguration:

    input: str
    output_folder: str
    reaction_filter: ReactionFilterConfiguration
    scoring_function: ScoringFuncionParameters
