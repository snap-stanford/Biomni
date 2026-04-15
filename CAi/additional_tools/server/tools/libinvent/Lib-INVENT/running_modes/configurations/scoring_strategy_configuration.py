from dataclasses import dataclass

from reinvent_scoring import ScoringFuncionParameters

from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from running_modes.configurations import ReactionFilterConfiguration


@dataclass
class ScoringStrategyConfiguration:
    reaction_filter: ReactionFilterConfiguration
    diversity_filter: DiversityFilterParameters
    scoring_function: ScoringFuncionParameters
    name: str
