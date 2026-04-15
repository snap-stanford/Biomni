from copy import deepcopy
from typing import List

import numpy as np

from diversity_filters.base_diversity_filter import BaseDiversityFilter
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
# The import below is a deal breaker
# from reinvent_scoring.scoring.score_summary import FinalSummary
from running_modes.dto import SampledSequencesDTO


class NoFilter(BaseDiversityFilter):
    """Doesn't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary, sampled_sequences: List[SampledSequencesDTO], step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.minscore:
                smile = score_summary.scored_smiles[i]
                decorations = f'{sampled_sequences[i].scaffold}|{sampled_sequences[i].decoration}'
                self._add_to_memory(i, scores[i], smile, decorations, score_summary.scaffold_log, step)
        return scores
