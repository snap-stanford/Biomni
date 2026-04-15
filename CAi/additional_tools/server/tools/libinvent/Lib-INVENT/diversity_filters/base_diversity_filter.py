import abc
from typing import List

import numpy as np
import pandas as pd

from diversity_filters.diversity_filter_memory import DiversityFilterMemory
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
# from reinvent_scoring.scoring.score_summary import FinalSummary, ComponentSummary
from reinvent_chemistry.conversions import Conversions

from running_modes.dto import SampledSequencesDTO


class BaseDiversityFilter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, parameters: DiversityFilterParameters):
        self.parameters = parameters
        self._chemistry = Conversions()
        self._diversity_filter_memory = DiversityFilterMemory()

    @abc.abstractmethod
    def update_score(self, score_summary, sampled_sequences: List[SampledSequencesDTO], step=0) -> np.array:
        raise NotImplementedError("The method 'evaluate' is not implemented!")

    def get_memory_as_dataframe(self) -> pd.DataFrame:
        return self._diversity_filter_memory.get_memory()

    def set_memory_from_dataframe(self, memory: pd.DataFrame):
        self._diversity_filter_memory.set_memory(memory)

    def number_of_smiles_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def number_of_scaffold_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_scaffolds()

    def _calculate_scaffold(self, smile):
        raise NotImplementedError

    def _smiles_exists(self, smile):
        return self._diversity_filter_memory.smiles_exists(smile)

    def _add_to_memory(self, indx: int, score, smile, scaffold, components: List, step):
        self._diversity_filter_memory.update(indx, score, smile, scaffold, components, step)

    def _penalize_score(self, scaffold, score):
        """Penalizes the score if the scaffold bucket is full"""
        if self._diversity_filter_memory.scaffold_instances_count(scaffold) > self.parameters.bucket_size:
            score = 0.
        return score
