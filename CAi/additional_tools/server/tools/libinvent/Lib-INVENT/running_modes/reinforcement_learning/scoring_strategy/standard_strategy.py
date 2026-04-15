from typing import List, Any

import numpy as np
from reinvent_scoring import FinalSummary, ScoringFunctionComponentNameEnum, LoggableComponent, ComponentParameters, \
    ComponentSummary

from running_modes.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration
from running_modes.dto import SampledSequencesDTO
from running_modes.reinforcement_learning.scoring_strategy.base_scoring_strategy import BaseScoringStrategy


class StandardScoringStrategy(BaseScoringStrategy):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, logger):
        super().__init__(strategy_configuration, logger)

    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step) -> FinalSummary:
        score_summary = self._apply_scoring_function(sampled_sequences)
        score_summary.total_score = self.diversity_filter.update_score(score_summary, sampled_sequences, step)
        return score_summary

    def _apply_scoring_function(self, sampled_sequences: List[SampledSequencesDTO]) -> FinalSummary:
        molecules = self._join_scaffolds_and_decorations(sampled_sequences)
        smiles = [self._conversion.mol_to_smiles(molecule) if molecule else "INVALID" for molecule in molecules]
        final_score: FinalSummary = self.scoring_function.get_final_score(smiles)
        final_score = self._apply_reaction_filters(molecules, final_score)
        return final_score

    def _apply_reaction_filters(self, molecules: List[Any], final_score: FinalSummary) -> FinalSummary:
        sf_component = ScoringFunctionComponentNameEnum()
        reaction_scores = [self.reaction_filter.evaluate(molecule) if molecule else 0 for molecule in molecules]
        loggable_component = LoggableComponent(name=sf_component.REACTION_FILTERS,
                                               component_type=sf_component.REACTION_FILTERS,
                                               score=reaction_scores)
        component_parameters = ComponentParameters(component_type=sf_component.REACTION_FILTERS,
                                                   name=sf_component.REACTION_FILTERS, weight=1, smiles=[],
                                                   model_path='')
        component_summary = ComponentSummary(total_score=reaction_scores, parameters=component_parameters)
        final_score.total_score = final_score.total_score * np.array(reaction_scores)
        final_score.scaffold_log.append(component_summary)
        final_score.profile.append(loggable_component)
        return final_score

    def _join_scaffolds_and_decorations(self, sampled_sequences: List[SampledSequencesDTO]):
        molecules = []
        for sample in sampled_sequences:
            scaffold = self._attachment_points.add_attachment_point_numbers(sample.scaffold, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, sample.decoration)
            molecules.append(molecule)
        return molecules
