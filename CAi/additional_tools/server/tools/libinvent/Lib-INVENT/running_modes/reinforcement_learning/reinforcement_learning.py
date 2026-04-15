import time
from typing import List
import numpy as np
import torch

from models.model import DecoratorModel
from models.rl_actions import LikelihoodEvaluation, SampleModel
from running_modes.configurations.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from running_modes.dto import SampledSequencesDTO
from running_modes.reinforcement_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.reinforcement_learning.logging import BaseReinforcementLogger
from running_modes.reinforcement_learning.scoring_strategy.scoring_strategy import ScoringStrategy


class ReinforcementLearning:

    def __init__(self, critic: DecoratorModel, actor: DecoratorModel, configuration: ReinforcementLearningConfiguration,
                 logger: BaseReinforcementLogger):
        self.critic = critic
        self.actor = actor
        self.configuration = self._double_single_scaffold_hack(configuration)
        self.logger = logger
        optimizer = torch.optim.Adam(self.actor.network.parameters(), lr=self.configuration.learning_rate)
        self.learning_strategy = LearningStrategy(self.critic, optimizer, self.configuration.learning_strategy,
                                                  self.logger)
        self.scoring_strategy = ScoringStrategy(self.configuration.scoring_strategy, self.logger)

    def _double_single_scaffold_hack(self, configuration: ReinforcementLearningConfiguration) -> ReinforcementLearningConfiguration:
        # trick to address the problem that the model requires a list of scaffolds.
        if len(configuration.scaffolds) == 1:
            configuration.scaffolds *= 2
            configuration.batch_size = max(int(configuration.batch_size/2), 1)
        return configuration

    def run(self):
        start_time = time.time()
        for step in range(self.configuration.n_steps):
            # 1. Sampling
            sampled_sequences = self._sampling()
            # 2. Scoring
            score_summary = self._scoring(sampled_sequences, step)
            # 3. Updating
            actor_nlls, critic_nlls, augmented_nlls = self._updating(sampled_sequences, score_summary.total_score)
            # 4. Logging
            self._logging(start_time, step, score_summary, actor_nlls, critic_nlls, augmented_nlls)
        self.scoring_strategy.save_filter_memory()

    def _sampling(self):
        sampling_action = SampleModel(self.actor, self.configuration.batch_size, self.logger,
                                      self.configuration.randomize_scaffolds)
        sampled_sequences = sampling_action.run(self.configuration.scaffolds)
        return sampled_sequences

    def _scoring(self, sampled_sequences, step: int):
        return self.scoring_strategy.evaluate(sampled_sequences, step)

    def _updating(self, sampled_sequences, score):
        scaffold_batch, decorator_batch, actor_nlls = self._calculate_likelihood(sampled_sequences)
        actor_nlls, critic_nlls, augmented_nlls = self.learning_strategy.run(scaffold_batch, decorator_batch, score, actor_nlls)
        return actor_nlls, critic_nlls, augmented_nlls

    def _logging(self, start_time, step, score_summary, actor_nlls, critic_nlls, augmented_nlls):
        self.logger.timestep_report(start_time, self.configuration.n_steps, step, score_summary, actor_nlls,
                                    critic_nlls, augmented_nlls, self.scoring_strategy.diversity_filter)

    def _calculate_likelihood(self, sampled_sequences: List[SampledSequencesDTO]):
        nll_calculation_action = LikelihoodEvaluation(self.actor, self.configuration.batch_size, self.logger)
        encoded_scaffold, encoded_decorators, nlls = nll_calculation_action.run(sampled_sequences)
        return encoded_scaffold, encoded_decorators, nlls
