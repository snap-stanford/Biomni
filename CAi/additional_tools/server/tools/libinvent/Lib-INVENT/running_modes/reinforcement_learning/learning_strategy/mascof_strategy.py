import torch
from torch.autograd import Variable

from running_modes.configurations import LearningStrategyConfiguration
from running_modes.reinforcement_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy


class MASCOFStrategy(BaseLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)


    def _calculate_loss(self, scaffold_batch, decorator_batch, score, actor_nlls):

        critic_nlls = self.critic_model.likelihood(*scaffold_batch, *decorator_batch)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -actor_nlls
        augmented_nlls = self._to_tensor(score)

        loss = -torch.sum(self._to_tensor(score)) * torch.sum(negative_actor_nlls)
        return loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls