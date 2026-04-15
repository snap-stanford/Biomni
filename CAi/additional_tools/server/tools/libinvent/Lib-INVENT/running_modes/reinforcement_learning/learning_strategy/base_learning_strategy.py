from abc import ABC, abstractmethod

import numpy as np
import torch

from running_modes.configurations import LearningStrategyConfiguration
from running_modes.enums import GenerativeModelRegimeEnum


class BaseLearningStrategy(ABC):
    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        self.critic_model = critic_model
        self.optimizer = optimizer
        self._configuration = configuration
        self._running_mode_enum = GenerativeModelRegimeEnum()
        self._logger = logger
        self._disable_prior_gradients()

    def log_message(self, message: str):
        self._logger.log_message(message)

    def run(self, scaffold_batch, decorator_batch, score, actor_nlls):
        loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
            self._calculate_loss(scaffold_batch, decorator_batch, score, actor_nlls)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return negative_actor_nlls, negative_critic_nlls, augmented_nlls

    @abstractmethod
    def _calculate_loss(self, scaffold_batch, decorator_batch, score, actor_nlls):
        raise NotImplementedError("_calculate_loss method is not implemented")

    def _to_tensor(self, tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor).cuda()
        return torch.autograd.Variable(tensor)

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        self.critic_model.set_mode(self._running_mode_enum.INFERENCE)
        for param in self.critic_model.network.parameters():
            param.requires_grad = False
