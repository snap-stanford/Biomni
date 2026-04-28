from abc import ABC, abstractmethod


class BaseReactionFilter(ABC):
    @abstractmethod
    def evaluate(self, molecule):
        raise NotImplementedError("The method 'evaluate' is not implemented!")
