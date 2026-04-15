from abc import abstractmethod, ABC


class BaseReactionFilter(ABC):

    @abstractmethod
    def evaluate(self, molecule):
        raise NotImplementedError("The method 'evaluate' is not implemented!")
