from abc import ABC, abstractmethod
import numpy


class BaseScaffoldDecoratingLogger(ABC):
    def __init__(self, logging_path: str):
        self._logging_path = logging_path

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def log_timestep(self, smiles: str, likelihoods: numpy.ndarray):
        raise NotImplementedError("log_timestep method is not implemented")

    def _get_unique_entries_fraction(self, some_list):
        return 100 * len(set(some_list)) / len(some_list)



