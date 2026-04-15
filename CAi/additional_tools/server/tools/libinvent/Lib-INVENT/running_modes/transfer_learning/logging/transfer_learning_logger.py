from running_modes.transfer_learning.logging.local_transfer_learning_logger import LocalTransferLearningLogger

from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class TransferLearningLogger:
    def __new__(cls, logging_path: str, weights: bool=False) -> BaseTransferLearningLogger:

        return LocalTransferLearningLogger(logging_path, weights)
