from running_modes.configurations import ConfigurationEnvelope
from running_modes.scoring.logging.base_scoring_logger import BaseScoringLogger
from running_modes.enums.scoring_runner_enum import ScoringRunnerEnum


class LocalScoringLogger(BaseScoringLogger):
    def __init__(self, configuration: ConfigurationEnvelope):
        super().__init__(configuration)
        self._scoring_runner_enum = ScoringRunnerEnum()

    def log_message(self, message: str):
        self._logger.info(message)
