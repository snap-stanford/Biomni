from running_modes.configurations import ConfigurationEnvelope
from running_modes.scoring.logging.local_scoring_logger import LocalScoringLogger
# from running_modes.scoring.logging.remote_scoring_logger import RemoteScoringLogger
# from running_modes.enums.logging_mode_enum import LoggingModeEnum


class ScoringLogger:

    def __new__(cls, configuration: ConfigurationEnvelope):
        logger = LocalScoringLogger(configuration)

        # else:
        #     logger = RemoteScoringLogger(configuration)
        return logger
