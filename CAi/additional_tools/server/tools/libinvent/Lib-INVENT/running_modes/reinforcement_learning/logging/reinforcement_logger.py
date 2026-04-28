from running_modes.configurations.log_configuration import LogConfiguration
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.reinforcement_learning.logging import BaseReinforcementLogger, LocalReinforcementLogger


class ReinforcementLogger:
    def __new__(cls, configuration: LogConfiguration) -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()
        if configuration.recipient == logging_mode_enum.LOCAL:
            return LocalReinforcementLogger(configuration)
        else:
            raise NotImplementedError("Remote logging mode is not implemented yet !")
