from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.scaffold_decorating.logging.base_scaffold_decorating_logger import BaseScaffoldDecoratingLogger
from running_modes.scaffold_decorating.logging.local_scaffold_decorating_logger import LocalScaffoldDecoratingLogger


class ScaffoldDecoratingLogger:

    def __new__(cls, logging_path: str) -> BaseScaffoldDecoratingLogger:
        logging_mode_enum = LoggingModeEnum()
        return LocalScaffoldDecoratingLogger(logging_path)
