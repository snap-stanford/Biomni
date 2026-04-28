"""
base_CAi Logging Module

Centralized logging configuration for the entire project.
Provides structured logging with different levels, file rotation, and colored console output.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{self.BOLD}{record.levelname:8s}{self.RESET}"
        return super().format(record)


class CAiLogger:
    """
    Centralized logger for base_CAi project.

    Features:
    - Colored console output
    - File logging with rotation
    - Separate log files for different components
    - Configurable log levels
    - Performance tracking
    """

    _instances = {}
    _initialized = False

    def __init__(
        self,
        name: str = "base_CAi",
        log_dir: str | None = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """
        Initialize logger.

        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files (default: ./logs)
            console_level: Console logging level
            file_level: File logging level
            max_bytes: Max size of each log file before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console output
            enable_file: Enable file output
        """
        self.name = name
        self.logger = logging.getLogger(name)

        # Prevent duplicate handlers
        if name in CAiLogger._instances:
            return

        # Set base level to DEBUG to allow all messages through
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Setup log directory
        if log_dir is None:
            log_dir = os.getenv("BASE_CAI_LOG_DIR", "./logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Console handler with colors
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level.upper()))

            # Use colored formatter for console
            console_format = "%(levelname)s | %(name)s | %(message)s"
            console_handler.setFormatter(ColoredFormatter(console_format))
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if enable_file:
            log_file = self.log_dir / f"{name}.log"
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            file_handler.setLevel(getattr(logging, file_level.upper()))

            # Detailed format for file
            file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(file_handler)

        CAiLogger._instances[name] = self

    @classmethod
    def get_logger(cls, name: str = "base_CAi", **kwargs) -> logging.Logger:
        """
        Get or create a logger instance.

        Args:
            name: Logger name
            **kwargs: Additional arguments for logger initialization

        Returns:
            Logger instance
        """
        if name not in cls._instances:
            cls(name=name, **kwargs)
        return cls._instances[name].logger

    @classmethod
    def set_global_level(cls, level: str):
        """
        Set logging level for all loggers.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level_value = getattr(logging, level.upper())
        for instance in cls._instances.values():
            instance.logger.setLevel(level_value)
            for handler in instance.logger.handlers:
                handler.setLevel(level_value)

    @classmethod
    def disable_console(cls):
        """Disable console output for all loggers."""
        for instance in cls._instances.values():
            for handler in instance.logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                    instance.logger.removeHandler(handler)

    @classmethod
    def enable_debug_mode(cls):
        """Enable debug mode for all loggers."""
        cls.set_global_level("DEBUG")

    def debug(self, msg, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)


# Convenience function for quick logger access
def get_logger(name: str = "base_CAi", **kwargs) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__ of the module)
        **kwargs: Additional configuration options

    Returns:
        Logger instance

    Example:
        >>> from base_CAi.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    return CAiLogger.get_logger(name, **kwargs)


# Initialize default logger
_default_logger = None


def setup_default_logger(
    console_level: str = None,
    file_level: str = None,
    log_dir: str = None,
):
    """
    Setup the default logger with custom configuration.

    Args:
        console_level: Console logging level (from env or default INFO)
        file_level: File logging level (from env or default DEBUG)
        log_dir: Log directory (from env or default ./logs)
    """
    global _default_logger

    # Read from environment variables if not provided
    if console_level is None:
        console_level = os.getenv("BASE_CAI_LOG_LEVEL", "INFO")
    if file_level is None:
        file_level = os.getenv("BASE_CAI_FILE_LOG_LEVEL", "DEBUG")
    if log_dir is None:
        log_dir = os.getenv("BASE_CAI_LOG_DIR", "./logs")

    _default_logger = CAiLogger(
        name="base_CAi",
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
    )

    return _default_logger.logger


# Auto-initialize with defaults
if not _default_logger:
    setup_default_logger()
