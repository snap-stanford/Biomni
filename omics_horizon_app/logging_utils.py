"\"\"\"Logging utilities shared by Omics Horizon components.\"\"\""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_file_logger(logger_name: str, log_filename: str) -> logging.Logger:
    """Configure and return a rotating file logger safe across reruns."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:  # pragma: no cover
        repo_root = Path.cwd()

    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / log_filename

    existing = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", None) == str(log_path)
    ]
    if not existing:
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger(logger_name)
