"""
logger_setup.py
----------------
Provides a single, rotating-file logger used across the whole
application. Addresses the Logging/Monitoring and Reliability
non-functional requirements.
"""

import logging
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid attaching duplicate handlers if called multiple times.
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # keep console clean

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
