"""Central logging setup.

Every module gets its own logger with ``logging.getLogger(__name__)``, so all of them
live under the ``anomaly_detector`` namespace. This module configures that namespace
once, either from the CLI flags or from Python code::

    from anomaly_detector.logging_config import setup_logging
    setup_logging("DEBUG", log_file="output/run.log")

Level resolution order (first match wins):
  1. the ``level`` argument (CLI: ``--log-level``, ``-v`` or ``-q``)
  2. the ``ANOMALY_DETECTOR_LOG_LEVEL`` environment variable
  3. ``WARNING``

The library itself never configures logging on import: it only attaches a
``NullHandler`` so applications that embed it stay in full control.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER_NAME = "anomaly_detector"
ENV_VAR = "ANOMALY_DETECTOR_LOG_LEVEL"
DEFAULT_LEVEL = "WARNING"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Marks handlers created here so repeated setup_logging() calls replace them
# instead of stacking duplicates.
_HANDLER_FLAG = "_anomaly_detector_handler"


def resolve_level(level: str | int | None = None) -> int:
    """Turn a level name/number (or the env var, or the default) into a logging int."""
    if level is None:
        level = os.environ.get(ENV_VAR, DEFAULT_LEVEL)
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    if name not in LEVELS:
        raise ValueError(f"invalid log level {level!r}; choose from: {', '.join(LEVELS)}")
    return getattr(logging, name)


def setup_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
    fmt: str = CONSOLE_FORMAT,
) -> logging.Logger:
    """Configure the package logger to write to stderr and, optionally, a file.

    Returns the configured ``anomaly_detector`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolve_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    setattr(console, _HANDLER_FLAG, True)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_FLAG, True)
        logger.addHandler(file_handler)

    logger.debug(
        "logging configured: level=%s file=%s", logging.getLevelName(logger.level), log_file
    )
    return logger
