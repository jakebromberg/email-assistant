"""Centralized logging configuration with file rotation."""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("gmail", level="INFO")
        >>> logger.info("Processing emails...")
    """
    # Return existing logger if already configured
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # File handler (all levels, daily rotation, keep 30 days)
    log_file = log_path / f"{name}.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Cache logger
    _loggers[name] = logger

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("gmail.client")
        >>> logger.debug("Fetching messages...")
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)
