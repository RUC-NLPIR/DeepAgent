"""Logging utilities for ADK"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


_loggers = {}


def setup_logging(
    log_dir: str = "logs",
    system_log_level: str = "INFO",
    eval_log_level: str = "DEBUG",
    console_output: bool = True
) -> tuple[logging.Logger, logging.Logger]:
    """
    Set up dual logging system for ADK

    Args:
        log_dir: Directory for log files
        system_log_level: Log level for system logger
        eval_log_level: Log level for evaluation logger
        console_output: Whether to output to console

    Returns:
        Tuple of (system_logger, eval_logger)
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # System logger
    system_logger = logging.getLogger("adk.system")
    system_logger.setLevel(getattr(logging, system_log_level.upper()))
    system_logger.handlers.clear()

    # System log file handler
    system_file = log_path / f"adk_system_{datetime.now().strftime('%Y%m%d')}.log"
    system_fh = RotatingFileHandler(
        system_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    system_fh.setLevel(getattr(logging, system_log_level.upper()))
    system_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    system_fh.setFormatter(system_formatter)
    system_logger.addHandler(system_fh)

    # Evaluation logger
    eval_logger = logging.getLogger("adk.evaluation")
    eval_logger.setLevel(getattr(logging, eval_log_level.upper()))
    eval_logger.handlers.clear()

    # Evaluation log file handler
    eval_file = log_path / f"adk_evaluation_{datetime.now().strftime('%Y%m%d')}.log"
    eval_fh = RotatingFileHandler(
        eval_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    eval_fh.setLevel(getattr(logging, eval_log_level.upper()))
    eval_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
    )
    eval_fh.setFormatter(eval_formatter)
    eval_logger.addHandler(eval_fh)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        system_logger.addHandler(console_handler)

    # Cache loggers
    _loggers['system'] = system_logger
    _loggers['evaluation'] = eval_logger

    return system_logger, eval_logger


def get_logger(logger_type: str = "system") -> logging.Logger:
    """
    Get a logger instance

    Args:
        logger_type: Type of logger ("system" or "evaluation")

    Returns:
        Logger instance
    """
    if logger_type not in _loggers:
        # Initialize if not already set up
        setup_logging()

    return _loggers.get(logger_type, logging.getLogger(f"adk.{logger_type}"))
