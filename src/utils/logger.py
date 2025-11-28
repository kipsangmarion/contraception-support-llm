"""
Logging infrastructure for the AI Contraception Counseling System.
Uses loguru for structured logging with file rotation and retention.
"""

import sys
from pathlib import Path
from loguru import logger
import yaml


def setup_logger(config_path: str = "configs/config.yaml") -> logger:
    """
    Set up the application logger with configuration from config.yaml.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configured logger instance
    """
    # Load configuration
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        log_config = config.get('logging', {})
    else:
        # Default configuration if config file doesn't exist
        log_config = {
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',
            'file': 'results/logs/app.log',
            'rotation': '100 MB',
            'retention': '30 days'
        }

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_config.get('format'),
        level=log_config.get('level', 'INFO'),
        colorize=True
    )

    # Create logs directory if it doesn't exist
    log_file = Path(log_config.get('file', 'results/logs/app.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Add file handler with rotation
    logger.add(
        log_file,
        format=log_config.get('format'),
        level=log_config.get('level', 'INFO'),
        rotation=log_config.get('rotation', '100 MB'),
        retention=log_config.get('retention', '30 days'),
        compression="zip",
        enqueue=True  # Thread-safe logging
    )

    logger.info("Logger initialized successfully")
    return logger


def get_logger(name: str = None) -> logger:
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional name to bind to the logger (e.g., module name)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Example usage functions for different log levels
def log_experiment_start(experiment_name: str, config: dict):
    """Log the start of an experiment with its configuration."""
    logger.info(f"Starting experiment: {experiment_name}")
    logger.debug(f"Experiment configuration: {config}")


def log_experiment_end(experiment_name: str, results: dict):
    """Log the end of an experiment with its results."""
    logger.info(f"Completed experiment: {experiment_name}")
    logger.debug(f"Experiment results: {results}")


def log_model_inference(model_name: str, input_text: str, output_text: str, latency: float):
    """Log model inference details."""
    logger.debug(
        f"Model inference | Model: {model_name} | "
        f"Input length: {len(input_text)} | "
        f"Output length: {len(output_text)} | "
        f"Latency: {latency:.3f}s"
    )


def log_error(error_type: str, error_message: str, context: dict = None):
    """Log an error with context."""
    logger.error(f"{error_type}: {error_message}")
    if context:
        logger.error(f"Error context: {context}")


# Initialize logger on module import
app_logger = setup_logger()
