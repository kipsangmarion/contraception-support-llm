"""
Reproducibility utilities for setting random seeds and ensuring deterministic behavior.
"""

import random
import numpy as np
import os
import yaml
from pathlib import Path
from typing import Optional


def set_random_seeds(
    python_seed: Optional[int] = None,
    numpy_seed: Optional[int] = None,
    torch_seed: Optional[int] = None,
    config_path: str = "configs/config.yaml"
) -> dict:
    """
    Set random seeds for reproducibility across different libraries.

    Args:
        python_seed: Seed for Python's random module
        numpy_seed: Seed for NumPy
        torch_seed: Seed for PyTorch (if installed)
        config_path: Path to config file containing default seeds

    Returns:
        Dictionary of seeds that were set
    """
    seeds = {}

    # Load seeds from config if not provided
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        seed_config = config.get('random_seeds', {})
        python_seed = python_seed or seed_config.get('python', 42)
        numpy_seed = numpy_seed or seed_config.get('numpy', 42)
        torch_seed = torch_seed or seed_config.get('torch', 42)
    else:
        # Default seeds
        python_seed = python_seed or 42
        numpy_seed = numpy_seed or 42
        torch_seed = torch_seed or 42

    # Set Python random seed
    random.seed(python_seed)
    seeds['python'] = python_seed

    # Set NumPy seed
    np.random.seed(numpy_seed)
    seeds['numpy'] = numpy_seed

    # Set environment variable for hash seed (for deterministic hashing)
    os.environ['PYTHONHASHSEED'] = str(python_seed)

    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)
            torch.cuda.manual_seed_all(torch_seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seeds['torch'] = torch_seed
    except ImportError:
        pass  # PyTorch not installed

    return seeds


def get_environment_info() -> dict:
    """
    Collect environment information for reproducibility documentation.

    Returns:
        Dictionary containing environment details
    """
    import platform
    import sys

    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }

    # Add package versions
    try:
        import numpy
        env_info['numpy_version'] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        env_info['pandas_version'] = pandas.__version__
    except ImportError:
        pass

    try:
        import torch
        env_info['torch_version'] = torch.__version__
    except ImportError:
        pass

    try:
        import openai
        env_info['openai_version'] = openai.__version__
    except ImportError:
        pass

    try:
        import langchain
        env_info['langchain_version'] = langchain.__version__
    except ImportError:
        pass

    try:
        import faiss
        env_info['faiss_version'] = faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'
    except ImportError:
        pass

    return env_info


def save_experiment_config(
    experiment_name: str,
    config: dict,
    seeds: dict,
    output_dir: str = "results/logs"
) -> None:
    """
    Save experiment configuration and environment info for reproducibility.

    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration dictionary
        seeds: Random seeds used
        output_dir: Directory to save the config
    """
    from datetime import datetime
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'configuration': config,
        'random_seeds': seeds,
        'environment': get_environment_info()
    }

    # Save as JSON
    config_file = output_path / f"{experiment_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)

    return str(config_file)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Initialize random seeds on module import
if __name__ != "__main__":
    seeds = set_random_seeds()
