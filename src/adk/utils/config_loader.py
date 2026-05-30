"""Configuration loader for ADK"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load ADK configuration from YAML file

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Configuration dictionary
    """
    global _config_cache

    # Load environment variables
    load_dotenv()

    if config_path is None:
        # Default to config/adk_config.yaml in the adk directory
        adk_dir = Path(__file__).parent.parent
        config_path = adk_dir / "config" / "adk_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Replace environment variables in config
    config = _replace_env_vars(config)

    _config_cache = config
    return config


def _replace_env_vars(obj: Any) -> Any:
    """Recursively replace ${VAR} with environment variables"""
    if isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.getenv(var_name, obj)
    else:
        return obj


def get_config() -> Dict[str, Any]:
    """
    Get cached configuration. Loads if not already loaded.

    Returns:
        Configuration dictionary
    """
    global _config_cache
    if _config_cache is None:
        return load_config()
    return _config_cache


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a specific config value using dot notation

    Args:
        key_path: Dot-separated path (e.g., "loop_a.enabled")
        default: Default value if key not found

    Returns:
        Configuration value

    Example:
        >>> get_config_value("models.reasoning_model.model_name")
        "gpt-4"
    """
    config = get_config()
    keys = key_path.split(".")

    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
