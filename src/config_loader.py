"""
Environment-based config loader for the movie-recs project.

Usage
-----
from src.config_loader import load_config

config = load_config()                 # uses APP_ENV or 'dev'
train_path = config["data"]["train_path"]
test_path = config["data"]["test_path"]
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  
except ImportError:  
    yaml = None


def _get_project_root() -> Path:
    """Return the project root (folder that contains src/)."""
    return Path(__file__).resolve().parents[1]


def _config_dir() -> Path:
    """Directory where config files live."""
    return _get_project_root() / "config"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            f"Attempted to load YAML config at {path}, but PyYAML is not installed. "
            "Install it (pip install pyyaml) or use a JSON config instead."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_from_files(env: str) -> Optional[Dict[str, Any]]:
    """
    Try to load config from:
      config/config.<env>.yaml
      config/config.<env>.yml
      config/config.<env>.json
    Returns dict if found, else None.
    """
    cfg_dir = _config_dir()

    yaml_candidates = [
        cfg_dir / f"config.{env}.yaml",
        cfg_dir / f"config.{env}.yml",
    ]
    for p in yaml_candidates:
        if p.exists():
            return _load_yaml(p)

    json_path = cfg_dir / f"config.{env}.json"
    if json_path.exists():
        return _load_json(json_path)

    return None


def _default_config(env: str) -> Dict[str, Any]:
    """
    Fallback config used when no file exists.
    Adjust paths/hyperparams here if needed.
    """
    root = _get_project_root()
    data_dir = root / "src" / "data"

    return {
        "env": env,
        "data": {
            "train_path": str(data_dir / "interactions_train.csv"),
            "test_path": str(data_dir / "interactions_test.csv"),
            "items_path": str(data_dir / "items.csv"),
        },
        "model": {
            "type": "als",
            "rank": 20,
            "reg": 0.1,
            "iterations": 10,
        },
        "eval": {
            "k": 10,
            "negatives_per_user": 99,
            "user_col": "userId",
            "item_col": "movieId",
        },
        "logging": {
            "model_dir": str(root / "models"),
            "logs_dir": str(root / "logs"),
        },
    }


def load_config(app_env: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration based on environment.

    Priority:
    1. app_env argument
    2. APP_ENV environment variable
    3. 'dev' (default)

    Then:
    - Try config/config.<env>.yaml|yml|json
    - If none found, return built-in default config.
    """
    if app_env is None:
        app_env = os.getenv("APP_ENV", "dev")

    cfg = _load_from_files(app_env)
    if cfg is None:
        cfg = _default_config(app_env)

    return cfg
