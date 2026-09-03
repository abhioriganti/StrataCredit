"""Configuration loader for StrataCredit.

All YAML config files in configs/ are loaded once and cached.
Access via the module-level singletons at the bottom of this file.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from stratacredit import CONFIG_DIR


@functools.cache
def load_yaml(path: Path) -> dict[str, Any]:
    """Load and cache a YAML file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Config:
    """Thin wrapper around a YAML dict with attribute-style access."""

    def __init__(self, path: Path) -> None:
        self._data: dict[str, Any] = load_yaml(path)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{name}'") from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get with default."""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data


# Module-level singletons — import these directly
data_config = Config(CONFIG_DIR / "data.yaml")
features_config = Config(CONFIG_DIR / "features.yaml")
models_config = Config(CONFIG_DIR / "models.yaml")
pool_config = Config(CONFIG_DIR / "pool_constraints.yaml")
