"""Caricamento file di configurazione."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: Path | str) -> Dict[str, Any]:
    """Legge un file YAML e restituisce un dizionario."""
    with open(path, encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg
