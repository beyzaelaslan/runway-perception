"""YAML config yükleyici.

Hiperparametreler koda gömülmez, configs/*.yaml içinde tutulur (CLAUDE.md kuralı).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML config dosyasını sözlük olarak yükler."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
