"""Configuration loading for RoadVision-PDC.

A single YAML file (configs/experiment.yaml) drives every experiment.
Runtime code must never fork behaviour based on device except through the
explicit `device` argument passed to the engine/profiler.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels above this file (src/roadvision/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


class Config:
    """Thin attribute-style wrapper around the YAML mapping."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def raw(self) -> dict[str, Any]:
        """Return the underlying dict (do not mutate in place)."""
        return self._data

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Config({self._data!r})"


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the experiment YAML.

    Raises FileNotFoundError / ValueError with actionable messages.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. Expected configs/experiment.yaml."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    _validate(data)
    return Config(data)


def _validate(data: dict[str, Any]) -> None:
    """Fail fast on configs that would produce incomparable experiments."""
    required_top = ["seed", "dataset", "preprocessing", "model", "training", "dataloader"]
    missing = [k for k in required_top if k not in data]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    ds = data["dataset"]
    for key in ("name", "root", "split", "classes", "class_to_idx"):
        if key not in ds:
            raise ValueError(f"dataset.{key} is required in the config.")

    split = ds["split"]
    for key in ("seed", "train_fraction", "val_fraction"):
        if key not in split:
            raise ValueError(f"dataset.split.{key} is required in the config.")
    total = float(split["train_fraction"]) + float(split["val_fraction"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"dataset.split fractions must sum to 1.0, got {total}."
        )

    model = data["model"]
    if int(model.get("num_classes", 0)) < 2:
        raise ValueError("model.num_classes must include background (>= 2).")

    n_classes_cfg = len(ds["classes"])
    num_classes = int(model["num_classes"])
    if num_classes != n_classes_cfg + 1:
        raise ValueError(
            f"model.num_classes={num_classes} must equal len(dataset.classes)+1 "
            f"({n_classes_cfg}+1) so background is accounted for."
        )

    train = data["training"]
    for key in ("optimizer", "lr", "epochs", "batch_size"):
        if key not in train:
            raise ValueError(f"training.{key} is required in the config.")


def dataset_approved(config: Config | None = None) -> bool:
    """Return True only when BOTH the config flag and env flag opt in.

    The downloader requires an explicit double opt-in so a stale env var or
    an edited config alone cannot trigger a multi-GB download by accident.
    """
    env_flag = os.environ.get("DATASET_APPROVED", "false").strip().lower()
    env_ok = env_flag in {"1", "true", "yes"}
    cfg_ok = False
    if config is not None:
        cfg_ok = bool(config.get("dataset", {}).get("approved", False))
    # Default config ships approved=false; require both signals true.
    return env_ok and cfg_ok


def config_hash(config: Config) -> str:
    """Stable SHA-256 of the config contents (for run manifests)."""
    payload = yaml.safe_dump(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
