"""Torchvision-compatible RDD2022 dataset with leakage-safe splits.

Design notes
------------
* Only **annotated** images are split. Official RDD2022 test images ship
  without ground truth and are never used for metrics.
* Splits are image-level, deterministic, and keyed by SHA-256 of the
  *relative path* — stable across machines and independent of directory
  enumeration order.
* A content-hash guard ensures the same image bytes (possibly stored under
  two names) cannot land in both splits.
* Every excluded sample is logged with a reason (see annotations.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from roadvision.data.annotations import (
    DEFAULT_CLASSES,
    convert_annotation,
    parse_voc_xml,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split utilities
# ---------------------------------------------------------------------------

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def assign_split(
    rel_path: str, *, seed: int, train_fraction: float
) -> str:
    """Deterministically assign ``rel_path`` to 'train' or 'val'.

    Uses SHA-256 of ``f"{seed}:{rel_path}"`` mapped to [0, 1) — reproducible
    on any machine, unaffected by listing order.
    """
    digest = _sha256_text(f"{seed}:{rel_path}")
    u = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if u < train_fraction else "val"


@dataclass
class SplitManifest:
    """Recorded evidence of how the dataset was split."""

    seed: int
    train_fraction: float
    val_fraction: float
    train_count: int
    val_count: int
    country_distribution: dict[str, dict[str, int]]
    excluded: list[dict[str, str]] = field(default_factory=list)
    content_duplicates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "country_distribution": self.country_distribution,
            "excluded": self.excluded,
            "content_duplicates": self.content_duplicates,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sample index construction
# ---------------------------------------------------------------------------

@dataclass
class SampleRecord:
    image_path: Path
    xml_path: Path
    country: str
    rel_path: str


def index_dataset(
    root: str | Path,
    countries: list[str] | None = None,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> tuple[list[SampleRecord], list[dict[str, str]]]:
    """Walk ``root`` for (image, xml) pairs under the selected countries.

    Expected RDD2022 country layout (train split has annotations)::

        <root>/<Country>/train/JPEGImages/*.jpg
        <root>/<Country>/train/Annotations/*.xml

    Falls back to a flatter layout if the country folder itself contains
    JPEGImages/Annotations. Returns (records, exclusions).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if countries is None:
        # Use every directory that looks like a country folder.
        countries = sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    records: list[SampleRecord] = []
    exclusions: list[dict[str, str]] = []

    for country in countries:
        country_dir = root / country
        if not country_dir.is_dir():
            exclusions.append(
                {"path": str(country_dir), "reason": f"country directory missing: {country}"}
            )
            continue

        # Prefer .../<country>/train/{JPEGImages,Annotations}; else direct.
        pairs: list[tuple[Path, Path]] = []
        for layout in (country_dir / "train", country_dir):
            img_dir = layout / "JPEGImages"
            xml_dir = layout / "Annotations"
            if img_dir.is_dir() and xml_dir.is_dir():
                for xml_path in sorted(xml_dir.glob("*.xml")):
                    # Match image by stem across common extensions.
                    image_path = None
                    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
                        cand = img_dir / (xml_path.stem + ext)
                        if cand.is_file():
                            image_path = cand
                            break
                    if image_path is None:
                        exclusions.append(
                            {
                                "path": str(xml_path),
                                "reason": "annotation has no matching image file",
                            }
                        )
                        continue
                    pairs.append((image_path, xml_path))
                break

        if not pairs:
            exclusions.append(
                {
                    "path": str(country_dir),
                    "reason": "no JPEGImages/Annotations layout found (or empty)",
                }
            )
            continue

        for image_path, xml_path in pairs:
            records.append(
                SampleRecord(
                    image_path=image_path,
                    xml_path=xml_path,
                    country=country,
                    rel_path=str(image_path.relative_to(root)).replace("\\", "/"),
                )
            )

    return records, exclusions


def build_splits(
    records: list[SampleRecord],
    *,
    seed: int,
    train_fraction: float,
    root: str | Path,
    content_hash: bool = True,
) -> tuple[dict[str, list[SampleRecord]], SplitManifest]:
    """Assign records to train/val with leakage guards.

    Guards
    ------
    1. filename overlap  — same rel_path cannot appear in both splits
       (guaranteed by construction: each record assigned exactly once);
    2. content hash      — identical file bytes under different names are
       grouped and kept in a single split (``content_hash=True``);
    3. duplicate rel_paths are dropped from the second occurrence and logged.
    """
    root = Path(root)
    seen_paths: set[str] = set()
    deduped: list[SampleRecord] = []
    excluded: list[dict[str, str]] = []
    for rec in records:
        if rec.rel_path in seen_paths:
            excluded.append({"path": rec.rel_path, "reason": "duplicate path in index"})
            continue
        seen_paths.add(rec.rel_path)
        deduped.append(rec)

    # Content-hash grouping: hash -> list of records
    groups: dict[str, list[SampleRecord]] = {}
    if content_hash:
        for rec in deduped:
            digest = _sha256_file(rec.image_path)
            groups.setdefault(digest, []).append(rec)
    else:
        for rec in deduped:
            groups.setdefault(_sha256_text(rec.rel_path), []).append(rec)

    splits: dict[str, list[SampleRecord]] = {"train": [], "val": []}
    content_duplicates: list[str] = []
    for digest, group in groups.items():
        # All records in a group share bytes -> assign group by first member.
        anchor = group[0]
        if len(group) > 1:
            for extra in group[1:]:
                content_duplicates.append(extra.rel_path)
                excluded.append(
                    {
                        "path": extra.rel_path,
                        "reason": f"content-duplicate of {anchor.rel_path}",
                    }
                )
        split = assign_split(
            anchor.rel_path, seed=seed, train_fraction=train_fraction
        )
        splits[split].append(anchor)

    # Deterministic ordering inside each split.
    for key in splits:
        splits[key].sort(key=lambda r: r.rel_path)

    # Leakage assertion (belt & braces).
    train_paths = {r.rel_path for r in splits["train"]}
    val_paths = {r.rel_path for r in splits["val"]}
    overlap = train_paths & val_paths
    if overlap:
        raise RuntimeError(f"Split leakage detected: {sorted(overlap)[:5]} ...")

    country_dist: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    for key, recs in splits.items():
        for r in recs:
            country_dist[key][r.country] += 1

    manifest = SplitManifest(
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=round(1.0 - train_fraction, 6),
        train_count=len(splits["train"]),
        val_count=len(splits["val"]),
        country_distribution={
            k: dict(sorted(v.items())) for k, v in country_dist.items()
        },
        excluded=excluded,
        content_duplicates=sorted(content_duplicates),
    )
    return splits, manifest


# ---------------------------------------------------------------------------
# Torchvision detection dataset
# ---------------------------------------------------------------------------

class RoadDamageDataset(Dataset):
    """RDD2022 detection dataset yielding ``(image, target)``.

    ``target`` keys: ``boxes``, ``labels``, ``image_id``, ``area``, ``iscrowd``
    — the contract expected by torchvision detection models.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        classes: tuple[str, ...] = DEFAULT_CLASSES,
        transforms: Callable | None = None,
        resize: tuple[int, int] | None = None,
        keep_excluded_log: bool = True,
    ):
        self.records = list(records)
        self.classes = tuple(classes)
        self.transforms = transforms
        self.resize = resize
        self.exclusions: list[dict[str, str]] = []
        self._keep_excluded_log = keep_excluded_log

    def __len__(self) -> int:
        return len(self.records)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: i + 1 for i, name in enumerate(self.classes)}  # 0 = background

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec = self.records[index]

        image = Image.open(rec.image_path).convert("RGB")
        orig_w, orig_h = image.size

        ann, xml_issues = parse_voc_xml(rec.xml_path)
        for issue in xml_issues:
            if self._keep_excluded_log:
                self.exclusions.append(
                    {"path": issue.path, "reason": issue.reason, "severity": issue.severity}
                )
            logger.log(
                logging.WARNING if issue.severity == "error" else logging.INFO,
                "%s",
                issue,
            )

        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        if ann is not None:
            # Parse-time validation happens in original coordinates; resize
            # scaling happens inside convert_annotation.
            boxes, labels, conv_issues = convert_annotation(
                ann, known_classes=self.classes, resize=self.resize
            )
            for issue in conv_issues:
                if self._keep_excluded_log:
                    self.exclusions.append(
                        {
                            "path": issue.path,
                            "reason": issue.reason,
                            "severity": issue.severity,
                        }
                    )
                logger.log(
                    logging.WARNING if issue.severity == "error" else logging.INFO,
                    "%s",
                    issue,
                )

        if self.resize is not None:
            new_w, new_h = self.resize
            if (orig_w, orig_h) != (new_w, new_h):
                image = image.resize((new_w, new_h), Image.BILINEAR)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        area = (
            (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            if boxes_t.numel()
            else torch.zeros(0, dtype=torch.float32)
        )
        target: dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def load_split(
    root: str | Path,
    config: dict[str, Any],
    split: str,
    *,
    transforms: Callable | None = None,
) -> tuple[RoadDamageDataset, SplitManifest]:
    """Convenience: index, split (seeded), and return one split's dataset.

    The full manifest (counts, countries, exclusions) is always returned so
    callers can persist it. ``split`` must be 'train' or 'val'.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    ds_cfg = config["dataset"]
    classes = tuple(ds_cfg["classes"])
    records, index_exclusions = index_dataset(
        ds_cfg["root"], ds_cfg.get("countries"), classes
    )
    split_cfg = ds_cfg["split"]
    splits, manifest = build_splits(
        records,
        seed=int(split_cfg["seed"]),
        train_fraction=float(split_cfg["train_fraction"]),
        root=ds_cfg["root"],
    )
    manifest.excluded = index_exclusions + manifest.excluded

    resize = tuple(config["preprocessing"]["resize"])  # (w, h)
    dataset = RoadDamageDataset(
        splits[split],
        classes=classes,
        transforms=transforms,
        resize=resize,
    )
    return dataset, manifest


def make_splits(
    root: str | Path, config: dict[str, Any]
) -> tuple[dict[str, list[SampleRecord]], SplitManifest]:
    """Public helper used by scripts to build and persist the manifest."""
    ds_cfg = config["dataset"]
    classes = tuple(ds_cfg["classes"])
    records, index_exclusions = index_dataset(
        ds_cfg["root"], ds_cfg.get("countries"), classes
    )
    split_cfg = ds_cfg["split"]
    splits, manifest = build_splits(
        records,
        seed=int(split_cfg["seed"]),
        train_fraction=float(split_cfg["train_fraction"]),
        root=ds_cfg["root"],
    )
    manifest.excluded = index_exclusions + manifest.excluded
    return splits, manifest


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - laptop has no CUDA
        torch.cuda.manual_seed_all(seed)
