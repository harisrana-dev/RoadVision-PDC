"""Tests for dataset indexing, splits, leakage guards, and loading.

All fixtures are synthetic — no RDD2022 download required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from roadvision.data.dataset import (
    RoadDamageDataset,
    SampleRecord,
    assign_split,
    build_splits,
    index_dataset,
    load_split,
    make_splits,
)


def make_country(
    root: Path, country: str, n_images: int, *, start_id: int = 0
) -> list[SampleRecord]:
    """Create a fake country tree with JPEGImages/Annotations."""
    img_dir = root / country / "train" / "JPEGImages"
    xml_dir = root / country / "train" / "Annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    records: list[SampleRecord] = []
    for i in range(n_images):
        idx = start_id + i
        name = f"img_{idx:04d}"
        # Unique pixel content per image (content-hash guard).
        img = Image.new("RGB", (64, 48), color=(idx % 256, (idx * 3) % 256, (idx * 7) % 256))
        img_path = img_dir / f"{name}.jpg"
        img.save(img_path)

        xml_path = xml_dir / f"{name}.xml"
        xml_path.write_text(
            f"""<annotation>
  <filename>{name}.jpg</filename>
  <size><width>64</width><height>48</height><depth>3</depth></size>
  <object>
    <name>D00</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>40</xmax><ymax>30</ymax></bndbox>
  </object>
</annotation>""",
            encoding="utf-8",
        )
        records.append(
            SampleRecord(
                image_path=img_path,
                xml_path=xml_path,
                country=country,
                rel_path=str(img_path.relative_to(root)).replace("\\", "/"),
            )
        )
    return records


@pytest.fixture()
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    make_country(root, "Czech", 12)
    make_country(root, "India", 8, start_id=100)
    make_country(root, "Japan", 6, start_id=200)
    return root


@pytest.fixture()
def minimal_config(fake_root: Path) -> dict:
    return {
        "seed": 42,
        "dataset": {
            "name": "RDD2022",
            "root": str(fake_root),
            "countries": ["Czech", "India", "Japan"],
            "split": {"seed": 42, "train_fraction": 0.8, "val_fraction": 0.2},
            "classes": ["D00", "D10", "D20", "D40"],
            "class_to_idx": {"D00": 1, "D10": 2, "D20": 3, "D40": 4},
        },
        "preprocessing": {
            "resize": [320, 320],
            "hflip_prob": 0.5,
        },
        "model": {"name": "ssdlite320_mobilenet_v3_large", "num_classes": 5},
        "training": {"optimizer": "sgd", "lr": 0.005, "epochs": 1, "batch_size": 2},
        "dataloader": {"num_workers": 0, "pin_memory": False},
    }


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_index_dataset_finds_pairs(fake_root: Path):
    records, exclusions = index_dataset(fake_root, ["Czech", "India", "Japan"])
    assert len(records) == 12 + 8 + 6
    assert exclusions == []
    countries = {r.country for r in records}
    assert countries == {"Czech", "India", "Japan"}


def test_index_dataset_missing_country_logged(fake_root: Path):
    records, exclusions = index_dataset(fake_root, ["Czech", "Norway"])
    assert len(records) == 12
    assert any("Norway" in e["reason"] for e in exclusions)


def test_index_dataset_missing_root():
    with pytest.raises(FileNotFoundError):
        index_dataset(Path("/nonexistent/dataset/root"))


# ---------------------------------------------------------------------------
# Split determinism & leakage
# ---------------------------------------------------------------------------

def test_assign_split_deterministic():
    a = assign_split("Czech/train/JPEGImages/x.jpg", seed=42, train_fraction=0.8)
    b = assign_split("Czech/train/JPEGImages/x.jpg", seed=42, train_fraction=0.8)
    assert a == b
    # Different seed may differ (not guaranteed for one path, but must not crash)
    assign_split("Czech/train/JPEGImages/x.jpg", seed=7, train_fraction=0.8)


def test_splits_no_overlap_and_complete(fake_root: Path):
    records, _ = index_dataset(fake_root, ["Czech", "India", "Japan"])
    splits, manifest = build_splits(
        records, seed=42, train_fraction=0.8, root=fake_root
    )
    train_p = {r.rel_path for r in splits["train"]}
    val_p = {r.rel_path for r in splits["val"]}
    assert not (train_p & val_p), "leakage: same path in both splits"
    assert len(train_p) + len(val_p) == len(records)
    assert manifest.train_count == len(splits["train"])
    assert manifest.val_count == len(splits["val"])


def test_splits_content_duplicate_guards_leakage(tmp_path: Path):
    root = tmp_path / "raw"
    make_country(root, "Czech", 6)
    # Add a byte-identical copy of the first image under a different name.
    src_imgs = sorted((root / "Czech/train/JPEGImages").glob("*.jpg"))
    src_xmls = sorted((root / "Czech/train/Annotations").glob("*.xml"))
    dup_img = root / "Czech/train/JPEGImages/zz_duplicate.jpg"
    dup_xml = root / "Czech/train/Annotations/zz_duplicate.xml"
    dup_img.write_bytes(src_imgs[0].read_bytes())  # identical bytes
    dup_xml.write_text(src_xmls[0].read_text(encoding="utf-8"), encoding="utf-8")

    records, _ = index_dataset(root, ["Czech"])
    assert len(records) == 7
    splits, manifest = build_splits(
        records, seed=42, train_fraction=0.8, root=root, content_hash=True
    )
    # Duplicate must be excluded, not split across train/val.
    assert len(manifest.content_duplicates) == 1
    total = len(splits["train"]) + len(splits["val"])
    assert total == 6  # one excluded as content duplicate
    assert manifest.content_duplicates[0].endswith("zz_duplicate.jpg")


def test_manifest_records_seed_counts_countries(fake_root: Path, minimal_config: dict):
    splits, manifest = make_splits(fake_root, minimal_config)
    assert manifest.seed == 42
    assert manifest.train_count + manifest.val_count == 26
    # Country distribution recorded for both splits
    for key in ("train", "val"):
        dist = manifest.country_distribution[key]
        assert set(dist) <= {"Czech", "India", "Japan"}
        assert sum(dist.values()) == (
            manifest.train_count if key == "train" else manifest.val_count
        )
    # Round-trip through JSON
    out = fake_root.parent / "split_manifest.json"
    manifest.save(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["seed"] == 42
    assert data["train_count"] == manifest.train_count


def test_same_seed_same_split(fake_root: Path):
    records, _ = index_dataset(fake_root, ["Czech", "India", "Japan"])
    s1, _ = build_splits(records, seed=42, train_fraction=0.8, root=fake_root)
    s2, _ = build_splits(records, seed=42, train_fraction=0.8, root=fake_root)
    assert [r.rel_path for r in s1["train"]] == [r.rel_path for r in s2["train"]]
    assert [r.rel_path for r in s1["val"]] == [r.rel_path for r in s2["val"]]


# ---------------------------------------------------------------------------
# Dataset __getitem__
# ---------------------------------------------------------------------------

def test_dataset_getitem_contract(minimal_config: dict):
    from roadvision.data.preprocessing import build_transforms

    transforms = build_transforms(minimal_config, train=True)
    ds, manifest = load_split(
        minimal_config["dataset"]["root"], minimal_config, "train",
        transforms=transforms,
    )
    assert len(ds) == manifest.train_count
    image, target = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert image.dtype == torch.float32
    # After resize the tensor must be (3, 320, 320)
    assert tuple(image.shape[1:]) == (320, 320)

    for key in ("boxes", "labels", "image_id", "area", "iscrowd"):
        assert key in target, f"missing target key {key}"
    assert target["boxes"].shape[-1] == 4
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert (target["labels"] >= 1).all() and (target["labels"] <= 4).all()
    assert target["boxes"].shape[0] == target["labels"].shape[0]
    if target["boxes"].shape[0]:
        # Boxes must lie within the resized frame.
        assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
        assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()
        assert target["boxes"].max() <= 320
    # area consistency
    if target["boxes"].shape[0]:
        expected = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (
            target["boxes"][:, 3] - target["boxes"][:, 1]
        )
        assert torch.allclose(target["area"], expected, atol=1e-3)


def test_load_split_val_has_samples(minimal_config: dict):
    ds, manifest = load_split(
        minimal_config["dataset"]["root"], minimal_config, "val"
    )
    assert len(ds) == manifest.val_count
    assert manifest.val_count > 0  # 20% of 26 should not be empty


def test_load_split_rejects_bad_name(minimal_config: dict):
    with pytest.raises(ValueError):
        load_split(minimal_config["dataset"]["root"], minimal_config, "test")


def test_road_damage_dataset_direct_empty_ok():
    ds = RoadDamageDataset(records=[])
    assert len(ds) == 0
