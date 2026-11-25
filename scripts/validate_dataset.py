"""Validate an RDD2022 data root: indexing, splits, stats, optional visuals.

Refuses to run when no data is present (dataset is approval-gated).

Usage:
    python scripts/validate_dataset.py --config configs/experiment.yaml
    python scripts/validate_dataset.py --samples 8   # also save overlay images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadvision.config import load_config  # noqa: E402
from roadvision.data.dataset import load_split, make_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--samples", type=int, default=0, help="Save N annotated sample images."
    )
    parser.add_argument("--out", default="results/visualizations")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["dataset"]["root"])
    # .gitkeep counts as "present" — require an actual country subdirectory.
    has_data = root.exists() and any(p.is_dir() for p in root.iterdir())
    if not has_data:
        print(
            f"Dataset root '{root}' has no country data.\n"
            "Dataset is approval-gated: record approval in "
            "docs/DATASET_APPROVAL.md, set dataset.approved=true and "
            "DATASET_APPROVED=true, then run:\n"
            "  python -m roadvision.data.download --countries Czech India Japan"
        )
        return 2

    # Build full index + split manifest (also asserts no leakage).
    splits, manifest = make_splits(root, config.to_dict())
    print("Split summary")
    print(f"  seed            : {manifest.seed}")
    print(f"  train images    : {manifest.train_count}")
    print(f"  val images      : {manifest.val_count}")
    print(f"  countries (train): {manifest.country_distribution.get('train')}")
    print(f"  countries (val)  : {manifest.country_distribution.get('val')}")
    print(f"  excluded        : {len(manifest.excluded)}")
    for item in manifest.excluded[:20]:
        print(f"    - {item['path']}: {item['reason']}")
    if len(manifest.excluded) > 20:
        print(f"    ... {len(manifest.excluded) - 20} more")

    manifest_path = Path("data/processed/split_manifest.json")
    manifest.save(manifest_path)
    print(f"Manifest written: {manifest_path}")

    # Per-split dataset construction exercises the VOC parser end-to-end.
    for split in ("train", "val"):
        ds, _ = load_split(root, config.to_dict(), split)
        print(f"  dataset[{split}] length = {len(ds)}")
        if len(ds) == 0:
            print(f"ERROR: split '{split}' is empty.")
            return 1

    if args.samples > 0:
        _save_samples(config, args.samples, Path(args.out))

    print("Validation OK.")
    return 0


def _save_samples(config, n: int, out_dir: Path) -> None:
    """Render bounding boxes on N validation images and save PNGs."""
    import random

    from PIL import Image, ImageDraw

    from roadvision.data.dataset import load_split

    ds, _ = load_split(config["dataset"]["root"], config.to_dict(), "val")
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = list(range(len(ds)))
    random.Random(config["seed"]).shuffle(idxs)
    colors = {1: "#E6194B", 2: "#3CB44B", 3: "#4363D8", 4: "#F58231"}
    names = {1: "D00", 2: "D10", 3: "D20", 4: "D40"}

    saved = 0
    for i in idxs:
        if saved >= n:
            break
        rec = ds.records[i]
        image = Image.open(rec.image_path).convert("RGB")
        w, h = image.size
        # Re-parse boxes in ORIGINAL coordinates for correct overlay.
        from roadvision.data.annotations import convert_annotation, parse_voc_xml

        ann, _issues = parse_voc_xml(rec.xml_path)
        if ann is None:
            continue
        boxes, labels, _ = convert_annotation(ann, ds.classes)  # original frame
        draw = ImageDraw.Draw(image)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            color = colors.get(label, "#000000")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), names.get(label, str(label)), fill=color)
        out_path = out_dir / f"val_{rec.rel_path.replace('/', '__')}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        saved += 1
        print(f"  sample saved: {out_path}  (orig {w}x{h})")

    # Also dump raw split stats as JSON for the report.
    stats = {
        "train": len(ds),
        "samples_saved": saved,
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
