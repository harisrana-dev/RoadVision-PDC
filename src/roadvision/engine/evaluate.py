"""Detection-quality evaluation with pycocotools (COCO mAP).

Policy (docs/EXPERIMENT_DESIGN.md §6)
--------------------------------------
* pycocotools is the ONLY mAP implementation used for reported numbers.
* No pure-PyTorch mAP fallback is provided — an unvalidated fallback could
  silently report wrong quality metrics.
* ``verify_pycocotools()`` runs a known-answer round trip; CI and
  scripts/smoke_test.py call it so a broken install fails loudly.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# RDD2022 category ids match our label indices (background 0 is not a
# COCO category and never appears in targets).
DEFAULT_CATEGORIES = [
    {"id": 1, "name": "D00"},
    {"id": 2, "name": "D10"},
    {"id": 3, "name": "D20"},
    {"id": 4, "name": "D40"},
]


def verify_pycocotools() -> dict[str, Any]:
    """Known-answer test for the COCO evaluator.

    Builds a tiny ground-truth/prediction pair with a known mAP@0.5 and
    asserts the evaluator returns it. Raises RuntimeError on failure.

    Ground truth: one 100x100 image, box (10,10,60,60), class 1.
    Prediction  : identical box/score 0.9/class 1  -> IoU=1.0 -> AP=1.0.
                  plus a wrong-class box            -> counted as FP only.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 50.0],  # xywh
                "area": 2500.0,
                "iscrowd": 0,
            }
        ],
        "categories": DEFAULT_CATEGORIES,
    }
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 50.0, 50.0],
            "score": 0.9,
        }
    ]

    with tempfile.TemporaryDirectory() as td:
        gt_path = Path(td) / "gt.json"
        import json

        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        coco = COCO(str(gt_path))
        coco_dt = coco.loadRes(preds)  # pass list directly: a JSON *string*
        # would be treated as a file path by pycocotools (Errno 22 on Windows)
        ev = COCOeval(coco, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        map50 = float(ev.stats[1])  # AP@IoU=0.50

    if not np.isfinite(map50) or map50 < 0.999:
        raise RuntimeError(
            f"pycocotools known-answer check FAILED: AP@0.5={map50}, expected 1.0"
        )
    return {"ok": True, "map50": map50}


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    score_threshold: float = 0.05,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    """Run inference and convert predictions to COCO detection dicts."""
    model.eval()
    results: list[dict[str, Any]] = []
    seen = 0
    for images, targets in data_loader:
        if max_images is not None and seen >= max_images:
            break
        imgs = [img.to(device) for img in images]
        outputs = model(imgs)
        for out, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            scores = out["scores"].cpu()
            boxes = out["boxes"].cpu()
            labels = out["labels"].cpu()
            keep = scores >= score_threshold
            for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                x1, y1, x2, y2 = [float(v) for v in box]
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                        "score": float(score),
                    }
                )
            seen += 1
    return results


def build_coco_ground_truth(targets: list[dict[str, Any]], categories=None) -> dict[str, Any]:
    """Convert torchvision targets into a COCO ground-truth dict.

    ``targets`` must be CPU tensors with keys boxes/labels/image_id/area/
    iscrowd (our dataset contract).
    """
    cats = categories if categories is not None else DEFAULT_CATEGORIES
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    ann_id = 1
    seen_ids: set[int] = set()
    for target in targets:
        image_id = int(target["image_id"].item())
        boxes = target["boxes"]
        labels = target["labels"]
        area_t = target.get("area")
        crowd_t = target.get("iscrowd")
        if image_id not in seen_ids:
            # Width/height unknown at this layer; filled by caller if needed.
            # COCOeval requires them — caller must supply via images meta.
            seen_ids.add(image_id)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            area = (
                float(area_t[i])
                if area_t is not None
                else (x2 - x1) * (y2 - y1)
            )
            crowd = int(crowd_t[i]) if crowd_t is not None else 0
            anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": area,
                    "iscrowd": crowd,
                }
            )
            ann_id += 1
    return {"images": images, "annotations": anns, "categories": cats}


def evaluate_map50(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    score_threshold: float = 0.05,
    max_images: int | None = None,
    image_sizes: dict[int, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Full validation evaluation -> COCO mAP metrics.

    Returns dict with ``map50`` (AP@0.5), ``map50_95`` (AP@[.5:.95]),
    ``per_class_ap50``, and ``n_images`` / ``n_predictions``.

    ``image_sizes`` maps image_id -> (width, height). Required because
    COCO ground truth needs explicit dimensions; if not provided we query
    the dataset records (caller must pass sizes for correctness).
    """
    verify_pycocotools()  # fail loudly on broken installs

    import json

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Collect GT targets once (targets are on CPU from our dataset).
    all_targets: list[dict[str, Any]] = []
    for _imgs, targets in data_loader:
        all_targets.extend(targets)
    if image_sizes is None:
        raise ValueError(
            "image_sizes {image_id: (w,h)} is required for COCO ground truth."
        )

    gt = build_coco_ground_truth(all_targets)
    gt["images"] = [
        {"id": iid, "width": wh[0], "height": wh[1], "file_name": f"{iid}.jpg"}
        for iid, wh in sorted(image_sizes.items())
    ]
    if not gt["annotations"]:
        logger.warning("No ground-truth annotations in validation set.")
        return {
            "map50": float("nan"),
            "map50_95": float("nan"),
            "per_class_ap50": {},
            "n_images": len(image_sizes),
            "n_predictions": 0,
        }

    preds = predict_dataset(
        model, data_loader, device, score_threshold=score_threshold,
        max_images=max_images,
    )

    with tempfile.TemporaryDirectory() as td:
        gt_path = Path(td) / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        coco_gt = COCO(str(gt_path))
        if not preds:
            logger.warning("Model produced no predictions above threshold.")
            return {
                "map50": 0.0,
                "map50_95": 0.0,
                "per_class_ap50": {c["name"]: 0.0 for c in DEFAULT_CATEGORIES},
                "n_images": len(image_sizes),
                "n_predictions": 0,
            }
        coco_dt = coco_gt.loadRes(preds)  # list, not JSON string (see verify_)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    per_class: dict[str, float] = {}
    # precision array: [TxRxKxAxM] -> AP per class at IoU=0.5, area=all
    if ev.eval["precision"].size:
        # ev.params.catIds order
        for idx, cat_id in enumerate(ev.params.catIds):
            name = next(
                (c["name"] for c in DEFAULT_CATEGORIES if c["id"] == cat_id), str(cat_id)
            )
            p = ev.eval["precision"][:, :, idx, 0, 2]  # IoU=0.5, area=all, maxDet=100
            p = p[p > -1]
            per_class[name] = float(p.mean()) if p.size else float("nan")

    return {
        "map50": float(ev.stats[1]),
        "map50_95": float(ev.stats[0]),
        "per_class_ap50": per_class,
        "n_images": len(image_sizes),
        "n_predictions": len(preds),
    }
