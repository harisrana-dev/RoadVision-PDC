"""Detector factory for RoadVision-PDC.

Initial candidate (NOT permanently frozen — see docs/EXPERIMENT_DESIGN.md §7):
    torchvision.models.detection.ssdlite320_mobilenet_v3_large

Head configuration (documented deliberately)
--------------------------------------------
torchvision's ``num_classes`` for detection models **includes the
background class**. RDD2022 has 4 damage classes, so we build the head with
``num_classes = 5`` (background=0, D00=1, D10=2, D20=3, D40=4).

The full COCO-pretrained detector weights were trained with 91 classes and
cannot be loaded into a 5-class head. We therefore use the standard transfer
learning recipe:

* ``weights=None``                       -> task head freshly initialized
* ``weights_backbone=IMAGENET1K_V1``     -> MobileNetV3-Large backbone
                                            pretrained on ImageNet

This is the configuration every CPU/GPU experiment shares.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.mobilenet import MobileNet_V3_Large_Weights

logger = logging.getLogger(__name__)

BACKBONE_WEIGHTS = {
    "IMAGENET1K_V1": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    "IMAGENET1K_V2": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    "NONE": None,
}


def build_detector(config: dict[str, Any]) -> nn.Module:
    """Construct the detection model from the ``model:`` config section.

    Works identically on CPU and CUDA — no device logic here; the caller
    moves the model with ``model.to(device)``.
    """
    model_cfg = config["model"]
    name = model_cfg.get("name", "ssdlite320_mobilenet_v3_large")
    if name != "ssdlite320_mobilenet_v3_large":
        raise ValueError(
            f"Unknown model {name!r}. Only ssdlite320_mobilenet_v3_large is "
            "implemented in v1 (candidate list in docs/EXPERIMENT_DESIGN.md §7)."
        )

    num_classes = int(model_cfg["num_classes"])
    backbone_key = str(model_cfg.get("weights_backbone", "IMAGENET1K_V1")).upper()
    if backbone_key not in BACKBONE_WEIGHTS:
        raise ValueError(
            f"Unknown weights_backbone {backbone_key!r}; "
            f"choose one of {sorted(BACKBONE_WEIGHTS)}"
        )
    backbone_weights = BACKBONE_WEIGHTS[backbone_key]

    if model_cfg.get("weights") not in (None, "null"):
        raise ValueError(
            "model.weights must be null: full COCO detector weights are "
            "incompatible with a 5-class head (see module docstring)."
        )

    trainable_layers = model_cfg.get("trainable_backbone_layers")
    kwargs: dict[str, Any] = {
        "weights": None,  # fresh detection head
        "num_classes": num_classes,
        "weights_backbone": backbone_weights,
    }
    # Only meaningful when backbone weights are actually loaded (torchvision
    # warns otherwise); keep the config value for pretrained runs.
    if trainable_layers is not None and backbone_weights is not None:
        kwargs["trainable_backbone_layers"] = int(trainable_layers)

    model = ssdlite320_mobilenet_v3_large(**kwargs)
    logger.info(
        "Built SSDLite320/MobileNetV3-Large num_classes=%d backbone=%s "
        "(head freshly initialized, backbone=%s)",
        num_classes,
        backbone_key,
        "ImageNet-pretrained" if backbone_weights else "random",
    )
    return model


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Total (or trainable) parameter count — used by feasibility reports."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def make_synthetic_batch(
    batch_size: int = 2,
    num_boxes: int = 3,
    image_size: int = 320,
    num_classes: int = 5,
    *,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Create a synthetic (images, targets) batch for smoke tests.

    Images are random tensors in [0, 1]; boxes are valid xyxy within bounds.
    No dataset required — keeps tests hermetic.
    """
    g = torch.Generator().manual_seed(seed)
    images: list[torch.Tensor] = []
    targets: list[dict[str, torch.Tensor]] = []
    for i in range(batch_size):
        img = torch.rand(3, image_size, image_size, generator=g)
        n = num_boxes
        # Random but valid boxes: x1<x2, y1<y2 within the image.
        xs = torch.rand(n, 2, generator=g) * (image_size - 1)
        ys = torch.rand(n, 2, generator=g) * (image_size - 1)
        x1, x2 = xs.min(dim=1).values, xs.max(dim=1).values
        y1, y2 = ys.min(dim=1).values, ys.max(dim=1).values
        # enforce >=2 px to avoid degenerate boxes
        x2 = torch.clamp(x2 + 2.0, max=float(image_size - 1))
        y2 = torch.clamp(y2 + 2.0, max=float(image_size - 1))
        boxes = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)
        labels = torch.randint(1, num_classes, (n,), generator=g)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        targets.append(
            {
                "boxes": boxes,
                "labels": labels.to(torch.int64),
                "image_id": torch.tensor([i], dtype=torch.int64),
                "area": area,
                "iscrowd": torch.zeros((n,), dtype=torch.int64),
            }
        )
        images.append(img)
    return images, targets
