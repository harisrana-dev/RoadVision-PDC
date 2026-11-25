"""Image/target transforms shared by CPU and GPU experiments.

Kept deliberately minimal for v1 so the controlled comparison (Experiment D)
has exactly one preprocessing path. Any change here must be re-validated
across devices.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as F


class Compose:
    """Simple (image, target) -> (image, target) composition."""

    def __init__(self, transforms: list[Any]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """PIL image -> float tensor in [0, 1]; boxes stay as float xyxy."""

    def __call__(self, image: Image.Image, target: dict[str, torch.Tensor]):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Flip image and boxes together with probability ``prob``."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: dict[str, torch.Tensor]):
        if random.random() >= self.prob:
            return image, target
        image = image.flip(-1)
        boxes = target["boxes"]
        if boxes.numel():
            _, width = image.shape[-2], image.shape[-1]
            # boxes are xyxy: x1' = W - x2, x2' = W - x1
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes = boxes.clone()
            boxes[:, 0] = x1
            boxes[:, 2] = x2
            target = {**target, "boxes": boxes}
        return image, target


def build_transforms(config: dict[str, Any], *, train: bool) -> Compose:
    """Build the train or eval transform pipeline from config.

    Normalization is intentionally *not* applied here: torchvision SSD
    reference training normalizes inside the model's transform (and the
    ImageNet backbone weights expect raw [0,1] input handled by the model's
    internal ``ImageList`` transform). Keeping normalization out of the
    dataset avoids double-normalization bugs.
    """
    steps: list[Any] = [ToTensor()]
    if train and float(config["preprocessing"].get("hflip_prob", 0.0)) > 0:
        steps.append(RandomHorizontalFlip(float(config["preprocessing"]["hflip_prob"])))
    return Compose(steps)
