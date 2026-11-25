"""Pascal VOC annotation parsing and validation for RDD2022.

Coordinate conventions
----------------------
Pascal VOC stores bounding boxes as **1-based, inclusive** integer pixel
coordinates ``(xmin, ymin, xmax, ymax)`` relative to the original image.

Torchvision detection models expect **0-based float** boxes in ``xyxy``
format relative to the *current* image tensor:

    x1 = xmin - 1.0
    y1 = ymin - 1.0
    x2 = xmax          # VOC's inclusive xmax becomes the (exclusive-style) x2
    y2 = ymax

After a resize from (old_w, old_h) to (new_w, new_h) boxes are scaled by
``(new_w / old_w, new_h / old_h)`` and clamped to the destination frame.

Every invalid sample is returned with an explicit ``reason`` so callers can
log exclusions — nothing is silently discarded and no coordinates are
invented.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# RDD2022 damage classes in canonical order (background is implicit index 0).
DEFAULT_CLASSES = ("D00", "D10", "D20", "D40")


@dataclass
class VOCObject:
    """A single annotated object inside an XML file."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: bool = False


@dataclass
class VOCAnnotation:
    """Parsed contents of one Pascal VOC XML file."""

    filename: str
    width: int
    height: int
    objects: list[VOCObject] = field(default_factory=list)

    @property
    def image_id(self) -> int:
        """Stable numeric id derived from the filename stem (0 if non-numeric)."""
        stem = Path(self.filename).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 0


@dataclass
class ParseIssue:
    """A validation problem found in an annotation file."""

    path: str
    reason: str
    severity: str = "error"  # "error" => exclude sample; "warn" => keep but log

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.reason}"


def parse_voc_xml(xml_path: str | Path) -> tuple[VOCAnnotation | None, list[ParseIssue]]:
    """Parse one VOC XML file.

    Returns ``(annotation, issues)``. ``annotation`` is None when the file is
    unusable (missing/unreadable/malformed XML, missing size, no filename).
    Issues always lists every problem encountered.
    """
    path = Path(xml_path)
    issues: list[ParseIssue] = []

    if not path.is_file():
        issues.append(ParseIssue(str(path), "XML file does not exist"))
        return None, issues

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as exc:
        issues.append(ParseIssue(str(path), f"malformed XML: {exc}"))
        return None, issues
    except OSError as exc:
        issues.append(ParseIssue(str(path), f"cannot read file: {exc}"))
        return None, issues

    if root.tag != "annotation":
        issues.append(
            ParseIssue(str(path), f"unexpected root tag <{root.tag}> (want <annotation>)")
        )
        return None, issues

    filename = (root.findtext("filename") or "").strip()
    if not filename:
        issues.append(ParseIssue(str(path), "missing <filename>"))
        return None, issues

    size = root.find("size")
    width = height = 0
    if size is None:
        issues.append(ParseIssue(str(path), "missing <size> block"))
        return None, issues
    try:
        width = int(float(size.findtext("width") or ""))
        height = int(float(size.findtext("height") or ""))
    except ValueError:
        issues.append(ParseIssue(str(path), "non-numeric <width>/<height>"))
        return None, issues
    if width <= 0 or height <= 0:
        issues.append(ParseIssue(str(path), f"invalid image size {width}x{height}"))
        return None, issues

    objects: list[VOCObject] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            issues.append(
                ParseIssue(str(path), f"object '{name or '?'}' missing <bndbox>", "warn")
            )
            continue
        try:
            xmin = float(bnd.findtext("xmin") or "")
            ymin = float(bnd.findtext("ymin") or "")
            xmax = float(bnd.findtext("xmax") or "")
            ymax = float(bnd.findtext("ymax") or "")
        except ValueError:
            issues.append(
                ParseIssue(
                    str(path), f"object '{name}' has non-numeric box coordinates", "warn"
                )
            )
            continue
        difficult = (obj.findtext("difficult") or "0").strip() in {"1", "true", "True"}
        objects.append(
            VOCObject(name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, difficult=difficult)
        )

    if not objects:
        issues.append(ParseIssue(str(path), "annotation contains zero objects", "warn"))

    return VOCAnnotation(filename=filename, width=width, height=height, objects=objects), issues


def voc_box_to_xyxy(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    """Convert a 1-based inclusive VOC box to 0-based float xyxy.

    See module docstring for the full rationale.
    """
    return (xmin - 1.0, ymin - 1.0, float(xmax), float(ymax))


def validate_box(
    box_xyxy: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    *,
    min_size: float = 1.0,
) -> str | None:
    """Return a reason string if the box is invalid, else None.

    A box is invalid when it is degenerate (non-positive area after
    conversion), non-finite, or entirely outside the image bounds.
    Partial out-of-bounds boxes are allowed (they get clamped later) but a
    box with no overlap with the image is rejected.
    """
    x1, y1, x2, y2 = box_xyxy
    for v in (x1, y1, x2, y2):
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
            return "non-finite coordinate"
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return f"degenerate box (w={x2 - x1:.2f}, h={y2 - y1:.2f})"
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return f"box [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] outside image {img_w}x{img_h}"
    return None


def clamp_box(
    box_xyxy: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Clamp a box into ``[0, w] x [0, h]``."""
    x1, y1, x2, y2 = box_xyxy
    return (
        max(0.0, min(float(img_w), x1)),
        max(0.0, min(float(img_h), y1)),
        max(0.0, min(float(img_w), x2)),
        max(0.0, min(float(img_h), y2)),
    )


def convert_annotation(
    ann: VOCAnnotation,
    known_classes: tuple[str, ...] = DEFAULT_CLASSES,
    *,
    resize: tuple[int, int] | None = None,
) -> tuple[list[tuple[float, float, float, float]], list[int], list[ParseIssue]]:
    """Convert a parsed VOC annotation into torch-ready boxes/labels.

    Parameters
    ----------
    ann:
        Parsed annotation (original image coordinates).
    known_classes:
        Accepted class names; anything else is an issue (not silently mapped).
    resize:
        Optional ``(new_w, new_h)`` — boxes are scaled and clamped to the
        resized frame (matches the preprocessing resize).

    Returns
    -------
    boxes (xyxy float), labels (int), issues.
    """
    issues: list[ParseIssue] = []
    class_to_idx = {name: idx + 1 for idx, name in enumerate(known_classes)}  # 1-based; 0 = background
    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []

    scale_x = scale_y = 1.0
    out_w, out_h = ann.width, ann.height
    if resize is not None:
        out_w, out_h = resize
        if ann.width <= 0 or ann.height <= 0:
            issues.append(ParseIssue(ann.filename, "cannot resize boxes: bad source size"))
            return [], [], issues
        scale_x = out_w / ann.width
        scale_y = out_h / ann.height

    for i, obj in enumerate(ann.objects):
        if obj.name not in class_to_idx:
            issues.append(
                ParseIssue(
                    ann.filename,
                    f"unknown class '{obj.name}' (object #{i}) — excluded",
                    "error",
                )
            )
            continue

        box = voc_box_to_xyxy(obj.xmin, obj.ymin, obj.xmax, obj.ymax)
        reason = validate_box(box, ann.width, ann.height)
        if reason is not None:
            issues.append(
                ParseIssue(ann.filename, f"object #{i} ({obj.name}) invalid: {reason}", "error")
            )
            continue

        if resize is not None:
            x1, y1, x2, y2 = box
            box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            box = clamp_box(box, out_w, out_h)
            reason = validate_box(box, out_w, out_h)
            if reason is not None:
                issues.append(
                    ParseIssue(
                        ann.filename,
                        f"object #{i} ({obj.name}) invalid after resize: {reason}",
                        "error",
                    )
                )
                continue

        boxes.append(box)
        labels.append(class_to_idx[obj.name])

    return boxes, labels, issues
