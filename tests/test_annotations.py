"""Tests for Pascal VOC parsing and coordinate conversion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from roadvision.data.annotations import (
    DEFAULT_CLASSES,
    clamp_box,
    convert_annotation,
    parse_voc_xml,
    validate_box,
    voc_box_to_xyxy,
)


def write_voc(
    path: Path,
    *,
    filename: str = "img_001.jpg",
    width: int = 100,
    height: int = 80,
    objects: list[tuple[str, float, float, float, float]] | None = None,
    skip_size: bool = False,
    skip_filename: bool = False,
    corrupt: bool = False,
) -> Path:
    """Write a minimal VOC XML file for tests."""
    if corrupt:
        path.write_text("<annotation><broken", encoding="utf-8")
        return path

    root = ET.Element("annotation")
    if not skip_filename:
        ET.SubElement(root, "filename").text = filename
    if not skip_size:
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
    for name, xmin, ymin, xmax, ymax in objects or []:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(xmin)
        ET.SubElement(bnd, "ymin").text = str(ymin)
        ET.SubElement(bnd, "xmax").text = str(xmax)
        ET.SubElement(bnd, "ymax").text = str(ymax)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_parse_valid_xml(tmp_path: Path):
    xml = write_voc(
        tmp_path / "a.xml",
        objects=[("D00", 10, 20, 50, 60), ("D40", 1, 1, 10, 10)],
    )
    ann, issues = parse_voc_xml(xml)
    assert ann is not None
    assert ann.filename == "img_001.jpg"
    assert ann.width == 100 and ann.height == 80
    assert len(ann.objects) == 2
    assert ann.objects[0].name == "D00"
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []


def test_parse_missing_file(tmp_path: Path):
    ann, issues = parse_voc_xml(tmp_path / "nope.xml")
    assert ann is None
    assert any("does not exist" in i.reason for i in issues)


def test_parse_malformed_xml(tmp_path: Path):
    xml = write_voc(tmp_path / "bad.xml", corrupt=True)
    ann, issues = parse_voc_xml(xml)
    assert ann is None
    assert any("malformed XML" in i.reason for i in issues)


def test_parse_missing_size(tmp_path: Path):
    xml = write_voc(tmp_path / "s.xml", skip_size=True)
    ann, issues = parse_voc_xml(xml)
    assert ann is None
    assert any("<size>" in i.reason for i in issues)


def test_parse_missing_filename(tmp_path: Path):
    xml = write_voc(tmp_path / "f.xml", skip_filename=True)
    ann, issues = parse_voc_xml(xml)
    assert ann is None
    assert any("filename" in i.reason for i in issues)


def test_parse_zero_objects_warns(tmp_path: Path):
    xml = write_voc(tmp_path / "e.xml", objects=[])
    ann, issues = parse_voc_xml(xml)
    assert ann is not None
    assert any(i.severity == "warn" and "zero objects" in i.reason for i in issues)


def test_voc_box_conversion():
    # 1-based inclusive (xmin=10) -> 0-based (x1=9); xmax passes through.
    assert voc_box_to_xyxy(10, 20, 50, 60) == (9.0, 19.0, 50.0, 60.0)
    # Full-image box: xmin=1,ymin=1,w=100,h=80 -> (0,0,100,80)
    assert voc_box_to_xyxy(1, 1, 100, 80) == (0.0, 0.0, 100.0, 80.0)


def test_validate_box_degenerate():
    assert validate_box((10, 10, 10, 50), 100, 100) is not None  # zero width
    assert validate_box((10, 10, 50, 10), 100, 100) is not None  # zero height
    assert validate_box((0, 0, 20, 20), 100, 100) is None  # valid


def test_validate_box_outside():
    assert validate_box((200, 200, 300, 300), 100, 100) is not None
    assert validate_box((-50, -50, -10, -10), 100, 100) is not None
    # Partially visible is OK
    assert validate_box((-10, -10, 20, 20), 100, 100) is None


def test_validate_box_nonfinite():
    import math

    assert validate_box((math.nan, 0, 10, 10), 100, 100) is not None
    assert validate_box((0, 0, math.inf, 10), 100, 100) is not None


def test_clamp_box():
    assert clamp_box((-5, -5, 200, 200), 100, 80) == (0.0, 0.0, 100.0, 80.0)


def test_convert_annotation_unknown_class(tmp_path: Path):
    xml = write_voc(tmp_path / "u.xml", objects=[("D99", 10, 10, 50, 50)])
    ann, _ = parse_voc_xml(xml)
    boxes, labels, issues = convert_annotation(ann, DEFAULT_CLASSES)
    assert boxes == [] and labels == []
    assert any("unknown class" in i.reason for i in issues)


def test_convert_annotation_invalid_box_excluded(tmp_path: Path):
    xml = write_voc(
        tmp_path / "i.xml",
        objects=[("D00", 50, 10, 10, 40),  # xmax < xmin -> inverted/degenerate
                 ("D10", 5, 5, 50, 50)],   # valid
    )
    ann, _ = parse_voc_xml(xml)
    boxes, labels, issues = convert_annotation(ann, DEFAULT_CLASSES)
    assert len(boxes) == 1
    assert labels == [2]  # D10 -> index 2
    assert any("invalid" in i.reason for i in issues)


def test_convert_annotation_with_resize():
    # Box (9,19,50,60) original 100x80 -> resize 320x320
    # scale = (3.2, 4.0)
    xml_boxes = voc_box_to_xyxy(10, 20, 50, 60)
    from roadvision.data.annotations import VOCAnnotation, VOCObject

    ann = VOCAnnotation(
        filename="x.jpg",
        width=100,
        height=80,
        objects=[VOCObject(name="D00", xmin=10, ymin=20, xmax=50, ymax=60)],
    )
    boxes, labels, issues = convert_annotation(ann, DEFAULT_CLASSES, resize=(320, 320))
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 == pytest.approx(xml_boxes[0] * 3.2)
    assert y1 == pytest.approx(xml_boxes[1] * 4.0)
    assert x2 == pytest.approx(xml_boxes[2] * 3.2)
    assert y2 == pytest.approx(xml_boxes[3] * 4.0)
    assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 320
    assert labels == [1]
    assert issues == []


def test_class_mapping_is_one_based_with_background_zero():
    from roadvision.data.dataset import RoadDamageDataset

    ds = RoadDamageDataset(records=[])
    assert ds.class_to_idx == {"D00": 1, "D10": 2, "D20": 3, "D40": 4}
    assert 0 not in ds.class_to_idx.values()  # background reserved
