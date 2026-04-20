import numpy as np
import pytest

from utils import BboxUtils


def test_xyxy_to_xywh_roundtrip():
    xyxy = np.array([10, 20, 110, 220])
    xywh = BboxUtils.xyxy_to_xywh(xyxy)
    np.testing.assert_array_equal(xywh, np.array([10, 20, 100, 200]))
    np.testing.assert_array_equal(BboxUtils.xywh_to_xyxy(xywh), xyxy)


def test_calculate_area_xyxy():
    assert BboxUtils.calculate_area(np.array([0, 0, 10, 20])) == 200


def test_calculate_area_invalid_length():
    assert BboxUtils.calculate_area(np.array([1, 2, 3])) == 0.0


def test_iou_identical_is_one():
    bbox = np.array([0, 0, 10, 10])
    assert BboxUtils.calculate_iou(bbox, bbox) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = np.array([0, 0, 10, 10])
    b = np.array([20, 20, 30, 30])
    assert BboxUtils.calculate_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.array([0, 0, 10, 10])
    b = np.array([5, 0, 15, 10])
    # 교집합 50, 합집합 150 → 1/3
    assert BboxUtils.calculate_iou(a, b) == pytest.approx(1 / 3)


def test_bbox_center():
    assert BboxUtils.get_bbox_center(np.array([0, 0, 10, 20])) == (5, 10)
