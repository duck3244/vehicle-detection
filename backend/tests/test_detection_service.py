"""detection_service의 순수 로직을 fake pipeline으로 검증."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from PIL import Image

from services.detection_service import (
    Detection,
    DetectionResult,
    InvalidImageError,
    decode_image,
    run_batch_detection,
    run_single_detection,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeYOLO:
    def __init__(self, detections):
        self._detections = detections
        self.last_conf = None

    def detect_vehicles(self, image, conf_threshold=None):
        self.last_conf = conf_threshold
        return self._detections

    def visualize_detections(self, image, detections, show_labels=True, show_confidence=True):
        # 검정 이미지에 단색 테두리만
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return out


class FakeSAM:
    def __init__(self, shape):
        self._shape = shape
        self._image = None

    def set_image(self, image):
        self._image = image
        return True

    def segment_from_boxes(self, boxes):
        masks = []
        for box in boxes:
            m = np.zeros(self._shape[:2], dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in box]
            m[y1:y2, x1:x2] = True
            masks.append(m)
        return masks


class FakePipeline:
    def __init__(self, detections, shape, enable_sam=True):
        self.yolo_detector = FakeYOLO(detections)
        self.sam_segmentor = FakeSAM(shape)
        self.enable_sam = enable_sam
        self.yolo_model_name = "fake-yolo"
        self.confidence_threshold = 0.25


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (320, 240), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def fake_detections() -> List[dict]:
    return [
        {"class": "car", "confidence": 0.91, "bbox": np.array([10, 20, 100, 150])},
        {"class": "truck", "confidence": 0.73, "bbox": np.array([120, 40, 260, 180])},
    ]


# --------------------------------------------------------------------------- #
# decode_image
# --------------------------------------------------------------------------- #


def test_decode_image_valid(sample_image_bytes):
    arr = decode_image(sample_image_bytes)
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8


def test_decode_image_invalid_raises():
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")


# --------------------------------------------------------------------------- #
# run_single_detection
# --------------------------------------------------------------------------- #


def test_run_single_detection_with_sam(tmp_path, sample_image_bytes, fake_detections):
    pipeline = FakePipeline(fake_detections, shape=(240, 320, 3), enable_sam=True)
    result = run_single_detection(
        sample_image_bytes,
        pipeline=pipeline,
        output_root=tmp_path,
        use_sam=True,
        confidence=0.4,
    )

    assert isinstance(result, DetectionResult)
    assert pipeline.yolo_detector.last_conf == 0.4
    assert len(result.detections) == 2
    assert result.detections[0].class_kr == "자동차"
    assert result.detections[1].class_kr == "트럭"
    assert result.meta["sam_used"] is True
    assert result.meta["num_detections"] == 2

    # 저장 경로 확인
    assert result.annotated_image_path.exists()
    assert result.run_dir.exists()
    for det in result.detections:
        assert det.mask_path is not None
        assert det.mask_path.exists()


def test_run_single_detection_sam_disabled_no_masks(tmp_path, sample_image_bytes, fake_detections):
    pipeline = FakePipeline(fake_detections, shape=(240, 320, 3), enable_sam=True)
    result = run_single_detection(
        sample_image_bytes,
        pipeline=pipeline,
        output_root=tmp_path,
        use_sam=False,
    )
    assert result.meta["sam_used"] is False
    for det in result.detections:
        assert det.mask_path is None


def test_run_single_detection_no_vehicles(tmp_path, sample_image_bytes):
    pipeline = FakePipeline([], shape=(240, 320, 3), enable_sam=True)
    result = run_single_detection(
        sample_image_bytes,
        pipeline=pipeline,
        output_root=tmp_path,
    )
    assert result.detections == []
    assert result.meta["sam_used"] is False
    assert result.annotated_image_path.exists()


def test_run_single_detection_invalid_image(tmp_path):
    pipeline = FakePipeline([], shape=(240, 320, 3))
    with pytest.raises(InvalidImageError):
        run_single_detection(b"garbage", pipeline=pipeline, output_root=tmp_path)


# --------------------------------------------------------------------------- #
# run_batch_detection
# --------------------------------------------------------------------------- #


def test_run_batch_detection_progress(tmp_path, sample_image_bytes, fake_detections):
    pipeline = FakePipeline(fake_detections, shape=(240, 320, 3), enable_sam=True)
    calls: List[tuple] = []
    results = run_batch_detection(
        [sample_image_bytes] * 3,
        pipeline=pipeline,
        output_root=tmp_path,
        progress_cb=lambda done, total, result: calls.append((done, total)),
    )
    assert len(results) == 3
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_run_batch_detection_continues_on_bad_image(tmp_path, sample_image_bytes, fake_detections):
    pipeline = FakePipeline(fake_detections, shape=(240, 320, 3), enable_sam=True)
    results = run_batch_detection(
        [sample_image_bytes, b"bad", sample_image_bytes],
        pipeline=pipeline,
        output_root=tmp_path,
    )
    assert len(results) == 3
    assert results[1].detections == []
    assert "error" in results[1].meta
    assert len(results[0].detections) == 2
    assert len(results[2].detections) == 2
