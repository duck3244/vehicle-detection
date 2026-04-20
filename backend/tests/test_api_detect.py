"""/api/detect 엔드포인트 테스트 (fake pipeline)."""

from __future__ import annotations

import asyncio
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import settings
from api.deps import get_pipeline
from api.main import create_app


class FakeYOLO:
    device = "cpu"

    def detect_vehicles(self, image, conf_threshold=None):
        return [
            {"class": "car", "confidence": 0.9, "bbox": np.array([10, 20, 90, 110])},
            {"class": "bus", "confidence": 0.8, "bbox": np.array([100, 30, 200, 150])},
        ]

    def visualize_detections(self, image, detections, **kwargs):
        return image


class FakeSAM:
    def __init__(self, shape):
        self.shape = shape

    def set_image(self, image):
        self.shape = image.shape
        return True

    def segment_from_boxes(self, boxes):
        masks = []
        for b in boxes:
            m = np.zeros(self.shape[:2], dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in b]
            m[y1:y2, x1:x2] = True
            masks.append(m)
        return masks


class FakePipeline:
    def __init__(self):
        self.yolo_detector = FakeYOLO()
        self.sam_segmentor = FakeSAM((240, 320, 3))
        self.enable_sam = True
        self.yolo_model_name = "fake-yolo"
        self.sam_model_type = "vit_h"
        self.confidence_threshold = 0.25


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    # 정적 파일 루트를 임시 디렉토리로 변경
    monkeypatch.setattr(settings, "API_RUNS_DIR", tmp_path)
    app = create_app(with_lifespan=False)
    app.state.jobs = {}
    app.state.jobs_lock = asyncio.Lock()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    return TestClient(app)


def _sample_jpeg() -> bytes:
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detect_returns_detections_with_sam(app_client: TestClient):
    files = {"file": ("car.jpg", _sample_jpeg(), "image/jpeg")}
    resp = app_client.post("/api/detect", files=files, data={"use_sam": "true"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["detections"]) == 2
    assert body["detections"][0]["class"] == "car"
    assert body["detections"][0]["class_kr"] == "자동차"
    assert body["detections"][0]["mask_url"].endswith(".png")
    assert body["annotated_image_url"].endswith("annotated.jpg")
    assert body["meta"]["sam_used"] is True
    assert body["meta"]["num_detections"] == 2


def test_detect_without_sam(app_client: TestClient):
    files = {"file": ("car.jpg", _sample_jpeg(), "image/jpeg")}
    resp = app_client.post("/api/detect", files=files, data={"use_sam": "false"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["sam_used"] is False
    for det in body["detections"]:
        assert det["mask_url"] is None


def test_detect_rejects_bad_extension(app_client: TestClient):
    files = {"file": ("evil.exe", b"bad", "application/octet-stream")}
    resp = app_client.post("/api/detect", files=files)
    assert resp.status_code == 415


def test_detect_rejects_oversize(app_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 100)
    files = {"file": ("big.jpg", b"x" * 200, "image/jpeg")}
    resp = app_client.post("/api/detect", files=files)
    assert resp.status_code == 413


def test_detect_rejects_invalid_image(app_client: TestClient):
    files = {"file": ("fake.jpg", b"not an image", "image/jpeg")}
    resp = app_client.post("/api/detect", files=files)
    assert resp.status_code == 400
