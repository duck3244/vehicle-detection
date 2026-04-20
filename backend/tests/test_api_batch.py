"""/api/detect/batch + /api/jobs/{id} 테스트."""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import settings
from api.deps import get_pipeline
from api.main import create_app
from api.routers.jobs import cleanup_expired


class FakeYOLO:
    device = "cpu"

    def detect_vehicles(self, image, conf_threshold=None):
        return [{"class": "car", "confidence": 0.8, "bbox": np.array([0, 0, 10, 10])}]

    def visualize_detections(self, image, detections, **kwargs):
        return image


class FakeSAM:
    def set_image(self, image):
        self.shape = image.shape
        return True

    def segment_from_boxes(self, boxes):
        return [np.ones((10, 10), dtype=bool) for _ in boxes]


class FakePipeline:
    def __init__(self):
        self.yolo_detector = FakeYOLO()
        self.sam_segmentor = FakeSAM()
        self.enable_sam = True
        self.yolo_model_name = "fake"
        self.sam_model_type = "vit_h"
        self.confidence_threshold = 0.25


def _sample() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "API_RUNS_DIR", tmp_path)
    app = create_app(with_lifespan=False)
    app.state.jobs = {}
    app.state.jobs_lock = asyncio.Lock()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    return TestClient(app)


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in {"done", "failed"}:
            return body
        time.sleep(0.05)
    pytest.fail(f"job {job_id} 가 시간 내 완료되지 않음")


def test_batch_submits_and_completes(app_client: TestClient):
    files = [("files", ("a.jpg", _sample(), "image/jpeg"))] * 3
    resp = app_client.post("/api/detect/batch", files=files, data={"use_sam": "true"})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_job(app_client, job_id)
    assert body["status"] == "done"
    assert body["done"] == 3
    assert body["total"] == 3
    assert body["progress"] == 100
    assert len(body["results"]) == 3
    assert body["results"][0]["detections"][0]["class"] == "car"


def test_batch_rejects_overflow(app_client: TestClient):
    files = [("files", (f"a{i}.jpg", _sample(), "image/jpeg")) for i in range(11)]
    resp = app_client.post("/api/detect/batch", files=files)
    assert resp.status_code == 400


def test_batch_rejects_bad_extension(app_client: TestClient):
    files = [("files", ("bad.exe", b"x", "application/octet-stream"))]
    resp = app_client.post("/api/detect/batch", files=files)
    assert resp.status_code == 415


def test_job_not_found(app_client: TestClient):
    resp = app_client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_cleanup_expires_old_jobs(app_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "JOB_TTL_SECONDS", 0)
    monkeypatch.setattr(settings, "RESULT_TTL_SECONDS", 0)
    # 하나 job 추가
    files = [("files", ("a.jpg", _sample(), "image/jpeg"))]
    resp = app_client.post("/api/detect/batch", files=files)
    job_id = resp.json()["job_id"]
    _wait_for_job(app_client, job_id)

    app = app_client.app
    asyncio.get_event_loop().run_until_complete(cleanup_expired(app))

    # 만료됐으므로 404
    resp = app_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 404
