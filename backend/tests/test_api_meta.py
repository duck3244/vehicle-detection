"""/api/system/health, /api/models 엔드포인트 테스트."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from api.deps import get_pipeline
from api.main import create_app


class FakePipeline:
    yolo_detector = object()
    sam_segmentor = object()
    enable_sam = True
    yolo_model_name = "yolov8n.pt"
    sam_model_type = "vit_h"
    confidence_threshold = 0.25


@pytest.fixture
def client() -> TestClient:
    app = create_app(with_lifespan=False)
    app.state.jobs = {}
    app.state.jobs_lock = asyncio.Lock()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/system/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"]["yolo"] == "yolov8n.pt"
    assert body["models_loaded"]["sam"] == "vit_h"
    assert "gpu" in body
    assert "memory" in body


def test_models_list(client: TestClient) -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "yolo" in body
    assert "sam" in body
    assert body["default_yolo"].endswith(".pt")
    assert body["default_sam"] in body["sam"]


def test_openapi_schema(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    paths = spec["paths"]
    assert "/api/system/health" in paths
    assert "/api/models" in paths
