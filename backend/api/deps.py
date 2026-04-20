"""FastAPI 의존성 및 앱 상태 접근자."""

from __future__ import annotations

from fastapi import Depends, Request

from pipeline import VehicleDetectionPipeline


def get_pipeline(request: Request) -> VehicleDetectionPipeline:
    """lifespan에서 로드된 싱글턴 파이프라인을 반환."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized (lifespan 미실행)")
    return pipeline


__all__ = ["Depends", "get_pipeline"]
