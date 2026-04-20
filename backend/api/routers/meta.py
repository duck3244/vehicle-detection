"""시스템 메타 엔드포인트 (/health, /models)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_pipeline
from api.schemas import HealthResponse, ModelsResponse
from config import model_config
from utils import PerformanceUtils

router = APIRouter(prefix="/api", tags=["meta"])


def _gpu_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
            }
        return {"available": False}
    except Exception:
        return {"available": False}


@router.get("/system/health", response_model=HealthResponse)
def health(pipeline=Depends(get_pipeline)) -> HealthResponse:
    """현재 런타임 상태 (GPU/메모리/모델 로드 여부)."""
    mem = PerformanceUtils.get_memory_usage() or {}
    models_loaded = {
        "yolo": pipeline.yolo_model_name,
        "sam": pipeline.sam_model_type if pipeline.enable_sam else None,
    }
    status = "ok" if pipeline.yolo_detector is not None else "degraded"
    return HealthResponse(status=status, gpu=_gpu_info(), memory=mem, models_loaded=models_loaded)


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(
        yolo=model_config.YOLO_MODELS,
        sam=model_config.SAM_MODELS,
        default_yolo=model_config.DEFAULT_YOLO_MODEL,
        default_sam="vit_h",
    )
