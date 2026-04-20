"""API 요청/응답 스키마 (Pydantic v2)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    class_name: str = Field(..., description="영문 클래스명", alias="class")
    class_kr: str = Field(..., description="한글 클래스명")
    score: float = Field(..., ge=0.0, le=1.0)
    bbox: list[int] = Field(..., min_length=4, max_length=4, description="[x1,y1,x2,y2]")
    mask_url: Optional[str] = Field(None, description="SAM 마스크 PNG URL, 없으면 null")

    model_config = {"populate_by_name": True}


class DetectMeta(BaseModel):
    inference_ms: int
    num_detections: int
    sam_used: bool
    yolo_model: str
    device: Optional[str] = None


class DetectResponse(BaseModel):
    run_id: str
    detections: list[DetectionItem]
    annotated_image_url: str
    meta: DetectMeta


class BatchJobResponse(BaseModel):
    job_id: str


JobStatus = Literal["pending", "running", "done", "failed"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(0, ge=0, le=100)
    total: int = 0
    done: int = 0
    results: Optional[list[DetectResponse]] = None
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    yolo: dict[str, str]
    sam: dict[str, str]
    default_yolo: str
    default_sam: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    gpu: dict[str, Any]
    memory: dict[str, Any]
    models_loaded: dict[str, Any]
