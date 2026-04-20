"""감지 엔드포인트."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status

from api import settings
from api.deps import get_pipeline
from api.routers.jobs import new_job, update_job
from api.schemas import BatchJobResponse, DetectionItem, DetectMeta, DetectResponse
from services.detection_service import (
    DetectionResult,
    InvalidImageError,
    run_single_detection,
)
from utils import logger

router = APIRouter(prefix="/api", tags=["detect"])


def _validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "파일명이 없습니다")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"허용되지 않는 확장자: {suffix}",
        )


async def _read_bounded(file: UploadFile) -> bytes:
    """max_upload_bytes 초과 시 413 반환."""
    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"파일이 {settings.MAX_UPLOAD_BYTES} 바이트를 초과했습니다",
        )
    return data


def build_detect_response(request: Request, result: DetectionResult, device: str | None) -> DetectResponse:
    base = str(request.base_url).rstrip("/") + settings.STATIC_MOUNT_PATH
    run_prefix = f"{base}/{result.run_id}"

    items: list[DetectionItem] = []
    for i, det in enumerate(result.detections):
        mask_url = None
        if det.mask_path is not None and det.mask_path.exists():
            mask_url = f"{run_prefix}/{det.mask_path.name}"
        items.append(
            DetectionItem(
                **{"class": det.class_name},
                class_kr=det.class_kr,
                score=det.score,
                bbox=det.bbox,
                mask_url=mask_url,
            )
        )

    return DetectResponse(
        run_id=result.run_id,
        detections=items,
        annotated_image_url=f"{run_prefix}/{result.annotated_image_path.name}",
        meta=DetectMeta(
            inference_ms=result.inference_ms,
            num_detections=result.meta["num_detections"],
            sam_used=result.meta["sam_used"],
            yolo_model=result.meta["yolo_model"],
            device=device,
        ),
    )


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {}, 413: {}, 415: {}, 422: {}},
)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    use_sam: bool = Form(settings.DEFAULT_USE_SAM),
    confidence: Optional[float] = Form(None),
    pipeline=Depends(get_pipeline),
) -> DetectResponse:
    """단일 이미지 감지 (동기)."""
    _validate_upload(file)
    data = await _read_bounded(file)

    try:
        # 동기 파이프라인을 스레드풀에서 실행 (event loop blocking 방지)
        result = await asyncio.to_thread(
            run_single_detection,
            data,
            pipeline,
            settings.API_RUNS_DIR,
            use_sam,
            confidence,
        )
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    device = getattr(getattr(pipeline, "yolo_detector", None), "device", None)
    return build_detect_response(request, result, device=device)


async def _run_batch(
    request_base_url: str,
    app,
    pipeline,
    job_id: str,
    payloads: list[bytes],
    use_sam: bool,
    confidence: Optional[float],
) -> None:
    """백그라운드 태스크: 배치 감지 실행 후 job state 업데이트."""
    device = getattr(getattr(pipeline, "yolo_detector", None), "device", None)
    update_job(app, job_id, status="running")
    results_json: list[dict] = []
    total = len(payloads)
    try:
        for idx, data in enumerate(payloads):
            try:
                result = await asyncio.to_thread(
                    run_single_detection,
                    data,
                    pipeline,
                    settings.API_RUNS_DIR,
                    use_sam,
                    confidence,
                )
                base = request_base_url.rstrip("/") + settings.STATIC_MOUNT_PATH
                run_prefix = f"{base}/{result.run_id}"
                items = []
                for i, det in enumerate(result.detections):
                    mask_url = None
                    if det.mask_path is not None and det.mask_path.exists():
                        mask_url = f"{run_prefix}/{det.mask_path.name}"
                    items.append(
                        {
                            "class": det.class_name,
                            "class_kr": det.class_kr,
                            "score": det.score,
                            "bbox": det.bbox,
                            "mask_url": mask_url,
                        }
                    )
                results_json.append(
                    {
                        "run_id": result.run_id,
                        "detections": items,
                        "annotated_image_url": f"{run_prefix}/{result.annotated_image_path.name}",
                        "meta": {
                            "inference_ms": result.inference_ms,
                            "num_detections": result.meta["num_detections"],
                            "sam_used": result.meta["sam_used"],
                            "yolo_model": result.meta["yolo_model"],
                            "device": device,
                        },
                    }
                )
            except InvalidImageError as exc:
                results_json.append({"error": str(exc)})
            done = idx + 1
            update_job(
                app,
                job_id,
                done=done,
                progress=int(done / total * 100) if total else 100,
            )
        update_job(app, job_id, status="done", results=results_json)
    except Exception as exc:
        logger.exception("배치 job 실패 job_id=%s", job_id)
        update_job(app, job_id, status="failed", error=str(exc))


@router.post(
    "/detect/batch",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {}, 413: {}, 415: {}, 422: {}},
)
async def detect_batch(
    request: Request,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    use_sam: bool = Form(settings.DEFAULT_USE_SAM),
    confidence: Optional[float] = Form(None),
    pipeline=Depends(get_pipeline),
) -> BatchJobResponse:
    """배치 감지 (비동기 job)."""
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "파일이 없습니다")
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"배치 크기 제한 {settings.MAX_BATCH_SIZE} 초과",
        )

    payloads: list[bytes] = []
    for f in files:
        _validate_upload(f)
        payloads.append(await _read_bounded(f))

    job_id = new_job(request.app, total=len(payloads))
    background.add_task(
        _run_batch,
        str(request.base_url),
        request.app,
        pipeline,
        job_id,
        payloads,
        use_sam,
        confidence,
    )
    return BatchJobResponse(job_id=job_id)
