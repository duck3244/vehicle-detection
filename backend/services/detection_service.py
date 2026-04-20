"""
순수 감지 서비스 계층.

FastAPI 라우터·Streamlit·CLI 모두에서 재사용할 수 있도록,
I/O(업로드 파일, HTTP 응답 등)와 비즈니스 로직을 분리한다.
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import detection_config
from utils import logger


# --------------------------------------------------------------------------- #
# DTO
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    class_name: str
    class_kr: str
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]
    mask_path: Optional[Path] = None


@dataclass
class DetectionResult:
    run_id: str
    run_dir: Path
    annotated_image_path: Path
    detections: List[Detection]
    inference_ms: int
    meta: Dict[str, Any] = field(default_factory=dict)


class PipelineLike(Protocol):
    """테스트 대체를 위한 최소 프로토콜."""

    yolo_detector: Any
    sam_segmentor: Any
    enable_sam: bool
    yolo_model_name: str
    confidence_threshold: float


# --------------------------------------------------------------------------- #
# 입력 검증
# --------------------------------------------------------------------------- #


class InvalidImageError(ValueError):
    """이미지 바이트가 유효하지 않음."""


def decode_image(image_bytes: bytes) -> np.ndarray:
    """이미지 바이트를 BGR ndarray로 디코드하고 Pillow로 포맷 검증.

    Pillow의 verify()는 Streaming parser이므로 verify 후 파일을 재오픈해야 한다.
    """
    try:
        probe = Image.open(io.BytesIO(image_bytes))
        probe.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"유효하지 않은 이미지: {exc}") from exc

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# 내부 유틸
# --------------------------------------------------------------------------- #


def _clear_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # torch 미설치 환경 포함
        logger.debug(f"GPU 캐시 정리 생략: {exc}")


# --------------------------------------------------------------------------- #
# 공개 API
# --------------------------------------------------------------------------- #


def run_single_detection(
    image_bytes: bytes,
    pipeline: PipelineLike,
    output_root: Path,
    use_sam: bool = True,
    confidence: Optional[float] = None,
) -> DetectionResult:
    """단일 이미지 감지.

    Args:
        image_bytes: 업로드된 이미지의 raw 바이트.
        pipeline: 싱글턴으로 미리 로드된 파이프라인.
        output_root: 결과 파일이 저장될 루트. `<output_root>/<run_id>/` 생성됨.
        use_sam: SAM 세그멘테이션 수행 여부(MVP 기본 True).
        confidence: None이면 pipeline 기본값 사용.

    Returns:
        DetectionResult
    """
    t0 = time.time()
    run_id = uuid.uuid4().hex
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = decode_image(image_bytes)
        conf = confidence if confidence is not None else pipeline.confidence_threshold

        # YOLO
        raw_dets = pipeline.yolo_detector.detect_vehicles(image, conf_threshold=conf)

        # SAM (raw_dets가 있을 때만)
        masks: List[np.ndarray] = []
        want_sam = (
            use_sam
            and pipeline.enable_sam
            and pipeline.sam_segmentor is not None
            and bool(raw_dets)
        )
        if want_sam and pipeline.sam_segmentor.set_image(image):
            boxes = [det["bbox"] for det in raw_dets]
            masks = pipeline.sam_segmentor.segment_from_boxes(boxes)

        # 오버레이 이미지 저장
        annotated = pipeline.yolo_detector.visualize_detections(
            image, raw_dets, show_labels=True, show_confidence=True
        )
        annotated_path = run_dir / "annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        # 마스크 PNG 저장
        detections_out: List[Detection] = []
        for i, det in enumerate(raw_dets):
            mask_path: Optional[Path] = None
            if i < len(masks):
                mask_img = (np.asarray(masks[i]).astype(np.uint8) > 0).astype(np.uint8) * 255
                mp = run_dir / f"mask_{i}.png"
                cv2.imwrite(str(mp), mask_img)
                mask_path = mp

            detections_out.append(
                Detection(
                    class_name=det["class"],
                    class_kr=detection_config.CLASS_NAMES_KR.get(det["class"], det["class"]),
                    score=float(det["confidence"]),
                    bbox=[int(x) for x in det["bbox"]],
                    mask_path=mask_path,
                )
            )

        result = DetectionResult(
            run_id=run_id,
            run_dir=run_dir,
            annotated_image_path=annotated_path,
            detections=detections_out,
            inference_ms=int((time.time() - t0) * 1000),
            meta={
                "num_detections": len(detections_out),
                "sam_used": bool(masks),
                "yolo_model": pipeline.yolo_model_name,
            },
        )
        logger.info(
            "감지 완료 run_id=%s detections=%d sam=%s ms=%d",
            run_id,
            len(detections_out),
            bool(masks),
            result.inference_ms,
        )
        return result
    finally:
        _clear_gpu()


def run_batch_detection(
    images: List[bytes],
    pipeline: PipelineLike,
    output_root: Path,
    use_sam: bool = True,
    confidence: Optional[float] = None,
    progress_cb: Optional[Any] = None,
) -> List[DetectionResult]:
    """배치 감지.

    Step 4의 비동기 job은 이 함수 위에서 asyncio.Task로 래핑된다.
    여기서는 순수 반복 로직만 제공하고, 개별 실패는 예외 대신
    빈 DetectionResult로 대체하여 전체 실행을 이어간다.
    """
    results: List[DetectionResult] = []
    total = len(images)
    for idx, image_bytes in enumerate(images):
        try:
            result = run_single_detection(
                image_bytes,
                pipeline=pipeline,
                output_root=output_root,
                use_sam=use_sam,
                confidence=confidence,
            )
        except InvalidImageError as exc:
            logger.warning("배치 %d/%d 유효하지 않은 이미지: %s", idx + 1, total, exc)
            result = DetectionResult(
                run_id=uuid.uuid4().hex,
                run_dir=output_root,
                annotated_image_path=Path(),
                detections=[],
                inference_ms=0,
                meta={"error": str(exc)},
            )
        results.append(result)

        if progress_cb is not None:
            try:
                progress_cb(idx + 1, total, result)
            except Exception as exc:
                logger.debug("progress_cb 예외 무시: %s", exc)

    return results
