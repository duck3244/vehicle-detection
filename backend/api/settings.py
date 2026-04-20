"""API 런타임 설정 (MVP 값)."""

from __future__ import annotations

from pathlib import Path

from config import OUTPUT_DIR

# 결과 정적 파일 루트
API_RUNS_DIR: Path = OUTPUT_DIR / "api_runs"
API_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# TTL (초)
RESULT_TTL_SECONDS: int = 60 * 60  # 1시간
JOB_TTL_SECONDS: int = 60 * 60
CLEANUP_INTERVAL_SECONDS: int = 5 * 60  # 5분마다 정리

# 업로드 제약
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10MB
MAX_BATCH_SIZE: int = 10
ALLOWED_EXTENSIONS: frozenset = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# CORS
CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

# SAM 기본값
DEFAULT_USE_SAM: bool = True

# 정적 서빙 경로 (FastAPI 마운트 경로)
STATIC_MOUNT_PATH: str = "/api/static"
