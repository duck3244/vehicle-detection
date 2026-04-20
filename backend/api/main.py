"""FastAPI 엔트리포인트.

실행:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import settings
from api.logging_config import configure_logging, get_logger
from api.middleware import RequestIdMiddleware
from api.routers import detect, jobs, meta
from pipeline import VehicleDetectionPipeline

configure_logging()
logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API lifespan 시작: 파이프라인 로드")
    app.state.pipeline = VehicleDetectionPipeline(enable_sam=settings.DEFAULT_USE_SAM)
    app.state.jobs = {}  # job_id -> dict
    app.state.jobs_lock = asyncio.Lock()
    cleanup_task = asyncio.create_task(_cleanup_loop(app))
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("API lifespan 종료")


async def _cleanup_loop(app: FastAPI) -> None:
    from api.routers.jobs import cleanup_expired

    while True:
        try:
            await asyncio.sleep(settings.CLEANUP_INTERVAL_SECONDS)
            await cleanup_expired(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            logger.warning("cleanup loop 예외: %s", exc)


def create_app(*, with_lifespan: bool = True) -> FastAPI:
    app = FastAPI(
        title="Vehicle Detection API",
        version="0.1.0",
        lifespan=lifespan if with_lifespan else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )
    app.add_middleware(RequestIdMiddleware)
    app.mount(
        settings.STATIC_MOUNT_PATH,
        StaticFiles(directory=settings.API_RUNS_DIR),
        name="static",
    )
    app.include_router(meta.router)
    app.include_router(detect.router)
    app.include_router(jobs.router)
    return app


app = create_app()
