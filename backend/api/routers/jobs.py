"""배치 감지 작업 상태 관리 및 TTL 정리.

인메모리 dict 기반. MVP에서는 Celery/Redis를 도입하지 않는다.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from api import settings
from api.schemas import JobStatusResponse
from utils import logger

router = APIRouter(prefix="/api", tags=["jobs"])


def new_job(app, total: int) -> str:
    job_id = uuid.uuid4().hex
    app.state.jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "done": 0,
        "total": total,
        "results": None,
        "error": None,
        "created_at": time.time(),
    }
    return job_id


def update_job(app, job_id: str, **fields: Any) -> None:
    job = app.state.jobs.get(job_id)
    if job is None:
        return
    job.update(fields)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(request: Request, job_id: str) -> JobStatusResponse:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job_id 없음 (만료되었을 수 있음)")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        total=job["total"],
        done=job["done"],
        results=job.get("results"),
        error=job.get("error"),
    )


async def cleanup_expired(app) -> None:
    """TTL이 지난 job 및 정적 파일 정리."""
    now = time.time()

    async with app.state.jobs_lock:
        expired = [
            jid
            for jid, job in app.state.jobs.items()
            if now - job.get("created_at", now) > settings.JOB_TTL_SECONDS
        ]
        for jid in expired:
            app.state.jobs.pop(jid, None)
    if expired:
        logger.info("만료 job %d건 제거", len(expired))

    root = settings.API_RUNS_DIR
    if not root.exists():
        return
    removed = 0
    for run_dir in list(root.iterdir()):
        try:
            if not run_dir.is_dir():
                continue
            mtime = run_dir.stat().st_mtime
            if now - mtime > settings.RESULT_TTL_SECONDS:
                for f in run_dir.iterdir():
                    f.unlink(missing_ok=True)
                run_dir.rmdir()
                removed += 1
        except OSError as exc:
            logger.debug("cleanup 실패 %s: %s", run_dir, exc)
    if removed:
        logger.info("만료 run_dir %d건 제거", removed)
