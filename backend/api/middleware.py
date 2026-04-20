"""미들웨어 - request-id 바인딩 및 액세스 로그."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """들어온 요청마다 X-Request-Id를 보장하고 structlog contextvar에 바인딩."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        log = structlog.get_logger("api.access")
        start = time.perf_counter()
        log.info("request.start")
        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            log.exception("request.error", duration_ms=round(duration_ms, 1))
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request.end",
            status=response.status_code,
            duration_ms=round(duration_ms, 1),
        )
        return response
