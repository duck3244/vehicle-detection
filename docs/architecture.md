# Architecture

본 문서는 `vehicle-detection` 프로젝트의 구조와 런타임 동작을 요약합니다. 세부 다이어그램은 [uml.md](./uml.md) 참조.

## 1. 상위 구성

```
┌────────────────────────┐       HTTP (/api/*)        ┌───────────────────────────────┐
│  Frontend (React/Vite) │ ─────────────────────────► │  Backend (FastAPI, uvicorn)   │
│  :5173                 │ ◄───────────────────────── │  :8000                        │
│  - TanStack Query      │   JSON + static images     │  - Routers / Services         │
│  - Konva 오버레이      │                            │  - VehicleDetectionPipeline   │
└────────────────────────┘                            │    └─ YOLOVehicleDetector     │
         ▲                                            │    └─ SAMSegmentor            │
         │                                            │  - /api/static → output/...   │
         │ dev proxy: /api → :8000                    └───────────────────────────────┘
         │                                                       │
         │                                                       ▼
         │                                          ┌────────────────────────────┐
         │                                          │  Model weights             │
         │                                          │  yolov8*.pt, sam_vit_*.pth │
         │                                          └────────────────────────────┘
```

실행 가능한 표면은 3가지:
- **REST API** — `backend/api/main.py` (우선 지원, 프론트엔드의 유일한 의존 대상)
- **CLI** — `backend/main.py`
- **Streamlit (deprecated)** — `backend/app.py`

세 진입점 모두 내부적으로 `VehicleDetectionPipeline`을 재사용합니다.

## 2. 레이어 구성

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Presentation                                                           │
│  - frontend/src (React, Konva, shadcn UI, react-query)                 │
│  - backend/api/routers (detect.py, jobs.py, meta.py)                   │
│  - backend/main.py (CLI)                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ Application / Service                                                   │
│  - backend/services/detection_service.py                               │
│      · run_single_detection, run_batch_detection, decode_image         │
│      · DTO: Detection, DetectionResult                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Domain / Pipeline                                                       │
│  - backend/pipeline.py::VehicleDetectionPipeline                       │
│      · 컴포넌트 조합, process_image, process_batch, 통계                │
│  - backend/yolo_detector.py::YOLOVehicleDetector                       │
│  - backend/sam_segmentor.py::SAMSegmentor                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Infrastructure                                                          │
│  - backend/config.py (경로·상수·하드웨어·클래스 정의)                   │
│  - backend/utils.py (Image/File/Text/Validation/Performance 유틸)      │
│  - backend/api/middleware.py (X-Request-Id)                            │
│  - backend/api/logging_config.py (structlog)                           │
│  - backend/api/settings.py (TTL, 업로드 제한, CORS)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

핵심 원칙:
- **서비스 레이어 분리** — HTTP/CLI/Streamlit 어디서든 `run_single_detection(image_bytes, pipeline, output_root, ...)`만 호출하면 동일한 결과 DTO가 나오도록 설계.
- **파이프라인 싱글턴** — FastAPI `lifespan`에서 모델을 1회 로드한 뒤 `app.state.pipeline`에 보관해 요청마다 재로드를 피함.
- **블로킹 회피** — GPU 추론은 동기 함수이므로 라우터는 `asyncio.to_thread(run_single_detection, ...)`로 스레드풀에 위임하여 event loop를 보호.

## 3. 런타임 구성 요소

### 3.1 Backend

| 모듈 | 역할 |
|---|---|
| `api/main.py` | `FastAPI` 생성, `CORSMiddleware`/`RequestIdMiddleware` 연결, `/api/static` 마운트, lifespan에서 `VehicleDetectionPipeline` 로드, TTL cleanup 태스크 시작. |
| `api/settings.py` | 업로드 제한(10MB/배치 10장), 허용 확장자, TTL(1h), cleanup 주기(5m), CORS 오리진, 정적 마운트 경로(`/api/static`). |
| `api/deps.py` | `get_pipeline(request)` — `app.state.pipeline` 싱글턴 주입. |
| `api/middleware.py` | 요청마다 `X-Request-Id` 헤더 부여(없으면 UUID 생성). |
| `api/logging_config.py` | `structlog` 기반 JSON 로깅 구성. |
| `api/schemas.py` | Pydantic 응답: `DetectionItem`, `DetectMeta`, `DetectResponse`, `JobStatusResponse`, `HealthResponse`, `ModelsResponse`. |
| `api/routers/detect.py` | `POST /api/detect` 동기 처리, `POST /api/detect/batch` 비동기 job 등록. `BackgroundTasks`로 `_run_batch` 실행. |
| `api/routers/jobs.py` | `GET /api/jobs/{id}`, 메모리 dict(`app.state.jobs`) 기반 상태 관리 및 TTL cleanup. |
| `api/routers/meta.py` | `GET /api/system/health`, `GET /api/models`. |
| `services/detection_service.py` | `decode_image`, `run_single_detection`, `run_batch_detection` — 순수 비즈니스 로직. |
| `pipeline.py` | 컴포넌트 초기화/오케스트레이션, 단일·배치 처리, 시각화, 통계. |
| `yolo_detector.py` | `ultralytics.YOLO` 래퍼. COCO 차량 클래스 필터링, bbox/area/center 계산. |
| `sam_segmentor.py` | `segment-anything` 래퍼. `set_image` 후 bbox 프롬프트로 마스크 생성. |
| `config.py` | `ModelConfig`, `DetectionConfig`, `HardwareConfig` 등 상수 그룹. |
| `utils.py` | 이미지/파일/텍스트 유틸, 성능 측정, 의존성 체크, 로거. |

### 3.2 Frontend

| 영역 | 내용 |
|---|---|
| 빌드 | Vite 5 + TypeScript + Tailwind + shadcn/ui. `vite.config.ts`가 `/api` → `http://localhost:8000` 프록시. |
| 상태/쿼리 | `@tanstack/react-query` (`src/api/hooks.ts`) — `useDetectSingle`, `useDetectBatch`, `useJob`(1s 폴링). |
| API 클라이언트 | `axios` 인스턴스(baseURL `/api`, timeout 90s). 타입은 `npm run gen:api`로 `openapi.json` → `src/api/schema.ts` 자동 생성. |
| 페이지 | `App.tsx`에서 Tabs로 `SinglePage`/`BatchPage` 전환. |
| 시각화 | `DetectionCanvas` (react-konva)가 원본 이미지 + bbox/mask 오버레이 렌더. 서버 주석 이미지 토글 지원. |
| 컴포넌트 | `Dropzone`(react-dropzone), `DetectionOptions`(SAM toggle + confidence slider), `DetectionList`. |

### 3.3 정적 자산 & 결과 저장

- 감지 결과는 `backend/output/api_runs/<run_id>/` 아래 `annotated.jpg`, `mask_i.png` 형태로 저장.
- FastAPI가 `StaticFiles`로 해당 디렉토리를 `/api/static`에 마운트 → 응답 JSON의 `annotated_image_url`, `mask_url`이 그 경로를 가리킴.
- 1시간 TTL이 지난 `run_dir`은 `cleanup_expired()`가 5분 주기로 삭제.

## 4. 주요 런타임 플로우

### 4.1 단일 감지 (동기)

1. 프론트엔드가 `POST /api/detect` multipart 전송 (`file`, `use_sam`, `confidence`).
2. `detect` 라우터: 확장자·크기 검증 → `asyncio.to_thread(run_single_detection, ...)`.
3. 서비스 레이어: `decode_image` → `pipeline.yolo_detector.detect_vehicles` → (조건부) `pipeline.sam_segmentor.set_image` + `segment_from_boxes` → `visualize_detections` → 이미지/마스크 저장.
4. 라우터: `DetectResponse`로 직렬화, 정적 URL 합성.
5. 프론트엔드: `DetectionCanvas`에 bbox/mask 렌더 또는 서버 주석 이미지를 `<img>`로 교체.

### 4.2 배치 감지 (비동기 job)

1. `POST /api/detect/batch` — 모든 파일 검증 후 payload 배열을 메모리에 로드.
2. `new_job(app, total=N)` — UUID4로 `job_id` 생성, `app.state.jobs[job_id]`에 `pending` 상태 삽입.
3. `BackgroundTasks`에 `_run_batch` 등록 → 202 + `{job_id}` 반환.
4. 백그라운드: 각 이미지마다 `run_single_detection` 수행, `done`/`progress` 갱신, 최종 `status="done"` + `results` JSON 저장. 개별 실패는 `{"error": ...}`로 결과 슬롯에 기록.
5. 프론트엔드: `useJob(jobId)`가 1초 간격으로 `GET /api/jobs/{id}` 폴링, `done`/`failed` 도달 시 정지.

### 4.3 정리 루프

`lifespan` 시작 시 `_cleanup_loop` 태스크가 `asyncio.sleep(CLEANUP_INTERVAL_SECONDS=300)`를 돌며 `cleanup_expired`를 호출. `JOB_TTL_SECONDS=3600` 초과 job은 dict에서 pop, `RESULT_TTL_SECONDS=3600` 초과 `run_dir`은 파일 삭제 후 디렉토리 제거.

## 5. 스레딩 / 동시성 모델

- **이벤트 루프 (uvicorn)** — 모든 HTTP 핸들러, 배치 job 백그라운드 태스크, cleanup 루프.
- **스레드풀 (`asyncio.to_thread`)** — 모델 추론은 동기 코드라 이벤트 루프를 막지 않도록 스레드에서 실행.
- **job store 락** — `app.state.jobs_lock` (asyncio.Lock) — cleanup 시 pop 경합 방지.
- **모델은 싱글턴** — PyTorch 모델을 요청마다 로드하지 않고 lifespan에서 1회 로드.

## 6. 데이터 모델 / 계약

**요청:** `multipart/form-data`
- `file` (또는 `files[]`) — 이미지 바이너리
- `use_sam` (bool, 기본 `DEFAULT_USE_SAM=True`)
- `confidence` (float, nullable — pipeline 기본값 사용)

**응답 (`DetectResponse`):**

```jsonc
{
  "run_id": "hex32",
  "detections": [
    {
      "class": "car",
      "class_kr": "자동차",
      "score": 0.89,
      "bbox": [x1, y1, x2, y2],
      "mask_url": "https://.../api/static/<run_id>/mask_0.png"  // SAM off면 null
    }
  ],
  "annotated_image_url": "https://.../api/static/<run_id>/annotated.jpg",
  "meta": {
    "inference_ms": 812,
    "num_detections": 3,
    "sam_used": true,
    "yolo_model": "yolov8n.pt",
    "device": "cuda"
  }
}
```

**배치:** `POST /api/detect/batch` → `{"job_id": "..."}`. `GET /api/jobs/{id}`가 `{status, progress, done, total, results?, error?}` 반환.

## 7. 설정 & 환경

| 항목 | 위치 | 기본값 |
|---|---|---|
| 업로드 최대 크기 | `api/settings.py::MAX_UPLOAD_BYTES` | 10 MB |
| 배치 최대 장수 | `api/settings.py::MAX_BATCH_SIZE` | 10 |
| 결과/작업 TTL | `api/settings.py::RESULT_TTL_SECONDS` / `JOB_TTL_SECONDS` | 3600 s |
| cleanup 주기 | `api/settings.py::CLEANUP_INTERVAL_SECONDS` | 300 s |
| CORS 오리진 | `api/settings.py::CORS_ORIGINS` | `http://localhost:5173`, `http://127.0.0.1:5173` |
| SAM 기본 활성 | `api/settings.py::DEFAULT_USE_SAM` | `True` |
| 정적 마운트 | `api/settings.py::STATIC_MOUNT_PATH` | `/api/static` |
| 기본 YOLO 모델 | `config.py::ModelConfig.DEFAULT_YOLO_MODEL` | `yolov8n.pt` |
| 기본 confidence | `config.py::DetectionConfig.DEFAULT_CONFIDENCE_THRESHOLD` | 0.25 |
| GPU 선택 | `config.py::HardwareConfig`, `env_config.get_device()` | CUDA 가능 시 `cuda:0`, 아니면 `cpu` |

## 8. 관측성

- **structlog** — JSON 로그. `get_logger("api.main")` 등 네임스페이스.
- **X-Request-Id** — 요청 단위 상관관계. 미들웨어가 헤더 부재 시 UUID 생성, 응답에도 동일 헤더 노출(`expose_headers`).
- **성능 측정** — `utils.PerformanceUtils.measure_time` 데코레이터가 `pipeline.process_image`, `YOLOVehicleDetector.detect_vehicles`에 적용되어 경과 시간 로깅.

## 9. 확장 포인트

- **배치 job 백엔드 교체** — 현재는 in-memory dict. Redis/Celery로 치환하려면 `api/routers/jobs.py`의 `new_job`/`update_job`/`get_job`과 `detect.py::_run_batch`만 어댑터로 바꾸면 됨.
- **모델 추가** — `config.py::ModelConfig.YOLO_MODELS` / `SAM_MODELS`에 항목을 추가하고, 필요한 가중치를 `backend/models/`에 배치.
- **클래스 추가** — `config.py::DetectionConfig.VEHICLE_CLASSES` / `CLASS_NAMES_KR` / `CLASS_COLORS`에 항목 추가. YOLO가 감지하는 COCO 클래스 중 원하는 것을 매핑.
- **인증/레이트리밋** — 현재 공개 엔드포인트. `api/main.py::create_app`에서 미들웨어 추가 지점이 자연스러움.
- **영속 스토리지** — `output/api_runs/`는 TTL로 삭제됨. 영구 보존이 필요하면 S3 등 외부 스토리지로 복제하는 훅을 `run_single_detection` 반환 직후 추가.

## 10. 테스트

- `backend/tests/` — `pytest`
  - `test_api_detect.py`, `test_api_batch.py`, `test_api_meta.py` — FastAPI `TestClient` + 파이프라인 모의.
  - `test_detection_service.py` — 서비스 레이어 단위 테스트.
  - `test_bbox_utils.py`, `test_config.py` — 유틸/설정 단위 테스트.
- `backend/pytest.ini`가 `testpaths=tests`, 조용한 출력(`-q`) 지정.
