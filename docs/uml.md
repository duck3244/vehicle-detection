# UML Diagrams

Mermaid 기반 UML 스니펫 모음입니다. GitHub/VS Code/IntelliJ Markdown 미리보기에서 바로 렌더됩니다. 상위 구조는 [architecture.md](./architecture.md) 참조.

## 1. Component Diagram

시스템 상위 구성요소와 경계를 나타냅니다.

```mermaid
graph LR
    subgraph Browser["Browser (:5173)"]
        UI["React SPA<br/>Vite · Tailwind · shadcn"]
        RQ["@tanstack/react-query"]
        KV["react-konva Canvas"]
        UI --> RQ
        UI --> KV
    end

    subgraph Backend["FastAPI (:8000)"]
        direction TB
        MW["Middleware<br/>CORS · RequestId"]
        subgraph Routers
            R1["routers/detect"]
            R2["routers/jobs"]
            R3["routers/meta"]
        end
        SVC["services/detection_service"]
        PIPE["VehicleDetectionPipeline"]
        YOLO["YOLOVehicleDetector<br/>(ultralytics)"]
        SAM["SAMSegmentor<br/>(segment-anything)"]
        STATIC["StaticFiles<br/>/api/static → output/api_runs"]
        JOBS[("app.state.jobs<br/>in-memory dict")]

        MW --> Routers
        R1 --> SVC
        R2 --> JOBS
        R3 --> PIPE
        SVC --> PIPE
        PIPE --> YOLO
        PIPE --> SAM
    end

    FS[("Filesystem<br/>output/api_runs/<run_id>/")]
    W[("Model weights<br/>yolov8*.pt · sam_vit_*.pth")]

    UI -- "HTTP /api/*" --> MW
    UI -- "GET /api/static/*" --> STATIC
    SVC --> FS
    STATIC --> FS
    YOLO --> W
    SAM --> W
```

## 2. Class Diagram — 백엔드 핵심

서비스/도메인 계층의 주요 타입과 관계.

```mermaid
classDiagram
    class VehicleDetectionPipeline {
        +str yolo_model_name
        +str sam_model_type
        +float confidence_threshold
        +bool enable_sam
        +YOLOVehicleDetector yolo_detector
        +SAMSegmentor sam_segmentor
        +dict pipeline_stats
        +process_image(path, save, show, refine) Dict
        +process_batch(path, pattern, max) Dict
        +update_settings(**kwargs)
        +get_pipeline_statistics() Dict
    }

    class YOLOVehicleDetector {
        +str model_name
        +float conf_threshold
        +str device
        +YOLO model
        +detect_vehicles(image, conf) List~Dict~
        +visualize_detections(image, dets) ndarray
        +filter_detections(dets, ...) List~Dict~
        +update_confidence_threshold(t)
    }

    class SAMSegmentor {
        +str model_type
        +Path model_path
        +device device
        +Sam sam_model
        +SamPredictor predictor
        +set_image(image) bool
        +segment_from_boxes(boxes) List~ndarray~
        +visualize_masks(masks) ndarray
    }

    class Detection {
        +str class_name
        +str class_kr
        +float score
        +List~int~ bbox
        +Path? mask_path
    }

    class DetectionResult {
        +str run_id
        +Path run_dir
        +Path annotated_image_path
        +List~Detection~ detections
        +int inference_ms
        +Dict meta
    }

    class PipelineLike {
        <<Protocol>>
        +yolo_detector
        +sam_segmentor
        +bool enable_sam
        +str yolo_model_name
        +float confidence_threshold
    }

    class InvalidImageError {
        <<Exception>>
    }

    class detection_service {
        <<module>>
        +decode_image(bytes) ndarray
        +run_single_detection(bytes, pipe, out, use_sam, conf) DetectionResult
        +run_batch_detection(list, pipe, out, ...) List~DetectionResult~
    }

    VehicleDetectionPipeline *-- YOLOVehicleDetector
    VehicleDetectionPipeline *-- SAMSegmentor
    VehicleDetectionPipeline ..|> PipelineLike
    DetectionResult "1" o-- "*" Detection
    detection_service ..> VehicleDetectionPipeline : uses
    detection_service ..> DetectionResult : returns
    detection_service ..> InvalidImageError : raises
```

## 3. Class Diagram — API 스키마 / 라우터

Pydantic 응답 모델과 라우터 의존성.

```mermaid
classDiagram
    class DetectionItem {
        +str class
        +str class_kr
        +float score
        +List~int~ bbox
        +str? mask_url
    }
    class DetectMeta {
        +int inference_ms
        +int num_detections
        +bool sam_used
        +str yolo_model
        +str? device
    }
    class DetectResponse {
        +str run_id
        +List~DetectionItem~ detections
        +str annotated_image_url
        +DetectMeta meta
    }
    class BatchJobResponse {
        +str job_id
    }
    class JobStatusResponse {
        +str job_id
        +JobStatus status
        +int progress
        +int total
        +int done
        +List~DetectResponse~? results
        +str? error
    }
    class HealthResponse {
        +str status
        +dict gpu
        +dict memory
        +dict models_loaded
    }
    class ModelsResponse {
        +dict yolo
        +dict sam
        +str default_yolo
        +str default_sam
    }

    class DetectRouter {
        <<FastAPI router /api>>
        +POST /detect
        +POST /detect/batch
    }
    class JobsRouter {
        <<FastAPI router /api>>
        +GET /jobs/:job_id
        +new_job(app, total) str
        +update_job(app, id) void
        +cleanup_expired(app) void
    }
    class MetaRouter {
        <<FastAPI router /api>>
        +GET /system/health
        +GET /models
    }
    class Deps {
        <<module>>
        +get_pipeline(request) VehicleDetectionPipeline
    }

    DetectResponse *-- DetectionItem
    DetectResponse *-- DetectMeta
    JobStatusResponse o-- DetectResponse
    DetectRouter ..> Deps : Depends
    MetaRouter ..> Deps : Depends
    DetectRouter ..> DetectResponse : returns
    DetectRouter ..> BatchJobResponse : returns
    JobsRouter ..> JobStatusResponse : returns
    MetaRouter ..> HealthResponse : returns
    MetaRouter ..> ModelsResponse : returns
```

## 4. Sequence — 단일 감지 (동기)

```mermaid
sequenceDiagram
    autonumber
    participant UI as React (SinglePage)
    participant AX as axios /api
    participant FA as FastAPI
    participant DR as routers/detect
    participant SV as detection_service
    participant PI as Pipeline (singleton)
    participant YO as YOLO
    participant SM as SAM
    participant FS as output/api_runs/<run_id>

    UI->>AX: POST /detect (multipart: file, use_sam, confidence)
    AX->>FA: HTTP
    FA->>DR: detect(file, use_sam, confidence, pipeline=Depends)
    DR->>DR: _validate_upload + _read_bounded (<=10MB)
    DR->>SV: asyncio.to_thread(run_single_detection, bytes, pipe, out)
    SV->>SV: decode_image (Pillow verify → cv2 BGR)
    SV->>YO: detect_vehicles(image, conf)
    YO-->>SV: [{class, bbox, confidence, ...}]
    alt use_sam & detections non-empty
        SV->>SM: set_image(image)
        SM-->>SV: True
        SV->>SM: segment_from_boxes(bboxes)
        SM-->>SV: [mask ndarray ...]
    end
    SV->>YO: visualize_detections(image, dets)
    YO-->>SV: annotated ndarray
    SV->>FS: cv2.imwrite annotated.jpg + mask_i.png
    SV-->>DR: DetectionResult
    DR->>DR: build_detect_response (URL 합성)
    DR-->>FA: DetectResponse
    FA-->>AX: 200 JSON
    AX-->>UI: data
    UI->>FA: GET /api/static/<run_id>/annotated.jpg (또는 mask PNG)
    FA-->>UI: image bytes
    UI->>UI: DetectionCanvas 렌더 (bbox/mask 오버레이)
```

## 5. Sequence — 배치 감지 (비동기 job)

```mermaid
sequenceDiagram
    autonumber
    participant UI as React (BatchPage)
    participant FA as FastAPI
    participant DR as routers/detect
    participant JR as routers/jobs
    participant BG as BackgroundTasks
    participant SV as detection_service
    participant PI as Pipeline

    UI->>FA: POST /detect/batch (files[1..N], use_sam, confidence)
    FA->>DR: detect_batch(...)
    DR->>DR: validate each (ext/size), read payloads
    DR->>JR: new_job(app, total=N)
    JR-->>DR: job_id
    DR->>BG: add_task(_run_batch, base_url, app, pipe, job_id, payloads)
    DR-->>UI: 202 {job_id}

    rect rgba(200,220,255,0.25)
        note over BG: 백그라운드 태스크 (이벤트 루프)
        BG->>JR: update_job(job_id, status=running)
        loop for payload in payloads
            BG->>SV: asyncio.to_thread(run_single_detection, payload)
            SV-->>BG: DetectionResult or InvalidImageError
            BG->>JR: update_job(done=k, progress=pct)
        end
        BG->>JR: update_job(status=done, results=[...])
    end

    loop 1s polling via useJob
        UI->>FA: GET /jobs/{job_id}
        FA->>JR: get_job
        JR-->>UI: {status, progress, done, total, results?}
        alt status == done or failed
            UI->>UI: 폴링 중지, 결과 테이블 렌더
        end
    end
```

## 6. Sequence — TTL cleanup 루프

```mermaid
sequenceDiagram
    autonumber
    participant LS as lifespan
    participant CL as _cleanup_loop
    participant JR as routers/jobs.cleanup_expired
    participant ST as app.state.jobs + filesystem

    LS->>CL: asyncio.create_task(_cleanup_loop(app))
    loop every CLEANUP_INTERVAL_SECONDS (300s)
        CL->>CL: await asyncio.sleep(300)
        CL->>JR: cleanup_expired(app)
        JR->>ST: async with jobs_lock
        JR->>ST: pop jobs older than JOB_TTL_SECONDS
        JR->>ST: unlink/rmdir run_dir older than RESULT_TTL_SECONDS
        ST-->>JR: ok
    end
    note over LS: shutdown 시 task.cancel() → CancelledError 전파
```

## 7. State — Batch Job 상태 머신

```mermaid
stateDiagram-v2
    [*] --> pending: new_job(app, total=N)
    pending --> running: _run_batch 시작
    running --> running: update_job(done++, progress)
    running --> done: 모든 payload 처리 완료
    running --> failed: 예기치 않은 예외
    done --> [*]: TTL 경과 후 dict에서 pop
    failed --> [*]: TTL 경과 후 dict에서 pop
```

> 개별 이미지의 `InvalidImageError`는 job을 `failed`로 전이시키지 않고, 해당 슬롯에 `{"error": "..."}` 객체로 기록된 뒤 계속 진행.

## 8. Deployment (논리 뷰)

```mermaid
graph TB
    subgraph Dev["개발 머신"]
        subgraph FE["frontend (Vite dev, :5173)"]
            V["vite dev server"]
        end
        subgraph BE["backend (uvicorn, :8000)"]
            U["uvicorn api.main:app"]
            FS[("output/api_runs/")]
            MD[("backend/models/")]
        end
        V -- "proxy /api" --> U
        U --> FS
        U --> MD
    end

    Browser --> V
```

프로덕션에서는 Vite 빌드 결과(`frontend/dist`)를 CDN 혹은 FastAPI `StaticFiles`로 서빙하고, uvicorn은 gunicorn/uvicorn-worker 뒤에 배치하는 형태로 확장 가능합니다.

## 9. 프론트엔드 컴포넌트 (요약)

```mermaid
graph TD
    App --> Tabs
    Tabs --> SinglePage
    Tabs --> BatchPage

    SinglePage --> Dropzone
    SinglePage --> DetectionOptions
    SinglePage --> DetectionCanvas
    SinglePage --> DetectionList
    SinglePage --> hSingle["useDetectSingle()"]

    BatchPage --> Dropzone
    BatchPage --> DetectionOptions
    BatchPage --> hBatch["useDetectBatch()"]
    BatchPage --> hJob["useJob(jobId) (1s polling)"]

    hSingle --> client["api/client.ts\n(axios + openapi types)"]
    hBatch --> client
    hJob --> client
```
