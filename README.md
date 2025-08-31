# 🚗 Vehicle Detection & Segmentation System

Meta's SAM 모델과 YOLOv8을 활용한 차량 자동 감지 및 정밀 세그멘테이션 시스템입니다.  
**명령행 도구**와 **웹 인터페이스** 모두 지원합니다.

## 📋 프로젝트 구조

```
vehicle-detection/
├── config.py              # 프로젝트 설정 및 상수
├── utils.py                # 공통 유틸리티 함수
├── yolo_detector.py        # YOLO 기반 차량 감지기
├── sam_segmentor.py        # SAM 기반 세그멘테이션
├── pipeline.py             # 통합 파이프라인
├── main.py                 # 명령행 메인 애플리케이션
├── app.py                  # 🌐 Streamlit 웹 애플리케이션
├── requirements.txt        # 패키지 의존성
├── install.sh              # 자동 설치 스크립트
├── WEB_SETUP.md           # 웹 애플리케이션 가이드
├── README.md              # 프로젝트 가이드
├── data/                  # 입력 데이터
├── output/                # 출력 결과
├── models/                # 모델 파일
└── korean_car.jpg         # 테스트용 한국 차량 이미지
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 자동 설치 (권장)

```bash
# 설치 스크립트 실행 (Linux/Mac)
chmod +x install.sh
./install.sh

# Windows에서는 수동 설치
pip install -r requirements.txt
```

### 3. 시스템 확인

```bash
# 의존성 및 시스템 정보 확인
python main.py --info
python main.py --check-deps
```

## 🌐 웹 애플리케이션 (추천)

### **빠른 실행**
```bash
# 웹 서버 시작
streamlit run app.py

# 브라우저에서 접속
# http://localhost:8501
```

### **🎛️ 웹 인터페이스 특징**
- **📸 드래그 앤 드롭**: 이미지 업로드 간편화
- **⚙️ 직관적 설정**: 사이드바에서 모든 옵션 조정
- **📊 실시간 시각화**: 차트와 통계 자동 생성
- **💾 다운로드 옵션**: 텍스트/이미지/JSON 다중 형식
- **🔄 배치 처리**: 여러 이미지 동시 처리 (최대 10개)
- **📱 반응형 디자인**: 모바일/태블릿 지원
- **🇰🇷 한국어 지원**: 현지화된 UI 및 결과

### **📋 웹 인터페이스 탭 구성**
1. **📸 이미지 분석**: 메인 분석 기능
2. **📊 시스템 정보**: GPU/모델 상태 확인  
3. **📚 사용 가이드**: 상세한 사용법
4. **🔧 고급 설정**: 배치 처리 및 성능 모니터링

## 💻 명령행 인터페이스

### **기본 사용법**

```bash
# 단일 이미지 처리 (YOLO만 사용 - 빠름)
python main.py korean_car.jpg

# 고급 세그멘테이션 포함 (YOLO + SAM)
python main.py korean_car.jpg --sam

# GUI 없이 안전 실행 (서버 환경)
python main.py korean_car.jpg --no-gui

# 결과 표시 (GUI 환경에서만)
python main.py korean_car.jpg --show
```

### **고급 옵션**

```bash
# 특정 모델과 신뢰도로 처리
python main.py korean_car.jpg -m yolov8s.pt -c 0.3

# 배치 처리 (디렉토리의 모든 이미지)
python main.py images/ --batch --no-gui

# GPU 강제 사용
python main.py korean_car.jpg --device cuda

# 상세 로그 출력
python main.py korean_car.jpg --verbose --no-gui

# 시스템 벤치마크
python main.py --benchmark
```

### **Python 스크립트에서 사용**

```python
from pipeline import quick_vehicle_detection, full_vehicle_analysis

# 빠른 감지
result = quick_vehicle_detection("korean_car.jpg", confidence=0.25)

# 완전한 분석 (세그멘테이션 포함)
result = full_vehicle_analysis("korean_car.jpg", save_result=True)

print(f"감지된 차량 수: {len(result['detections'])}")
```

## 🎯 주요 기능

### 1. **다양한 차량 클래스 감지**
| 차량 유형 | 영문명 | 설명 |
|---------|--------|------|
| 🚗 자동차 | car | 일반 승용차 |
| 🏍️ 오토바이 | motorcycle | 이륜차 |
| 🚌 버스 | bus | 대형 버스 |
| 🚚 트럭 | truck | 화물차 |
| 🚲 자전거 | bicycle | 자전거 |

### 2. **다양한 YOLO 모델 지원**
| 모델 | 크기 | 속도 | 정확도 | 용도 |
|------|------|------|--------|------|
| YOLOv8n | 6.2MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 실시간 처리 |
| YOLOv8s | 22.5MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 균형잡힌 성능 |
| YOLOv8m | 49.7MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 높은 정확도 |
| YOLOv8l | 83.7MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 매우 높은 성능 |
| YOLOv8x | 136.7MB | ⚡ | ⭐⭐⭐⭐⭐ | 최고 성능 |

### 3. **정밀한 세그멘테이션 (SAM)**
- **ViT-H**: 최고 품질, 느림 (2.6GB)
- **ViT-L**: 균형잡힌 성능 (1.3GB)  
- **ViT-B**: 빠른 처리 (375MB)
- **픽셀 단위** 정확한 차량 형태 추출
- **마스크 후처리** 및 정제 기능

### 4. **다양한 출력 형식**
- **📷 감지 결과 이미지**: 바운딩 박스 표시
- **🎯 세그멘테이션 마스크**: 픽셀 단위 정확한 형태
- **📄 상세한 텍스트 리포트**: 한국어/영어 지원
- **📊 JSON 데이터**: API 연동용 구조화된 데이터
- **💾 NumPy 마스크**: 추가 분석용 raw 데이터

## 📊 예상 결과

첨부된 한국 차량 이미지에 대해:

### **콘솔 출력**
```
==================================================
VEHICLE DETECTION SUMMARY  
==================================================
Total vehicles detected: 1
--------------------------------------------------
자동차 (car): 1
--------------------------------------------------
Vehicle 1: 자동차 (confidence: 0.892)
==================================================
```

### **생성되는 파일들**
```
output/
├── korean_car_yolo_detection.jpg      # 바운딩 박스 감지 결과
├── korean_car_sam_segmentation.jpg    # 세그멘테이션 마스크 (SAM 사용시)
├── korean_car_combined.jpg            # 감지 + 세그멘테이션 결합
├── korean_car_results.txt             # 상세한 감지 정보
├── korean_car_masks.npz               # 마스크 데이터 (NumPy)
└── processing_summary.json            # 처리 요약 정보
```

## ⚙️ 설정 및 커스터마이징

### **신뢰도 임계값 조정**

| 임계값 | 효과 | 사용 사례 |
|--------|------|-----------|
| 0.1-0.2 | 많은 감지, 거짓 양성 증가 | 탐지 누락 최소화 |
| 0.25 (기본) | 균형잡힌 감지 | 일반적인 용도 |
| 0.4-0.6 | 확실한 감지만, 누락 가능 | 높은 정밀도 필요 |

### **사용 시나리오별 최적 설정**

#### **🚦 실시간 교통 모니터링**
```bash
python main.py traffic.jpg -m yolov8n.pt -c 0.3 --device cuda --no-gui
```
- **모델**: YOLOv8 Nano (속도 우선)
- **SAM**: 비활성화 (빠른 처리)
- **신뢰도**: 0.3 (적당한 감지율)

#### **🅿️ 정밀한 주차장 분석**  
```bash
python main.py parking.jpg -m yolov8l.pt --sam -c 0.15 --no-gui
```
- **모델**: YOLOv8 Large (정확도 우선)
- **SAM**: ViT-H (최고 품질)
- **신뢰도**: 0.15 (누락 방지)

#### **📊 대량 데이터 분석**
```bash  
python main.py dataset/ --batch -m yolov8m.pt -c 0.25 --no-gui
```
- **배치 처리**: 디렉토리 전체
- **모델**: YOLOv8 Medium (균형)
- **자동 저장**: 모든 결과 파일 저장

## 🛠️ 문제 해결

### **일반적인 문제**

#### **1. GUI 관련 오류**
```bash
# 문제: can't invoke "wm" command
# 해결: GUI 없이 실행
python main.py image.jpg --no-gui
```

#### **2. CUDA 메모리 부족**
```bash
# 문제: CUDA out of memory
# 해결: CPU 모드 사용
python main.py image.jpg --device cpu
```

#### **3. 모델 다운로드 실패**
```bash
# 문제: 네트워크 오류로 모델 다운로드 실패
# 해결: 수동 다운로드
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### **4. SAM 설치 문제**
```bash
# 문제: SAM 패키지 없음
# 해결: 수동 설치 또는 YOLO만 사용
pip install git+https://github.com/facebookresearch/segment-anything.git
# 또는
python main.py image.jpg  # SAM 없이 YOLO만 사용
```

#### **5. 낮은 감지 성능**
- **이미지 해상도** 확인 (최소 640x640 권장)
- **신뢰도 임계값** 낮추기: `-c 0.1`
- **더 큰 모델** 사용: `-m yolov8l.pt`
- **적절한 조명** 및 각도 확인

### **성능 최적화**

#### **빠른 처리가 필요한 경우**
```bash
python main.py image.jpg -m yolov8n.pt -c 0.4 --device cuda --no-gui
```

#### **높은 정확도가 필요한 경우**  
```bash
python main.py image.jpg -m yolov8x.pt --sam --sam-model vit_h -c 0.1 --no-gui
```

#### **배치 처리 최적화**
```bash
python main.py images/ --batch --max-images 50 -m yolov8s.pt --no-gui
```

## 📈 확장 가능성

### **1. 새로운 차량 클래스 추가**
```python
# config.py에서 클래스 정의 추가
VEHICLE_CLASSES = {
    'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7,
    'trailer': 8,      # 새로운 클래스 추가
    'ambulance': 9,    # 구급차
    'police_car': 10   # 경찰차
}

CLASS_NAMES_KR = {
    'bicycle': '자전거', 'car': '자동차', 'motorcycle': '오토바이',
    'bus': '버스', 'truck': '트럭',
    'trailer': '트레일러',    # 새로운 한글명
    'ambulance': '구급차',   # 구급차
    'police_car': '경찰차'   # 경찰차
}
```

### **2. REST API 서버 구축**
```python
from flask import Flask, request, jsonify
from pipeline import quick_vehicle_detection

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_vehicles():
    file = request.files['image']
    result = quick_vehicle_detection(file)
    return jsonify({
        'vehicles': len(result['detections']),
        'processing_time': result['processing_time'],
        'detections': result['detections']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### **3. 실시간 비디오 처리**
```python
import cv2
from pipeline import VehicleDetectionPipeline

pipeline = VehicleDetectionPipeline()
cap = cv2.VideoCapture(0)  # 웹캠 또는 비디오 파일

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 프레임별 차량 감지
    detections = pipeline.yolo_detector.detect_vehicles(frame)
    
    # 결과 오버레이
    result_frame = pipeline.yolo_detector.visualize_detections(frame, detections)
    
    cv2.imshow('Vehicle Detection', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### **4. 클라우드 배포**
```bash
# Docker 컨테이너 생성
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]

# 실행
docker build -t vehicle-detection .
docker run -p 8501:8501 vehicle-detection
```

## 📄 API 레퍼런스

### **주요 클래스**

#### **VehicleDetectionPipeline**
```python
pipeline = VehicleDetectionPipeline(
    yolo_model='yolov8n.pt',      # YOLO 모델 파일
    sam_model='vit_h',            # SAM 모델 타입  
    confidence_threshold=0.25,     # 신뢰도 임계값
    enable_sam=True               # SAM 활성화 여부
)

# 단일 이미지 처리
result = pipeline.process_image('image.jpg')

# 배치 처리
results = pipeline.process_batch('images/', max_images=10)
```

#### **YOLOVehicleDetector**  
```python
detector = YOLOVehicleDetector(
    model_name='yolov8n.pt',     # 모델 파일
    conf_threshold=0.25,         # 신뢰도 임계값
    device='auto'                # 디바이스 설정
)

# 차량 감지
detections = detector.detect_vehicles('image.jpg')

# 시각화
result_image = detector.visualize_detections('image.jpg', detections)
```

#### **SAMSegmentor**
```python
segmentor = SAMSegmentor(
    model_type='vit_h',          # SAM 모델 타입
    device='auto'                # 디바이스 설정
)

# 이미지 설정
segmentor.set_image('image.jpg')

# 바운딩 박스에서 세그멘테이션
masks = segmentor.segment_from_boxes(bounding_boxes)
```

### **편의 함수**

#### **빠른 감지**
```python
from pipeline import quick_vehicle_detection

result = quick_vehicle_detection(
    'image.jpg',                  # 이미지 경로
    yolo_model='yolov8n.pt',     # YOLO 모델
    confidence=0.25,             # 신뢰도
    save_result=True,            # 결과 저장
    show_result=False            # 결과 표시
)
```

#### **완전한 분석**
```python  
from pipeline import full_vehicle_analysis

result = full_vehicle_analysis(
    'image.jpg',                 # 이미지 경로
    yolo_model='yolov8n.pt',    # YOLO 모델
    sam_model='vit_h',          # SAM 모델
    confidence=0.25,            # 신뢰도
    save_result=True,           # 결과 저장
    show_result=False           # 결과 표시
)
```

---

## 🎉 **두 가지 인터페이스로 완벽한 차량 감지!**

### **🌐 웹 애플리케이션** (추천)
```bash
streamlit run app.py
# → http://localhost:8501
```
**드래그 앤 드롭으로 간편하게!**

### **💻 명령행 도구** (고급 사용자)  
```bash
python main.py korean_car.jpg --sam --no-gui
```
**스크립트 및 자동화에 최적!**

**Happy Vehicle Detection! 🚗🔍**

*Made with ❤️ for intelligent transportation systems*파일
- `korean_car_yolo_detection.jpg`: 바운딩 박스가 그려진 감지 결과
- `korean_car_sam_segmentation.jpg`: 세그멘테이션 마스크 (SAM 사용시)
- `korean_car_combined.jpg`: 감지 + 세그멘테이션 결합 결과
- `korean_car_results.txt`: 상세한 감지 정보
- `korean_car_masks.npz`: 마스크 데이터 (NumPy 형식)

## ⚙️ 설정 및 커스터마이징

### 신뢰도 임계값 조정
```python
# 더 민감한 감지 (더 많은 객체 감지, 거짓 양성 가능)
python main.py image.jpg -c 0.1

# 더 보수적인 감지 (확실한 객체만)  
python main.py image.jpg -c 0.5
```

### 배치 처리 패턴
```bash
# JPG 파일만 처리
python main.py images/ --batch --pattern "*.jpg"

# 특정 접두사가 있는 파일만
python main.py images/ --batch --pattern "car_*.jpg"
```

### 성능 벤치마크
```bash
# 시스템 성능 테스트
python main.py --benchmark
```

## 🛠️ 문제 해결

### 일반적인 문제

**1. CUDA 메모리 부족**
```bash
# CPU 사용 강제
python main.py image.jpg --device cpu
```

**2. 모델 다운로드 실패**
```bash
# 수동 다운로드
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**3. SAM 설치 문제**
```bash
# SAM 없이 사용 (YOLO만)
python main.py image.jpg  # SAM 비활성화됨
```

**4. 낮은 감지 성능**
- 이미지 해상도 확인 (최소 640x640 권장)
- 신뢰도 임계값 낮추기: `-c 0.1`
- 더 큰 모델 사용: `-m yolov8l.pt`

### 성능 최적화

**GPU 메모리 최적화**
```python
import torch
torch.cuda.empty_cache()  # GPU 메모리 정리
```

**이미지 전처리**
```python
# 큰 이미지는 리사이즈하여 속도 향상
from utils import ImageUtils
resized_image = ImageUtils.resize_image(image, max_size=1024)
```

## 📈 확장 가능성

### 1. 새로운 차량 클래스 추가
```python
# config.py에서 클래스 정의 추가
VEHICLE_CLASSES = {
    'bicycle': 1,
    'car': 2, 
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'trailer': 8,  # 새로운 클래스 추가
}
```

### 2. 웹 API 서버
```python
from flask import Flask, request, jsonify
from pipeline import quick_vehicle_detection

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_vehicles():
    # 이미지 업로드 처리
    file = request.files['image']
    result = quick_vehicle_detection(file)
    return jsonify(result)
```

### 3. 실시간 비디오 처리
```python
import cv2
from pipeline import VehicleDetectionPipeline

pipeline = VehicleDetectionPipeline()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detections = pipeline.yolo_detector.detect_vehicles(frame)
    # 결과 표시 로직
```

## 📄 API 레퍼런스

### 주요 클래스

**VehicleDetectionPipeline**
```python
pipeline = VehicleDetectionPipeline(
    yolo_model='yolov8n.pt',
    sam_model='vit_h', 
    confidence_threshold=0.25,
    enable_sam=True
)

result = pipeline.process_image('image.jpg')
```

**YOLOVehicleDetector**  
```python
detector = YOLOVehicleDetector(model_name='yolov8n.pt')
detections = detector.detect_vehicles('image.jpg')
```

**SAMSegmentor**
```python
segmentor = SAMSegmentor(model_type='vit_h')
segmentor.set_image('image.jpg')
masks = segmentor.segment_from_boxes(bounding_boxes)
```
