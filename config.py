"""
프로젝트 전역 설정 파일
모든 상수와 설정값을 중앙 관리
"""

import os
from pathlib import Path

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# 디렉토리 생성
for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# 모델 설정
class ModelConfig:
    """모델 관련 설정"""
    
    # YOLO 모델 옵션
    YOLO_MODELS = {
        'nano': 'yolov8n.pt',      # 가장 빠름, 6.2MB
        'small': 'yolov8s.pt',     # 22.5MB
        'medium': 'yolov8m.pt',    # 49.7MB
        'large': 'yolov8l.pt',     # 83.7MB
        'xlarge': 'yolov8x.pt'     # 136.7MB
    }
    
    # 기본 YOLO 모델
    DEFAULT_YOLO_MODEL = YOLO_MODELS['nano']
    
    # SAM 모델 설정
    SAM_MODELS = {
        'vit_h': 'sam_vit_h_4b8939.pth',  # ViT-H, 최고 성능
        'vit_l': 'sam_vit_l_0b3195.pth',  # ViT-L, 균형
        'vit_b': 'sam_vit_b_01ec64.pth'   # ViT-B, 가장 빠름
    }
    
    # 기본 SAM 모델
    DEFAULT_SAM_MODEL = SAM_MODELS['vit_h']
    
    # SAM 다운로드 URL
    SAM_URLS = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }

# 감지 설정
class DetectionConfig:
    """감지 관련 설정"""
    
    # 신뢰도 임계값
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    HIGH_CONFIDENCE_THRESHOLD = 0.5
    LOW_CONFIDENCE_THRESHOLD = 0.1
    
    # NMS 임계값
    NMS_THRESHOLD = 0.45
    
    # 차량 관련 클래스 (COCO 데이터셋 기준)
    VEHICLE_CLASSES = {
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7
    }
    
    # 클래스별 한글 이름
    CLASS_NAMES_KR = {
        'bicycle': '자전거',
        'car': '자동차',
        'motorcycle': '오토바이',
        'bus': '버스',
        'truck': '트럭'
    }
    
    # 클래스별 색상 (BGR 형식)
    CLASS_COLORS = {
        'bicycle': (255, 0, 0),      # 빨간색
        'car': (0, 255, 0),          # 초록색
        'motorcycle': (0, 0, 255),   # 파란색
        'bus': (255, 255, 0),        # 노란색
        'truck': (255, 0, 255)       # 마젠타색
    }

# 시각화 설정
class VisualizationConfig:
    """시각화 관련 설정"""
    
    # 이미지 크기
    FIGURE_SIZE = (15, 8)
    SINGLE_FIGURE_SIZE = (10, 8)
    
    # 폰트 설정
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2
    
    # 색상
    TEXT_COLOR = (255, 255, 255)  # 흰색
    BACKGROUND_COLOR = (0, 0, 0)  # 검은색
    
    # 마스크 투명도
    MASK_ALPHA = 0.3
    IMAGE_ALPHA = 0.7

# 파일 설정
class FileConfig:
    """파일 관련 설정"""
    
    # 지원하는 이미지 형식
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 기본 파일명
    DEFAULT_INPUT_NAME = "input_image"
    DEFAULT_OUTPUT_NAME = "detection_result"
    
    # 출력 파일 형식
    OUTPUT_IMAGE_FORMAT = ".jpg"
    OUTPUT_TEXT_FORMAT = ".txt"
    
    # 저장 품질
    IMAGE_QUALITY = 95
    DPI = 300

# GPU 설정
class HardwareConfig:
    """하드웨어 관련 설정"""
    
    # GPU 사용 여부 (자동 감지)
    USE_GPU = True
    
    # CUDA 디바이스
    CUDA_DEVICE = "cuda:0"
    CPU_DEVICE = "cpu"
    
    # 배치 크기
    BATCH_SIZE = 1
    
    # 워커 수
    NUM_WORKERS = 0

# 로그 설정
class LogConfig:
    """로깅 관련 설정"""
    
    # 로그 레벨
    LOG_LEVEL = "INFO"
    
    # 로그 형식
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 로그 파일
    LOG_FILE = OUTPUT_DIR / "detection.log"

# 환경 설정 확인
class EnvironmentConfig:
    """환경 설정 확인"""
    
    @staticmethod
    def check_dependencies():
        """필수 의존성 확인"""
        required_packages = [
            'cv2',
            'numpy', 
            'matplotlib',
            'torch',
            'ultralytics'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        return missing_packages
    
    @staticmethod
    def get_device():
        """사용 가능한 디바이스 반환"""
        try:
            import torch
            if torch.cuda.is_available() and HardwareConfig.USE_GPU:
                return torch.device(HardwareConfig.CUDA_DEVICE)
            else:
                return torch.device(HardwareConfig.CPU_DEVICE)
        except ImportError:
            return HardwareConfig.CPU_DEVICE

# 개발 설정
class DevelopmentConfig:
    """개발 환경 설정"""
    
    DEBUG = True
    VERBOSE = True
    SAVE_INTERMEDIATE_RESULTS = True
    
    # 테스트 이미지
    TEST_IMAGES = [
        "korean_car.jpg",
        "sample_vehicle.jpg",
        "test_image.jpg"
    ]

# 전역 설정 객체들
model_config = ModelConfig()
detection_config = DetectionConfig()
viz_config = VisualizationConfig()
file_config = FileConfig()
hardware_config = HardwareConfig()
log_config = LogConfig()
env_config = EnvironmentConfig()
dev_config = DevelopmentConfig()

# 설정 검증
def validate_config():
    """설정값 검증"""
    assert 0 <= detection_config.DEFAULT_CONFIDENCE_THRESHOLD <= 1, "신뢰도 임계값은 0-1 사이여야 합니다"
    assert 0 <= viz_config.MASK_ALPHA <= 1, "마스크 투명도는 0-1 사이여야 합니다"
    assert file_config.IMAGE_QUALITY > 0, "이미지 품질은 0보다 커야 합니다"

# 설정 출력
def print_config():
    """현재 설정 출력"""
    print("=" * 50)
    print("VEHICLE DETECTION PROJECT CONFIG")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Default YOLO Model: {model_config.DEFAULT_YOLO_MODEL}")
    print(f"Default SAM Model: {model_config.DEFAULT_SAM_MODEL}")
    print(f"Confidence Threshold: {detection_config.DEFAULT_CONFIDENCE_THRESHOLD}")
    print(f"Device: {env_config.get_device()}")
    print("=" * 50)

if __name__ == "__main__":
    # 설정 검증 및 출력
    validate_config()
    print_config()
    
    # 의존성 확인
    missing = env_config.check_dependencies()
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("\n✅ All dependencies are installed!")
