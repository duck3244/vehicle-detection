"""
YOLO 기반 차량 감지 모듈
YOLOv8을 사용하여 이미지에서 차량을 감지
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics not installed. Run: pip install ultralytics")

from config import (
    detection_config, model_config, hardware_config, 
    viz_config, env_config
)
from utils import (
    ImageUtils, BboxUtils, ValidationUtils, 
    PerformanceUtils, logger, TextUtils
)

class YOLOVehicleDetector:
    """YOLO 기반 차량 감지기"""
    
    def __init__(self, 
                 model_name: str = None,
                 conf_threshold: float = None,
                 device: str = None):
        """
        초기화
        
        Args:
            model_name: YOLO 모델 이름 ('yolov8n.pt', 'yolov8s.pt', etc.)
            conf_threshold: 신뢰도 임계값
            device: 사용할 디바이스 ('cuda', 'cpu', 'auto')
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics package is required. Install with: pip install ultralytics")
        
        # 기본값 설정
        self.model_name = model_name or model_config.DEFAULT_YOLO_MODEL
        self.conf_threshold = conf_threshold or detection_config.DEFAULT_CONFIDENCE_THRESHOLD
        self.device = self._setup_device(device)
        
        # 모델 로드
        self.model = self._load_model()
        
        # 통계
        self.detection_stats = {
            'total_images': 0,
            'total_detections': 0,
            'class_counts': {},
            'avg_confidence': 0.0
        }
        
        logger.info(f"YOLOVehicleDetector initialized with {self.model_name} on {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """디바이스 설정"""
        if device == 'auto' or device is None:
            return str(env_config.get_device())
        elif device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _load_model(self) -> YOLO:
        """YOLO 모델 로드"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            model = YOLO(self.model_name)
            
            # 디바이스 설정
            if 'cuda' in self.device:
                model.to('cuda')
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    @PerformanceUtils.measure_time
    def detect_vehicles(self, 
                       image: Union[str, Path, np.ndarray],
                       conf_threshold: float = None,
                       return_raw_results: bool = False) -> List[Dict]:
        """
        차량 감지 실행
        
        Args:
            image: 입력 이미지 (경로 또는 numpy array)
            conf_threshold: 신뢰도 임계값 (None이면 기본값 사용)
            return_raw_results: YOLO 원본 결과도 함께 반환할지 여부
            
        Returns:
            감지된 차량 정보 리스트
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_array = ImageUtils.load_image(image)
            if image_array is None:
                return []
        else:
            image_array = image
        
        # 이미지 유효성 검사
        if not ValidationUtils.validate_image(image_array):
            logger.error("Invalid image provided")
            return []
        
        # 신뢰도 임계값 설정
        conf_threshold = conf_threshold or self.conf_threshold
        
        try:
            # YOLO 추론 실행
            logger.debug(f"Running YOLO inference with confidence threshold: {conf_threshold}")
            results = self.model.predict(
                source=image_array,
                conf=conf_threshold,
                device=self.device,
                verbose=False,
                save=False
            )
            
            # 결과 파싱
            detections = self._parse_yolo_results(results[0], image_array.shape[:2])
            
            # 통계 업데이트
            self._update_stats(detections)
            
            logger.info(f"Detected {len(detections)} vehicles")
            
            if return_raw_results:
                return detections, results[0]
            else:
                return detections
                
        except Exception as e:
            logger.error(f"Error during vehicle detection: {e}")
            return []
    
    def _parse_yolo_results(self, result, image_shape: Tuple[int, int]) -> List[Dict]:
        """YOLO 결과를 파싱하여 차량만 추출"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            class_name = self.model.names[int(cls_id)]
            
            # 차량 클래스만 필터링
            if class_name in detection_config.VEHICLE_CLASSES:
                bbox = box.astype(int)
                
                # 바운딩 박스 유효성 검사
                if ValidationUtils.validate_bbox(bbox, image_shape):
                    detection = {
                        'class': class_name,
                        'class_id': int(cls_id),
                        'confidence': float(conf),
                        'bbox': bbox,
                        'area': BboxUtils.calculate_area(bbox),
                        'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    }
                    
                    # 검증
                    if ValidationUtils.validate_detection(detection):
                        detections.append(detection)
                    else:
                        logger.warning(f"Invalid detection filtered out: {detection}")
        
        return detections
    
    def _update_stats(self, detections: List[Dict]):
        """통계 정보 업데이트"""
        self.detection_stats['total_images'] += 1
        self.detection_stats['total_detections'] += len(detections)
        
        # 클래스별 카운트
        for det in detections:
            class_name = det['class']
            self.detection_stats['class_counts'][class_name] = \
                self.detection_stats['class_counts'].get(class_name, 0) + 1
        
        # 평균 신뢰도 계산
        if detections:
            confidences = [det['confidence'] for det in detections]
            self.detection_stats['avg_confidence'] = np.mean(confidences)
    
    def detect_batch(self, 
                    image_paths: List[Union[str, Path]],
                    conf_threshold: float = None,
                    save_results: bool = True) -> Dict[str, List[Dict]]:
        """
        배치 감지
        
        Args:
            image_paths: 이미지 경로 리스트
            conf_threshold: 신뢰도 임계값
            save_results: 결과 저장 여부
            
        Returns:
            각 이미지별 감지 결과
        """
        results = {}
        
        logger.info(f"Starting batch detection for {len(image_paths)} images")
        
        for image_path in image_paths:
            logger.info(f"Processing: {image_path}")
            
            detections = self.detect_vehicles(image_path, conf_threshold)
            results[str(image_path)] = detections
            
            if save_results and detections:
                self.save_detection_result(image_path, detections)
        
        logger.info(f"Batch detection completed. Total images: {len(image_paths)}")
        return results
    
    def visualize_detections(self, 
                           image: Union[str, Path, np.ndarray],
                           detections: List[Dict] = None,
                           show_labels: bool = True,
                           show_confidence: bool = True) -> np.ndarray:
        """
        감지 결과 시각화
        
        Args:
            image: 입력 이미지
            detections: 감지 결과 (None이면 새로 감지)
            show_labels: 레이블 표시 여부
            show_confidence: 신뢰도 표시 여부
            
        Returns:
            시각화된 이미지
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_array = ImageUtils.load_image(image)
            if image_array is None:
                return None
        else:
            image_array = image.copy()
        
        # 감지 실행 (필요한 경우)
        if detections is None:
            detections = self.detect_vehicles(image_array)
        
        # 각 감지 결과에 바운딩 박스 그리기
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # 색상 가져오기
            color = detection_config.CLASS_COLORS.get(class_name, (128, 128, 128))
            
            # 레이블 생성
            label = ""
            if show_labels:
                kr_name = detection_config.CLASS_NAMES_KR.get(class_name, class_name)
                label = kr_name
                
                if show_confidence:
                    label += f": {confidence:.2f}"
            
            # 바운딩 박스 그리기
            from utils import VisualizationUtils
            image_array = VisualizationUtils.draw_bbox(
                image_array, bbox, label, color
            )
        
        return image_array
    
    def save_detection_result(self, 
                             image_path: Union[str, Path],
                             detections: List[Dict],
                             output_dir: Path = None) -> bool:
        """
        감지 결과 저장
        
        Args:
            image_path: 원본 이미지 경로
            detections: 감지 결과
            output_dir: 출력 디렉토리
            
        Returns:
            저장 성공 여부
        """
        try:
            from config import OUTPUT_DIR
            
            if output_dir is None:
                output_dir = OUTPUT_DIR
            
            # 이미지 로드 및 시각화
            image = ImageUtils.load_image(image_path)
            if image is None:
                return False
            
            visualized_image = self.visualize_detections(image, detections)
            
            # 출력 파일명 생성
            from utils import FileUtils
            output_image_path = FileUtils.create_output_filename(image_path)
            output_text_path = output_image_path.with_suffix('.txt')
            
            # 이미지 저장
            success_img = ImageUtils.save_image(visualized_image, output_image_path)
            
            # 텍스트 결과 저장
            image_info = ImageUtils.get_image_info(image)
            success_txt = TextUtils.save_detection_results(
                detections, output_text_path, image_info
            )
            
            if success_img and success_txt:
                logger.info(f"Results saved: {output_image_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error saving detection results: {e}")
            return False
    
    def filter_detections(self,
                         detections: List[Dict],
                         min_confidence: float = None,
                         min_area: int = None,
                         classes: List[str] = None) -> List[Dict]:
        """
        감지 결과 필터링
        
        Args:
            detections: 원본 감지 결과
            min_confidence: 최소 신뢰도
            min_area: 최소 면적 (픽셀)
            classes: 포함할 클래스 리스트
            
        Returns:
            필터링된 감지 결과
        """
        filtered = detections.copy()
        
        # 신뢰도 필터링
        if min_confidence is not None:
            filtered = [det for det in filtered if det['confidence'] >= min_confidence]
            logger.debug(f"Confidence filtering: {len(detections)} -> {len(filtered)}")
        
        # 면적 필터링
        if min_area is not None:
            filtered = [det for det in filtered if det['area'] >= min_area]
            logger.debug(f"Area filtering: {len(detections)} -> {len(filtered)}")
        
        # 클래스 필터링
        if classes is not None:
            filtered = [det for det in filtered if det['class'] in classes]
            logger.debug(f"Class filtering: {len(detections)} -> {len(filtered)}")
        
        return filtered
    
    def get_detection_statistics(self) -> Dict:
        """감지 통계 반환"""
        stats = self.detection_stats.copy()
        
        if stats['total_images'] > 0:
            stats['avg_detections_per_image'] = stats['total_detections'] / stats['total_images']
        else:
            stats['avg_detections_per_image'] = 0
        
        return stats
    
    def reset_statistics(self):
        """통계 초기화"""
        self.detection_stats = {
            'total_images': 0,
            'total_detections': 0,
            'class_counts': {},
            'avg_confidence': 0.0
        }
        logger.info("Detection statistics reset")
    
    def update_confidence_threshold(self, new_threshold: float):
        """신뢰도 임계값 업데이트"""
        if 0 <= new_threshold <= 1:
            old_threshold = self.conf_threshold
            self.conf_threshold = new_threshold
            logger.info(f"Confidence threshold updated: {old_threshold} -> {new_threshold}")
        else:
            logger.error("Confidence threshold must be between 0 and 1")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.conf_threshold,
            'supported_classes': list(detection_config.VEHICLE_CLASSES.keys()),
            'model_parameters': getattr(self.model.model, 'parameters', lambda: 0)() if hasattr(self.model, 'model') else 0
        }

# 편의 함수들
def detect_vehicle_in_image(image_path: Union[str, Path],
                           model_name: str = None,
                           conf_threshold: float = None,
                           save_result: bool = True,
                           show_result: bool = True) -> List[Dict]:
    """
    단일 이미지에서 차량 감지 (편의 함수)
    
    Args:
        image_path: 이미지 경로
        model_name: YOLO 모델 이름
        conf_threshold: 신뢰도 임계값
        save_result: 결과 저장 여부
        show_result: 결과 표시 여부
        
    Returns:
        감지 결과
    """
    detector = YOLOVehicleDetector(model_name, conf_threshold)
    detections = detector.detect_vehicles(image_path)
    
    if save_result and detections:
        detector.save_detection_result(image_path, detections)
    
    if show_result:
        TextUtils.print_detection_summary(detections)
        
        # 시각화된 이미지 표시
        if detections:
            import matplotlib.pyplot as plt
            
            original_image = ImageUtils.load_image(image_path)
            visualized_image = detector.visualize_detections(original_image, detections)
            
            from utils import VisualizationUtils
            fig = VisualizationUtils.create_comparison_plot(
                original_image, visualized_image,
                "Original Image", f"Detected Vehicles ({len(detections)})"
            )
            plt.show()
    
    return detections

def batch_detect_vehicles(image_directory: Union[str, Path],
                         model_name: str = None,
                         conf_threshold: float = None,
                         output_summary: bool = True) -> Dict[str, List[Dict]]:
    """
    디렉토리의 모든 이미지에서 차량 감지 (편의 함수)
    
    Args:
        image_directory: 이미지 디렉토리
        model_name: YOLO 모델 이름
        conf_threshold: 신뢰도 임계값
        output_summary: 요약 결과 출력 여부
        
    Returns:
        모든 이미지의 감지 결과
    """
    from utils import FileUtils
    
    detector = YOLOVehicleDetector(model_name, conf_threshold)
    image_files = FileUtils.get_image_files(image_directory)
    
    if not image_files:
        logger.warning(f"No image files found in {image_directory}")
        return {}
    
    results = detector.detect_batch(image_files, save_results=True)
    
    if output_summary:
        # 전체 요약 출력
        total_detections = sum(len(detections) for detections in results.values())
        total_images = len(results)
        
        print(f"\n{'='*60}")
        print(f"BATCH DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images}")
        print(f"Total vehicles detected: {total_detections}")
        print(f"Average vehicles per image: {total_detections/total_images:.1f}")
        
        # 클래스별 통계
        all_classes = {}
        for detections in results.values():
            for det in detections:
                class_name = det['class']
                all_classes[class_name] = all_classes.get(class_name, 0) + 1
        
        print(f"\nVehicle types found:")
        for class_name, count in all_classes.items():
            kr_name = detection_config.CLASS_NAMES_KR.get(class_name, class_name)
            print(f"  {kr_name} ({class_name}): {count}")
        
        print(f"{'='*60}\n")
        
        # 개별 이미지 결과
        for image_path, detections in results.items():
            if detections:
                print(f"{Path(image_path).name}: {len(detections)} vehicles")
    
    return results

# 메인 함수
def main():
    """YOLO 감지기 테스트"""
    print("🚗 YOLO Vehicle Detector Test")
    print("=" * 50)
    
    # 테스트용 이미지 경로
    test_image = "korean_car.jpg"
    
    try:
        # 감지기 초기화
        detector = YOLOVehicleDetector()
        
        # 모델 정보 출력
        model_info = detector.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # 이미지가 존재하는지 확인
        if not Path(test_image).exists():
            print(f"\n⚠️  Test image '{test_image}' not found!")
            print("Please place a vehicle image in the current directory.")
            return
        
        # 차량 감지 실행
        print(f"\nDetecting vehicles in '{test_image}'...")
        detections = detector.detect_vehicles(test_image)
        
        if detections:
            print(f"✅ Found {len(detections)} vehicle(s)!")
            
            # 감지 결과 출력
            TextUtils.print_detection_summary(detections)
            
            # 결과 저장
            success = detector.save_detection_result(test_image, detections)
            if success:
                print("📁 Results saved to output directory")
            
            # 통계 출력
            stats = detector.get_detection_statistics()
            print("\nDetection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        else:
            print("❌ No vehicles detected")
            print("Try lowering the confidence threshold or using a different image")
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Install required packages: pip install ultralytics torch")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()