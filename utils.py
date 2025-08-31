"""
유틸리티 함수 모음
공통으로 사용되는 헬퍼 함수들을 정의
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import requests
from tqdm import tqdm
import json
import time
from datetime import datetime

from config import (
    file_config, viz_config, model_config, 
    detection_config, log_config, OUTPUT_DIR, MODELS_DIR
)

# 로거 설정
def setup_logger(name: str = __name__) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.LOG_LEVEL))
    
    # 핸들러가 이미 있으면 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포매터
    formatter = logging.Formatter(log_config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 전역 로거
logger = setup_logger()

# 파일 관련 유틸리티
class FileUtils:
    """파일 처리 관련 유틸리티"""
    
    @staticmethod
    def is_image_file(file_path: Union[str, Path]) -> bool:
        """이미지 파일인지 확인"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in file_config.SUPPORTED_IMAGE_FORMATS
    
    @staticmethod
    def get_image_files(directory: Union[str, Path]) -> List[Path]:
        """디렉토리에서 모든 이미지 파일 찾기"""
        directory = Path(directory)
        image_files = []
        
        if directory.is_dir():
            for ext in file_config.SUPPORTED_IMAGE_FORMATS:
                image_files.extend(directory.glob(f"*{ext}"))
                image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    @staticmethod
    def create_output_filename(input_path: Union[str, Path], 
                              suffix: str = "_result") -> Path:
        """출력 파일명 생성"""
        input_path = Path(input_path)
        output_name = f"{input_path.stem}{suffix}{file_config.OUTPUT_IMAGE_FORMAT}"
        return OUTPUT_DIR / output_name
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """디렉토리 생성 보장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """파일 크기 반환 (bytes)"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict:
        """파일 정보 반환"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.suffix.lower()
        }

# 이미지 처리 유틸리티
class ImageUtils:
    """이미지 처리 관련 유틸리티"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """이미지 로드"""
        try:
            image_path = str(image_path)
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, 
                   output_path: Union[str, Path],
                   quality: int = None) -> bool:
        """이미지 저장"""
        try:
            output_path = str(output_path)
            
            # RGB to BGR 변환
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # 품질 설정
            if quality is None:
                quality = file_config.IMAGE_QUALITY
            
            # 저장
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                success = cv2.imwrite(output_path, image_bgr, 
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                success = cv2.imwrite(output_path, image_bgr)
            
            if success:
                logger.info(f"Image saved: {output_path}")
                return True
            else:
                logger.error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving image {output_path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, 
                     max_size: int = 1024, 
                     maintain_aspect: bool = True) -> np.ndarray:
        """이미지 리사이즈"""
        h, w = image.shape[:2]
        
        if max(h, w) <= max_size:
            return image
        
        if maintain_aspect:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
        else:
            new_h, new_w = max_size, max_size
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict:
        """이미지 정보 추출"""
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'aspect_ratio': w / h,
            'total_pixels': w * h,
            'dtype': str(image.dtype),
            'size_mb': (w * h * channels) / (1024 * 1024)
        }
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0-1 범위)"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """이미지 역정규화 (0-255 범위)"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            return (image * 255).astype(np.uint8)
        return image

# 바운딩 박스 유틸리티
class BboxUtils:
    """바운딩 박스 처리 유틸리티"""
    
    @staticmethod
    def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
        """XYXY 형식을 XYWH 형식으로 변환"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return np.array([x1, y1, w, h])
    
    @staticmethod
    def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
        """XYWH 형식을 XYXY 형식으로 변환"""
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h])
    
    @staticmethod
    def calculate_area(bbox: np.ndarray) -> float:
        """바운딩 박스 면적 계산"""
        if len(bbox) == 4:
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # XYXY 형식
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            else:  # XYWH 형식
                return bbox[2] * bbox[3]
        return 0.0
    
    @staticmethod
    def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """두 바운딩 박스의 IoU 계산"""
        # XYXY 형식으로 변환
        if len(bbox1) == 4 and bbox1[2] < bbox1[0]:
            bbox1 = BboxUtils.xywh_to_xyxy(bbox1)
        if len(bbox2) == 4 and bbox2[2] < bbox2[0]:
            bbox2 = BboxUtils.xywh_to_xyxy(bbox2)
        
        # 교집합 계산
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = BboxUtils.calculate_area(bbox1)
        area2 = BboxUtils.calculate_area(bbox2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def get_bbox_center(bbox: np.ndarray) -> Tuple[int, int]:
        """바운딩 박스 중심점 계산"""
        if len(bbox) == 4:
            return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        return (0, 0)

# 시각화 유틸리티
class VisualizationUtils:
    """시각화 관련 유틸리티"""
    
    @staticmethod
    def draw_bbox(image: np.ndarray,
                  bbox: np.ndarray,
                  label: str = "",
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = None) -> np.ndarray:
        """바운딩 박스 그리기"""
        if thickness is None:
            thickness = viz_config.BOX_THICKNESS
        
        image = image.copy()
        x1, y1, x2, y2 = bbox.astype(int)
        
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 레이블 그리기
        if label:
            font_scale = viz_config.FONT_SCALE
            font_thickness = viz_config.FONT_THICKNESS
            
            # 텍스트 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # 텍스트 배경
            cv2.rectangle(image, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)
            
            # 텍스트
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       viz_config.TEXT_COLOR, font_thickness)
        
        return image
    
    @staticmethod
    def create_comparison_plot(original: np.ndarray,
                              result: np.ndarray,
                              title1: str = "Original",
                              title2: str = "Result") -> plt.Figure:
        """비교 플롯 생성"""
        fig, axes = plt.subplots(1, 2, figsize=viz_config.FIGURE_SIZE)
        
        axes[0].imshow(original)
        axes[0].set_title(title1)
        axes[0].axis('off')
        
        axes[1].imshow(result)
        axes[1].set_title(title2)
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_detection_stats(detections: List[Dict]) -> plt.Figure:
        """감지 통계 플롯"""
        if not detections:
            return None
        
        # 클래스별 카운트
        class_counts = {}
        confidences = []
        
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(det['confidence'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 클래스별 분포
        ax1.bar(class_counts.keys(), class_counts.values())
        ax1.set_title('Vehicle Types Distribution')
        ax1.set_xlabel('Vehicle Type')
        ax1.set_ylabel('Count')
        
        # 신뢰도 분포
        ax2.hist(confidences, bins=10, alpha=0.7)
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.axvline(detection_config.DEFAULT_CONFIDENCE_THRESHOLD, 
                   color='r', linestyle='--', label='Threshold')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_grid_visualization(images: List[np.ndarray], 
                                 titles: List[str] = None,
                                 cols: int = 3) -> plt.Figure:
        """이미지 그리드 시각화"""
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for i, image in enumerate(images):
            axes[i].imshow(image)
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            axes[i].axis('off')
        
        # 빈 subplot 숨기기
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

# 모델 다운로드 유틸리티
class ModelUtils:
    """모델 다운로드 및 관리 유틸리티"""
    
    @staticmethod
    def download_file(url: str, output_path: Union[str, Path]) -> bool:
        """파일 다운로드"""
        try:
            output_path = Path(output_path)
            
            # 이미 존재하면 스킵
            if output_path.exists():
                logger.info(f"File already exists: {output_path}")
                return True
            
            logger.info(f"Downloading {url} to {output_path}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()  # 실패시 파일 삭제
            return False
    
    @staticmethod
    def download_sam_model(model_type: str = 'vit_h') -> bool:
        """SAM 모델 다운로드"""
        if model_type not in model_config.SAM_URLS:
            logger.error(f"Unknown SAM model type: {model_type}")
            return False
        
        url = model_config.SAM_URLS[model_type]
        model_path = MODELS_DIR / model_config.SAM_MODELS[model_type]
        
        return ModelUtils.download_file(url, model_path)
    
    @staticmethod
    def check_model_availability() -> Dict[str, bool]:
        """모델 사용 가능 여부 확인"""
        status = {}
        
        # YOLO 모델은 자동 다운로드되므로 항상 True
        for name, model_file in model_config.YOLO_MODELS.items():
            status[f'yolo_{name}'] = True
        
        # SAM 모델 확인
        for name, model_file in model_config.SAM_MODELS.items():
            model_path = MODELS_DIR / model_file
            status[f'sam_{name}'] = model_path.exists()
        
        return status

# 텍스트 출력 유틸리티
class TextUtils:
    """텍스트 출력 관련 유틸리티"""
    
    @staticmethod
    def save_detection_results(detections: List[Dict], 
                              output_path: Union[str, Path],
                              image_info: Dict = None) -> bool:
        """감지 결과를 텍스트 파일로 저장"""
        try:
            output_path = Path(output_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Vehicle Detection Results\n")
                f.write("=" * 50 + "\n\n")
                
                # 이미지 정보
                if image_info:
                    f.write("Image Information:\n")
                    for key, value in image_info.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # 감지 결과
                f.write(f"Total vehicles detected: {len(detections)}\n\n")
                
                # 클래스별 카운트
                class_counts = {}
                for det in detections:
                    class_name = det['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                f.write("Vehicle types:\n")
                for class_name, count in class_counts.items():
                    kr_name = detection_config.CLASS_NAMES_KR.get(class_name, class_name)
                    f.write(f"  {kr_name} ({class_name}): {count}\n")
                f.write("\n")
                
                # 개별 감지 결과
                f.write("Detection details:\n")
                for i, det in enumerate(detections, 1):
                    bbox = det['bbox']
                    kr_name = detection_config.CLASS_NAMES_KR.get(det['class'], det['class'])
                    
                    f.write(f"Vehicle {i}:\n")
                    f.write(f"  Type: {kr_name} ({det['class']})\n")
                    f.write(f"  Confidence: {det['confidence']:.3f}\n")
                    f.write(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n")
                    f.write(f"  Size: {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} pixels\n\n")
            
            logger.info(f"Detection results saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving detection results: {e}")
            return False
    
    @staticmethod
    def print_detection_summary(detections: List[Dict]):
        """감지 결과 요약 출력"""
        if not detections:
            print("No vehicles detected.")
            return
        
        print(f"\n{'='*50}")
        print(f"VEHICLE DETECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total vehicles detected: {len(detections)}")
        print(f"{'-'*50}")
        
        # 클래스별 카운트
        class_counts = {}
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            kr_name = detection_config.CLASS_NAMES_KR.get(class_name, class_name)
            print(f"{kr_name} ({class_name}): {count}")
        
        print(f"{'-'*50}")
        
        # 개별 감지 결과
        for i, det in enumerate(detections, 1):
            kr_name = detection_config.CLASS_NAMES_KR.get(det['class'], det['class'])
            print(f"Vehicle {i}: {kr_name} (confidence: {det['confidence']:.3f})")
        
        print(f"{'='*50}\n")
    
    @staticmethod
    def format_detection_report(detections: List[Dict],
                              image_info: Dict = None,
                              include_korean: bool = True) -> str:
        """감지 결과를 포맷된 리포트로 생성"""
        lines = []
        lines.append("Vehicle Detection Report")
        lines.append("=" * 50)
        
        # 이미지 정보
        if image_info:
            lines.append("\nImage Information:")
            for key, value in image_info.items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # 감지 통계
        lines.append(f"\nDetection Summary:")
        lines.append(f"Total vehicles detected: {len(detections)}")
        
        if not detections:
            lines.append("No vehicles found in the image.")
            return "\n".join(lines)
        
        # 클래스별 통계
        class_counts = {}
        confidence_sum = 0
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_sum += det['confidence']
        
        lines.append(f"Average confidence: {confidence_sum/len(detections):.3f}")
        lines.append(f"\nVehicle Types:")
        
        for class_name, count in class_counts.items():
            if include_korean:
                kr_name = detection_config.CLASS_NAMES_KR.get(class_name, class_name)
                lines.append(f"  {kr_name} ({class_name}): {count}")
            else:
                lines.append(f"  {class_name}: {count}")
        
        # 개별 감지 결과
        lines.append(f"\nDetailed Results:")
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            area = BboxUtils.calculate_area(bbox)
            
            if include_korean:
                kr_name = detection_config.CLASS_NAMES_KR.get(det['class'], det['class'])
                class_display = f"{kr_name} ({det['class']})"
            else:
                class_display = det['class']
            
            lines.append(f"  Vehicle {i}:")
            lines.append(f"    Type: {class_display}")
            lines.append(f"    Confidence: {det['confidence']:.3f}")
            lines.append(f"    Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            lines.append(f"    Size: {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} pixels")
            lines.append(f"    Area: {area:.0f} pixels²")
        
        lines.append("=" * 50)
        return "\n".join(lines)

# 한국어 텍스트 처리 유틸리티
class KoreanTextUtils:
    """한국어 텍스트 처리 관련 유틸리티"""
    
    @staticmethod
    def get_korean_vehicle_name(english_name: str) -> str:
        """영어 차량명을 한국어로 변환"""
        return detection_config.CLASS_NAMES_KR.get(english_name, english_name)
    
    @staticmethod
    def format_korean_report(detections: List[Dict]) -> str:
        """한국어 리포트 생성"""
        if not detections:
            return "감지된 차량이 없습니다."
        
        report_lines = []
        report_lines.append("차량 감지 결과")
        report_lines.append("=" * 30)
        report_lines.append(f"총 감지 차량 수: {len(detections)}대")
        report_lines.append("")
        
        # 차량 유형별 분류
        class_counts = {}
        for det in detections:
            kr_name = KoreanTextUtils.get_korean_vehicle_name(det['class'])
            class_counts[kr_name] = class_counts.get(kr_name, 0) + 1
        
        report_lines.append("차량 유형별 현황:")
        for kr_name, count in class_counts.items():
            report_lines.append(f"  {kr_name}: {count}대")
        
        report_lines.append("")
        report_lines.append("개별 차량 정보:")
        for i, det in enumerate(detections, 1):
            kr_name = KoreanTextUtils.get_korean_vehicle_name(det['class'])
            confidence_percent = det['confidence'] * 100
            report_lines.append(f"  {i}번째 차량: {kr_name} (신뢰도: {confidence_percent:.1f}%)")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def create_korean_labels(detections: List[Dict]) -> List[str]:
        """한국어 레이블 리스트 생성"""
        labels = []
        for det in detections:
            kr_name = KoreanTextUtils.get_korean_vehicle_name(det['class'])
            confidence_percent = det['confidence'] * 100
            label = f"{kr_name} {confidence_percent:.0f}%"
            labels.append(label)
        return labels

# 데이터 분석 유틸리티
class AnalysisUtils:
    """데이터 분석 관련 유틸리티"""
    
    @staticmethod
    def calculate_detection_statistics(detections: List[Dict]) -> Dict:
        """감지 결과 통계 계산"""
        if not detections:
            return {
                'total_count': 0,
                'class_distribution': {},
                'confidence_stats': {},
                'size_stats': {}
            }
        
        # 기본 통계
        total_count = len(detections)
        confidences = [det['confidence'] for det in detections]
        areas = [BboxUtils.calculate_area(det['bbox']) for det in detections]
        
        # 클래스별 분포
        class_distribution = {}
        for det in detections:
            class_name = det['class']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        # 신뢰도 통계
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
        
        # 크기 통계
        size_stats = {
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'median_area': np.median(areas)
        }
        
        return {
            'total_count': total_count,
            'class_distribution': class_distribution,
            'confidence_stats': confidence_stats,
            'size_stats': size_stats
        }
    
    @staticmethod
    def analyze_vehicle_sizes(detections: List[Dict]) -> Dict:
        """차량 크기 분석"""
        if not detections:
            return {}
        
        size_analysis = {}
        
        for det in detections:
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            class_name = det['class']
            if class_name not in size_analysis:
                size_analysis[class_name] = {
                    'widths': [],
                    'heights': [],
                    'areas': [],
                    'aspect_ratios': []
                }
            
            size_analysis[class_name]['widths'].append(width)
            size_analysis[class_name]['heights'].append(height)
            size_analysis[class_name]['areas'].append(area)
            size_analysis[class_name]['aspect_ratios'].append(aspect_ratio)
        
        # 통계 계산
        for class_name, data in size_analysis.items():
            for metric, values in data.items():
                if isinstance(values, list):
                    data[f'{metric}_mean'] = np.mean(values)
                    data[f'{metric}_std'] = np.std(values)
        
        return size_analysis
    
    @staticmethod
    def compare_detection_quality(detections1: List[Dict], 
                                detections2: List[Dict]) -> Dict:
        """두 감지 결과의 품질 비교"""
        stats1 = AnalysisUtils.calculate_detection_statistics(detections1)
        stats2 = AnalysisUtils.calculate_detection_statistics(detections2)
        
        comparison = {
            'count_diff': stats2['total_count'] - stats1['total_count'],
            'confidence_diff': stats2['confidence_stats']['mean'] - stats1['confidence_stats']['mean'],
            'quality_score1': AnalysisUtils._calculate_quality_score(detections1),
            'quality_score2': AnalysisUtils._calculate_quality_score(detections2)
        }
        
        return comparison
    
    @staticmethod
    def _calculate_quality_score(detections: List[Dict]) -> float:
        """감지 품질 점수 계산 (0-1 범위)"""
        if not detections:
            return 0.0
        
        # 신뢰도 평균과 일관성을 고려한 품질 점수
        confidences = [det['confidence'] for det in detections]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        # 높은 평균 신뢰도와 낮은 표준편차가 좋음
        quality_score = mean_conf * (1 - min(std_conf, 0.5))
        
        return max(0.0, min(1.0, quality_score))

# 성능 측정 유틸리티
class PerformanceUtils:
    """성능 측정 관련 유틸리티"""
    
    @staticmethod
    def measure_time(func):
        """함수 실행 시간 측정 데코레이터"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
            return result
        
        return wrapper
    
    @staticmethod
    def get_memory_usage():
        """메모리 사용량 반환"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': process.memory_percent()
            }
        except ImportError:
            return None
    
    @staticmethod
    def benchmark_function(func, *args, iterations: int = 5, **kwargs):
        """함수 성능 벤치마크"""
        times = []
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            times.append(end_time - start_time)
            results.append(result)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'times': times,
            'results': results
        }

# 검증 유틸리티
class ValidationUtils:
    """데이터 검증 관련 유틸리티"""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """이미지 유효성 검사"""
        if image is None:
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False
        
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            return False
        
        return True
    
    @staticmethod
    def validate_bbox(bbox: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """바운딩 박스 유효성 검사"""
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2]
        
        # 좌표 범위 검사
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        # 크기 검사
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True
    
    @staticmethod
    def validate_detection(detection: Dict) -> bool:
        """감지 결과 유효성 검사"""
        required_keys = ['class', 'confidence', 'bbox']
        
        for key in required_keys:
            if key not in detection:
                return False
        
        # 신뢰도 범위 검사
        if not (0 <= detection['confidence'] <= 1):
            return False
        
        # 클래스 검사
        if detection['class'] not in detection_config.VEHICLE_CLASSES:
            return False
        
        return True
    
    @staticmethod
    def validate_detection_list(detections: List[Dict]) -> List[str]:
        """감지 리스트 전체 검증"""
        errors = []
        
        for i, det in enumerate(detections):
            if not ValidationUtils.validate_detection(det):
                errors.append(f"Detection {i}: Invalid detection data")
                
                # 상세 오류 정보
                if 'confidence' in det and not (0 <= det['confidence'] <= 1):
                    errors.append(f"  - Invalid confidence: {det['confidence']}")
                
                if 'bbox' in det and len(det['bbox']) != 4:
                    errors.append(f"  - Invalid bbox: {det['bbox']}")
                
                if 'class' in det and det['class'] not in detection_config.VEHICLE_CLASSES:
                    errors.append(f"  - Invalid class: {det['class']}")
        
        return errors

# 디버깅 유틸리티
class DebugUtils:
    """디버깅 관련 유틸리티"""
    
    @staticmethod
    def print_detection_debug_info(detections: List[Dict], 
                                  image_shape: Tuple[int, int] = None):
        """감지 결과 디버그 정보 출력"""
        print(f"\n{'='*60}")
        print(f"DEBUG: Detection Information")
        print(f"{'='*60}")
        
        if image_shape:
            print(f"Image Shape: {image_shape[1]} x {image_shape[0]} (W x H)")
        
        print(f"Total Detections: {len(detections)}")
        
        for i, det in enumerate(detections):
            print(f"\nDetection {i+1}:")
            print(f"  Class: {det['class']}")
            print(f"  Confidence: {det['confidence']:.4f}")
            
            bbox = det['bbox']
            print(f"  Bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print(f"  Width: {bbox[2] - bbox[0]}")
            print(f"  Height: {bbox[3] - bbox[1]}")
            print(f"  Area: {BboxUtils.calculate_area(bbox):.0f}")
            
            if 'area' in det:
                print(f"  Stored Area: {det['area']:.0f}")
            if 'center' in det:
                print(f"  Center: {det['center']}")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def create_debug_visualization(image: np.ndarray, 
                                 detections: List[Dict]) -> np.ndarray:
        """디버그용 상세 시각화"""
        debug_image = image.copy()
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # 바운딩 박스
            color = detection_config.CLASS_COLORS.get(class_name, (128, 128, 128))
            cv2.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 상세 정보 텍스트
            texts = [
                f"#{i+1} {class_name}",
                f"Conf: {confidence:.3f}",
                f"Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}",
                f"Area: {BboxUtils.calculate_area(bbox):.0f}"
            ]
            
            # 텍스트 배경
            text_y = bbox[1] - 10
            for j, text in enumerate(texts):
                y_offset = text_y - (j * 15)
                if y_offset > 15:
                    cv2.putText(debug_image, text, (bbox[0], y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 중심점 표시
            center = BboxUtils.get_bbox_center(bbox)
            cv2.circle(debug_image, center, 3, color, -1)
        
        return debug_image
    
    @staticmethod
    def save_debug_info(detections: List[Dict], 
                       image_info: Dict = None,
                       output_path: str = None) -> bool:
        """디버그 정보 파일로 저장"""
        if output_path is None:
            output_path = OUTPUT_DIR / f"debug_info_{int(time.time())}.json"
        
        debug_data = {
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info,
            'detections': detections,
            'validation_errors': ValidationUtils.validate_detection_list(detections),
            'statistics': AnalysisUtils.calculate_detection_statistics(detections)
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Debug info saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")
            return False

# 데이터 변환 유틸리티
class ConversionUtils:
    """데이터 변환 관련 유틸리티"""
    
    @staticmethod
    def convert_color_space(image: np.ndarray, 
                           from_space: str, 
                           to_space: str) -> np.ndarray:
        """색상 공간 변환"""
        conversion_map = {
            ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
            ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
            ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
            ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
            ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
            ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
            ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
            ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
        }
        
        key = (from_space.upper(), to_space.upper())
        if key in conversion_map:
            return cv2.cvtColor(image, conversion_map[key])
        else:
            logger.warning(f"Conversion {from_space} -> {to_space} not supported")
            return image
    
    @staticmethod
    def detections_to_dict_list(detections: List[Dict]) -> List[Dict]:
        """감지 결과를 직렬화 가능한 형태로 변환"""
        serializable_detections = []
        
        for det in detections:
            serializable_det = {}
            for key, value in det.items():
                if isinstance(value, np.ndarray):
                    serializable_det[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_det[key] = value.item()
                else:
                    serializable_det[key] = value
            
            serializable_detections.append(serializable_det)
        
        return serializable_detections
    
    @staticmethod
    def dict_list_to_detections(dict_list: List[Dict]) -> List[Dict]:
        """직렬화된 데이터를 감지 결과로 변환"""
        detections = []
        
        for det_dict in dict_list:
            detection = det_dict.copy()
            
            # bbox를 numpy array로 변환
            if 'bbox' in detection and isinstance(detection['bbox'], list):
                detection['bbox'] = np.array(detection['bbox'])
            
            detections.append(detection)
        
        return detections

# 설정 관리 유틸리티
class ConfigUtils:
    """설정 관리 관련 유틸리티"""
    
    @staticmethod
    def save_config_to_file(config_dict: Dict, file_path: Union[str, Path]):
        """설정을 JSON 파일로 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @staticmethod
    def load_config_from_file(file_path: Union[str, Path]) -> Dict:
        """JSON 파일에서 설정 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Config loaded from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def get_system_info() -> Dict:
        """시스템 정보 수집"""
        import platform
        import sys
        
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__,
            'matplotlib_version': plt.matplotlib.__version__
        }
        
        try:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            info['torch_version'] = 'Not installed'
        
        try:
            import ultralytics
            info['ultralytics_version'] = ultralytics.__version__
        except ImportError:
            info['ultralytics_version'] = 'Not installed'
        
        return info

# 전역 유틸리티 함수들
def print_system_info():
    """시스템 정보 출력"""
    info = ConfigUtils.get_system_info()
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")

def setup_project_directories():
    """프로젝트 디렉토리 설정"""
    directories = [OUTPUT_DIR, MODELS_DIR]
    
    for directory in directories:
        FileUtils.ensure_directory(directory)
        logger.info(f"Directory ensured: {directory}")

def check_dependencies():
    """의존성 패키지 확인"""
    from config import env_config
    missing = env_config.check_dependencies()
    
    if missing:
        logger.warning(f"Missing packages: {missing}")
        print(f"⚠️  Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        logger.info("All dependencies are installed")
        print("✅ All dependencies are installed!")
        return True

def validate_all_configs():
    """모든 설정 검증"""
    try:
        from config import validate_config
        validate_config()
        
        # 추가 검증
        assert len(detection_config.VEHICLE_CLASSES) > 0, "차량 클래스가 정의되지 않았습니다"
        assert len(detection_config.CLASS_NAMES_KR) > 0, "한국어 클래스명이 정의되지 않았습니다"
        
        logger.info("All configurations validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def system_health_check():
    """시스템 전체 상태 확인"""
    print("🏥 System Health Check")
    print("=" * 40)
    
    checks = {
        "Dependencies": check_dependencies(),
        "Configurations": validate_all_configs(),
        "Directories": True,
    }
    
    # GPU 확인
    try:
        import torch
        checks["GPU Available"] = torch.cuda.is_available()
    except ImportError:
        checks["GPU Available"] = "PyTorch not installed"
    
    # 모델 사용 가능성 확인
    try:
        model_status = ModelUtils.check_model_availability()
        yolo_available = any('yolo' in k and v for k, v in model_status.items())
        sam_available = any('sam' in k and v for k, v in model_status.items())
        
        checks["YOLO Models"] = yolo_available
        checks["SAM Models"] = sam_available
    except Exception as e:
        logger.warning(f"Model availability check failed: {e}")
        checks["YOLO Models"] = "Unknown"
        checks["SAM Models"] = "Unknown"
    
    # 결과 출력
    all_good = True
    for check_name, status in checks.items():
        if status is True:
            print(f"✅ {check_name}: OK")
        elif status is False:
            print(f"❌ {check_name}: FAIL")
            all_good = False
        else:
            print(f"⚠️  {check_name}: {status}")
    
    print("=" * 40)
    
    if all_good:
        print("🎉 System is ready for vehicle detection!")
    else:
        print("⚠️  Some components need attention")
    
    return all_good

def create_sample_detection():
    """샘플 감지 결과 생성 (테스트용)"""
    return [
        {
            'class': 'car',
            'confidence': 0.892,
            'bbox': np.array([100, 100, 400, 300]),
            'area': 60000,
            'center': (250, 200)
        }
    ]

# 메인 함수
def main():
    """유틸리티 테스트"""
    print("🔧 Testing utility functions...")
    
    # 시스템 정보 출력
    print_system_info()
    
    # 디렉토리 설정
    setup_project_directories()
    
    # 의존성 확인
    check_dependencies()
    
    # 모델 사용 가능 여부 확인
    model_status = ModelUtils.check_model_availability()
    print("\n📦 Model Availability:")
    for model, available in model_status.items():
        status = "✅" if available else "❌"
        print(f"  {model}: {status}")
    
    # 메모리 사용량
    memory_info = PerformanceUtils.get_memory_usage()
    if memory_info:
        print(f"\n💾 Memory Usage:")
        print(f"  RSS: {memory_info['rss']:.1f} MB")
        print(f"  Percentage: {memory_info['percent']:.1f}%")
    
    # 샘플 감지 결과 테스트
    sample_detections = create_sample_detection()
    print(f"\n🚗 Sample Detection Test:")
    TextUtils.print_detection_summary(sample_detections)
    
    # 한국어 리포트 테스트
    korean_report = KoreanTextUtils.format_korean_report(sample_detections)
    print("🇰🇷 Korean Report:")
    print(korean_report)
    
    # 시스템 상태 종합 점검
    print("\n" + "="*50)
    system_health_check()
    
    print("\n✅ Utility functions test completed!")

if __name__ == "__main__":
    main()