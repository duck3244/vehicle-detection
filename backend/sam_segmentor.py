"""
SAM (Segment Anything Model) 기반 세그멘테이션 모듈
Meta의 SAM 모델을 사용하여 정밀한 객체 세그멘테이션 수행
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  Segment Anything not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")

from config import (
    model_config, detection_config, viz_config, 
    env_config, MODELS_DIR
)
from utils import (
    ImageUtils, ValidationUtils, PerformanceUtils, 
    ModelUtils, logger
)

class SAMSegmentor:
    """SAM 기반 세그멘테이션 클래스"""
    
    def __init__(self, 
                 model_type: str = 'vit_h',
                 model_path: str = None,
                 device: str = None):
        """
        초기화
        
        Args:
            model_type: SAM 모델 타입 ('vit_h', 'vit_l', 'vit_b')
            model_path: 모델 체크포인트 경로
            device: 사용할 디바이스
        """
        if not SAM_AVAILABLE:
            raise ImportError("Segment Anything package is required. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        
        self.model_type = model_type
        self.device = self._setup_device(device)
        
        # 모델 경로 설정
        if model_path is None:
            model_path = MODELS_DIR / model_config.SAM_MODELS[model_type]
        
        self.model_path = Path(model_path)
        
        # 모델 로드
        self.sam_model = None
        self.predictor = None
        self._load_model()
        
        # 현재 설정된 이미지
        self.current_image = None
        self.image_embedding = None
        
        logger.info(f"SAMSegmentor initialized with {model_type} on {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """디바이스 설정"""
        if device is None:
            return env_config.get_device()
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_model(self):
        """SAM 모델 로드"""
        try:
            # 모델 파일 존재 확인
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Attempting to download SAM model...")
                
                success = ModelUtils.download_sam_model(self.model_type)
                if not success:
                    raise FileNotFoundError(f"Failed to download SAM model: {self.model_type}")
            
            logger.info(f"Loading SAM model from: {self.model_path}")
            
            # 모델 로드
            self.sam_model = sam_model_registry[self.model_type](
                checkpoint=str(self.model_path)
            ).to(self.device)
            
            # 예측기 초기화
            self.predictor = SamPredictor(self.sam_model)
            
            logger.info(f"SAM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise
    
    def set_image(self, image: Union[str, Path, np.ndarray]) -> bool:
        """
        세그멘테이션할 이미지 설정
        
        Args:
            image: 입력 이미지 (경로 또는 numpy array)
            
        Returns:
            성공 여부
        """
        try:
            # 이미지 로드
            if isinstance(image, (str, Path)):
                image_array = ImageUtils.load_image(image)
                if image_array is None:
                    return False
            else:
                image_array = image
            
            # 이미지 유효성 검사
            if not ValidationUtils.validate_image(image_array):
                logger.error("Invalid image provided to SAM")
                return False
            
            # RGB로 변환 (SAM은 RGB 입력 필요)
            if len(image_array.shape) == 3:
                # 이미 RGB인지 확인 (일반적으로 ImageUtils.load_image는 RGB 반환)
                self.current_image = image_array
            else:
                # Grayscale인 경우 RGB로 변환
                self.current_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # SAM에 이미지 설정 (임베딩 계산)
            logger.debug("Computing image embedding...")
            self.predictor.set_image(self.current_image)
            
            logger.info(f"Image set for SAM: {self.current_image.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting image for SAM: {e}")
            return False
    
    @PerformanceUtils.measure_time
    def segment_from_boxes(self, 
                          boxes: List[np.ndarray],
                          multimask_output: bool = False) -> List[np.ndarray]:
        """
        바운딩 박스를 기반으로 세그멘테이션 수행
        
        Args:
            boxes: 바운딩 박스 리스트 (XYXY 형식)
            multimask_output: 다중 마스크 출력 여부
            
        Returns:
            생성된 마스크 리스트
        """
        if self.current_image is None or self.predictor is None:
            logger.error("Image not set or predictor not initialized")
            return []
        
        masks = []
        
        try:
            for i, box in enumerate(boxes):
                logger.debug(f"Processing box {i+1}/{len(boxes)}: {box}")
                
                # 박스 유효성 검사
                if not ValidationUtils.validate_bbox(box, self.current_image.shape[:2]):
                    logger.warning(f"Invalid bbox skipped: {box}")
                    masks.append(np.zeros(self.current_image.shape[:2], dtype=np.uint8))
                    continue
                
                # SAM 입력 형식으로 변환
                input_box = np.array(box).reshape(1, -1)
                
                # 마스크 예측
                mask, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=multimask_output
                )
                
                # 가장 좋은 마스크 선택 (점수 기준)
                if multimask_output:
                    best_mask_idx = np.argmax(scores)
                    selected_mask = mask[best_mask_idx]
                else:
                    selected_mask = mask[0]
                
                masks.append(selected_mask.astype(np.uint8))
                
                logger.debug(f"Generated mask {i+1} with score: {scores[0] if not multimask_output else scores[best_mask_idx]:.3f}")
            
            logger.info(f"Generated {len(masks)} masks from {len(boxes)} boxes")
            return masks
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return []
    
    def segment_from_points(self,
                           points: List[Tuple[int, int]],
                           labels: List[int],
                           multimask_output: bool = True) -> List[np.ndarray]:
        """
        포인트를 기반으로 세그멘테이션 수행
        
        Args:
            points: 클릭 포인트 리스트 [(x, y), ...]
            labels: 포인트 레이블 (1: foreground, 0: background)
            multimask_output: 다중 마스크 출력 여부
            
        Returns:
            생성된 마스크 리스트
        """
        if self.current_image is None or self.predictor is None:
            logger.error("Image not set or predictor not initialized")
            return []
        
        try:
            # 입력 데이터 준비
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            # 마스크 예측
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=None,
                multimask_output=multimask_output
            )
            
            # 마스크를 uint8로 변환
            masks = [mask.astype(np.uint8) for mask in masks]
            
            logger.info(f"Generated {len(masks)} masks from {len(points)} points")
            return masks
            
        except Exception as e:
            logger.error(f"Error during point-based segmentation: {e}")
            return []
    
    def refine_masks(self, 
                     masks: List[np.ndarray],
                     method: str = 'morphology') -> List[np.ndarray]:
        """
        마스크 후처리 및 정제
        
        Args:
            masks: 원본 마스크 리스트
            method: 정제 방법 ('morphology', 'contour', 'smooth')
            
        Returns:
            정제된 마스크 리스트
        """
        refined_masks = []
        
        for i, mask in enumerate(masks):
            try:
                if method == 'morphology':
                    # 모폴로지 연산으로 노이즈 제거
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
                
                elif method == 'contour':
                    # 컨투어 기반 정제
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    refined_mask = np.zeros_like(mask)
                    
                    # 가장 큰 컨투어만 유지
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.fillPoly(refined_mask, [largest_contour], 1)
                
                elif method == 'smooth':
                    # 가우시안 블러 후 이진화로 스무딩
                    blurred = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.5)
                    refined_mask = (blurred > 0.5).astype(np.uint8)
                
                else:
                    refined_mask = mask
                
                refined_masks.append(refined_mask)
                
            except Exception as e:
                logger.warning(f"Error refining mask {i}: {e}")
                refined_masks.append(mask)
        
        return refined_masks
    
    def combine_masks(self, 
                     masks: List[np.ndarray],
                     method: str = 'union') -> np.ndarray:
        """
        여러 마스크를 결합
        
        Args:
            masks: 마스크 리스트
            method: 결합 방법 ('union', 'intersection', 'weighted')
            
        Returns:
            결합된 마스크
        """
        if not masks:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        try:
            if method == 'union':
                combined = masks[0].copy()
                for mask in masks[1:]:
                    combined = np.logical_or(combined, mask).astype(np.uint8)
            
            elif method == 'intersection':
                combined = masks[0].copy()
                for mask in masks[1:]:
                    combined = np.logical_and(combined, mask).astype(np.uint8)
            
            elif method == 'weighted':
                # 가중 평균 (모든 마스크를 동일 가중치로)
                combined = np.zeros_like(masks[0], dtype=np.float32)
                for mask in masks:
                    combined += mask.astype(np.float32)
                combined = (combined / len(masks) > 0.5).astype(np.uint8)
            
            else:
                combined = masks[0]
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining masks: {e}")
            return masks[0]
    
    def visualize_masks(self, 
                       masks: List[np.ndarray],
                       alpha: float = None,
                       colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        마스크를 원본 이미지에 오버레이하여 시각화
        
        Args:
            masks: 마스크 리스트
            alpha: 투명도
            colors: 각 마스크의 색상 리스트
            
        Returns:
            시각화된 이미지
        """
        if self.current_image is None:
            logger.error("No image set for visualization")
            return None
        
        if not masks:
            return self.current_image.copy()
        
        alpha = alpha or viz_config.MASK_ALPHA
        visualized = self.current_image.copy()
        
        try:
            for i, mask in enumerate(masks):
                if colors and i < len(colors):
                    color = colors[i]
                else:
                    # 기본 색상 생성 (HSV 색상환에서 균등하게 선택)
                    hue = int(180 * i / len(masks))
                    hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
                    color = tuple(map(int, rgb_color))
                
                # 마스크 영역에 색상 적용
                colored_mask = np.zeros_like(self.current_image)
                colored_mask[mask > 0] = color
                
                # 오버레이
                visualized = cv2.addWeighted(visualized, 1 - alpha, colored_mask, alpha, 0)
            
            return visualized
            
        except Exception as e:
            logger.error(f"Error visualizing masks: {e}")
            return self.current_image.copy()
    
    def get_mask_statistics(self, masks: List[np.ndarray]) -> Dict:
        """
        마스크 통계 정보 계산
        
        Args:
            masks: 마스크 리스트
            
        Returns:
            통계 정보
        """
        if not masks:
            return {}
        
        stats = {
            'total_masks': len(masks),
            'areas': [],
            'centers': [],
            'bounding_boxes': []
        }
        
        for mask in masks:
            # 면적 계산
            area = np.sum(mask > 0)
            stats['areas'].append(area)
            
            # 중심점 계산
            if area > 0:
                y_coords, x_coords = np.where(mask > 0)
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                stats['centers'].append((center_x, center_y))
                
                # 바운딩 박스 계산
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                stats['bounding_boxes'].append([x_min, y_min, x_max, y_max])
            else:
                stats['centers'].append((0, 0))
                stats['bounding_boxes'].append([0, 0, 0, 0])
        
        # 평균 통계
        stats['avg_area'] = np.mean(stats['areas']) if stats['areas'] else 0
        stats['total_area'] = sum(stats['areas'])
        
        if self.current_image is not None:
            image_area = self.current_image.shape[0] * self.current_image.shape[1]
            stats['coverage_ratio'] = stats['total_area'] / image_area
        
        return stats

# 더미 SAM 클래스 (SAM이 없을 때 사용)
class DummySAMSegmentor:
    """SAM이 없을 때 사용하는 더미 클래스"""
    
    def __init__(self, *args, **kwargs):
        self.current_image = None
        logger.warning("Using dummy SAM segmentor (SAM not available)")
    
    def set_image(self, image):
        if isinstance(image, (str, Path)):
            self.current_image = ImageUtils.load_image(image)
        else:
            self.current_image = image
        return self.current_image is not None
    
    def segment_from_boxes(self, boxes, **kwargs):
        """더미 마스크 생성 (타원형)"""
        if self.current_image is None:
            return []
        
        masks = []
        h, w = self.current_image.shape[:2]
        
        for box in boxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = box.astype(int)
            
            # 타원형 마스크 생성
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = (x2 - x1) // 2, (y2 - y1) // 2
            
            if width > 0 and height > 0:
                cv2.ellipse(mask, (center_x, center_y), 
                           (width, height), 0, 0, 360, 1, -1)
            
            masks.append(mask)
        
        return masks
    
    def segment_from_points(self, points, labels, **kwargs):
        return []
    
    def visualize_masks(self, masks, **kwargs):
        if self.current_image is None or not masks:
            return self.current_image
        
        visualized = self.current_image.copy()
        
        for i, mask in enumerate(masks):
            # 간단한 컬러 오버레이
            color = plt.cm.Set3(i / len(masks))[:3]
            color = tuple(int(c * 255) for c in color)
            
            colored_mask = np.zeros_like(self.current_image)
            colored_mask[mask > 0] = color
            
            visualized = cv2.addWeighted(visualized, 0.7, colored_mask, 0.3, 0)
        
        return visualized
    
    def refine_masks(self, masks, method='morphology'):
        return masks
    
    def combine_masks(self, masks, method='union'):
        if not masks:
            return None
        if len(masks) == 1:
            return masks[0]
        
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.logical_or(combined, mask).astype(np.uint8)
        return combined
    
    def get_mask_statistics(self, masks):
        return {'total_masks': len(masks), 'note': 'Dummy statistics'}

# 팩토리 함수
def create_sam_segmentor(*args, **kwargs):
    """SAM 세그멘터 생성 팩토리 함수"""
    if SAM_AVAILABLE:
        return SAMSegmentor(*args, **kwargs)
    else:
        return DummySAMSegmentor(*args, **kwargs)

# 편의 함수
def segment_objects_from_detections(image: Union[str, Path, np.ndarray],
                                  detections: List[Dict],
                                  model_type: str = 'vit_h',
                                  refine: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    감지 결과를 기반으로 객체 세그멘테이션 수행 (편의 함수)
    
    Args:
        image: 입력 이미지
        detections: YOLO 감지 결과
        model_type: SAM 모델 타입
        refine: 마스크 정제 여부
        
    Returns:
        (마스크 리스트, 시각화된 이미지)
    """
    segmentor = create_sam_segmentor(model_type=model_type)
    
    # 이미지 설정
    if not segmentor.set_image(image):
        return [], None
    
    # 바운딩 박스 추출
    boxes = [det['bbox'] for det in detections]
    
    # 세그멘테이션 수행
    masks = segmentor.segment_from_boxes(boxes)
    
    # 마스크 정제 (옵션)
    if refine and hasattr(segmentor, 'refine_masks'):
        masks = segmentor.refine_masks(masks)
    
    # 시각화
    visualized = segmentor.visualize_masks(masks)
    
    return masks, visualized

# 메인 함수
def main():
    """SAM 세그멘터 테스트"""
    print("🎯 SAM Segmentor Test")
    print("=" * 50)
    
    # 테스트용 이미지와 가짜 감지 결과
    test_image = "korean_car.jpg"
    
    # 더미 감지 결과 (실제로는 YOLO에서 받아옴)
    dummy_detections = [
        {
            'class': 'car',
            'confidence': 0.9,
            'bbox': np.array([100, 100, 400, 300])  # 예시 바운딩 박스
        }
    ]
    
    try:
        # SAM 세그멘터 초기화
        segmentor = create_sam_segmentor()
        
        if Path(test_image).exists():
            # 이미지 설정
            if segmentor.set_image(test_image):
                print(f"✅ Image '{test_image}' loaded successfully")
                
                # 바운딩 박스에서 세그멘테이션
                boxes = [det['bbox'] for det in dummy_detections]
                masks = segmentor.segment_from_boxes(boxes)
                
                if masks:
                    print(f"✅ Generated {len(masks)} mask(s)")
                    
                    # 마스크 통계
                    if hasattr(segmentor, 'get_mask_statistics'):
                        stats = segmentor.get_mask_statistics(masks)
                        print("\nMask Statistics:")
                        for key, value in stats.items():
                            print(f"  {key}: {value}")
                    
                    # 시각화
                    visualized = segmentor.visualize_masks(masks)
                    
                    # 결과 저장
                    if visualized is not None:
                        from config import OUTPUT_DIR
                        output_path = OUTPUT_DIR / "sam_segmentation_result.jpg"
                        success = ImageUtils.save_image(visualized, output_path)
                        if success:
                            print(f"📁 Result saved to: {output_path}")
                
                else:
                    print("❌ No masks generated")
            else:
                print("❌ Failed to load image")
        else:
            print(f"⚠️  Test image '{test_image}' not found!")
            print("Testing with dummy segmentor...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in SAM test: {e}")

if __name__ == "__main__":
    main()