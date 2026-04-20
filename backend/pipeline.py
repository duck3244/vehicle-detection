"""
통합 차량 감지 및 세그멘테이션 파이프라인
YOLO + SAM을 조합한 완전한 차량 분석 시스템
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import time

# 프로젝트 모듈 import
from config import (
    detection_config, model_config, viz_config, 
    OUTPUT_DIR, file_config
)
from utils import (
    ImageUtils, FileUtils, TextUtils, VisualizationUtils,
    PerformanceUtils, ValidationUtils, logger
)
from yolo_detector import YOLOVehicleDetector, YOLO_AVAILABLE
from sam_segmentor import create_sam_segmentor, SAM_AVAILABLE

class VehicleDetectionPipeline:
    """통합 차량 감지 및 세그멘테이션 파이프라인"""
    
    def __init__(self,
                 yolo_model: str = None,
                 sam_model: str = None,
                 confidence_threshold: float = None,
                 device: str = None,
                 enable_sam: bool = True):
        """
        파이프라인 초기화
        
        Args:
            yolo_model: YOLO 모델 이름
            sam_model: SAM 모델 타입
            confidence_threshold: 신뢰도 임계값
            device: 사용할 디바이스
            enable_sam: SAM 세그멘테이션 활성화 여부
        """
        self.yolo_model_name = yolo_model or model_config.DEFAULT_YOLO_MODEL
        self.sam_model_type = sam_model or 'vit_h'
        self.confidence_threshold = confidence_threshold or detection_config.DEFAULT_CONFIDENCE_THRESHOLD
        self.device = device
        self.enable_sam = enable_sam and SAM_AVAILABLE
        
        # 컴포넌트 초기화
        self.yolo_detector = None
        self.sam_segmentor = None
        self._initialize_components()
        
        # 파이프라인 통계
        self.pipeline_stats = {
            'total_processed': 0,
            'total_detections': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info(f"VehicleDetectionPipeline initialized")
        logger.info(f"  YOLO: {self.yolo_model_name}")
        logger.info(f"  SAM: {'Enabled' if self.enable_sam else 'Disabled'}")
        logger.info(f"  Confidence: {self.confidence_threshold}")
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # YOLO 감지기 초기화
            if YOLO_AVAILABLE:
                self.yolo_detector = YOLOVehicleDetector(
                    model_name=self.yolo_model_name,
                    conf_threshold=self.confidence_threshold,
                    device=self.device
                )
                logger.info("YOLO detector initialized successfully")
            else:
                logger.error("YOLO not available - cannot initialize detector")
                raise ImportError("YOLO detector is required")
            
            # SAM 세그멘터 초기화 (선택적)
            if self.enable_sam:
                try:
                    self.sam_segmentor = create_sam_segmentor(
                        model_type=self.sam_model_type,
                        device=self.device
                    )
                    logger.info("SAM segmentor initialized successfully")
                except Exception as e:
                    logger.warning(f"SAM initialization failed: {e}")
                    logger.info("Continuing with YOLO-only mode")
                    self.enable_sam = False
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    @PerformanceUtils.measure_time
    def process_image(self,
                     image_path: Union[str, Path],
                     save_results: bool = True,
                     show_results: bool = False,
                     refine_masks: bool = True) -> Dict:
        """
        단일 이미지 처리
        
        Args:
            image_path: 입력 이미지 경로
            save_results: 결과 저장 여부
            show_results: 결과 표시 여부
            refine_masks: 마스크 정제 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        
        result = {
            'image_path': str(image_path),
            'success': False,
            'detections': [],
            'masks': [],
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            logger.info(f"Processing image: {image_path}")
            
            # 이미지 존재 확인
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # 이미지 로드
            image = ImageUtils.load_image(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            result['image_shape'] = image.shape
            
            # 1단계: YOLO 차량 감지
            logger.debug("Step 1: Running YOLO detection")
            detections = self.yolo_detector.detect_vehicles(image_path)
            result['detections'] = detections
            
            if not detections:
                logger.info("No vehicles detected")
                result['success'] = True
                return result
            
            logger.info(f"Detected {len(detections)} vehicle(s)")
            
            # 2단계: SAM 세그멘테이션 (선택적)
            masks = []
            if self.enable_sam and self.sam_segmentor:
                logger.debug("Step 2: Running SAM segmentation")
                
                # SAM에 이미지 설정
                if self.sam_segmentor.set_image(image):
                    # 바운딩 박스에서 세그멘테이션
                    boxes = [det['bbox'] for det in detections]
                    masks = self.sam_segmentor.segment_from_boxes(boxes)
                    
                    # 마스크 정제 (선택적)
                    if refine_masks and hasattr(self.sam_segmentor, 'refine_masks'):
                        masks = self.sam_segmentor.refine_masks(masks)
                    
                    logger.info(f"Generated {len(masks)} segmentation mask(s)")
                else:
                    logger.warning("Failed to set image for SAM")
            
            result['masks'] = masks
            
            # 3단계: 결과 시각화
            logger.debug("Step 3: Creating visualizations")
            visualizations = self._create_visualizations(image, detections, masks)
            result['visualizations'] = visualizations
            
            # 4단계: 결과 저장 (선택적)
            if save_results:
                logger.debug("Step 4: Saving results")
                save_success = self._save_results(image_path, detections, masks, visualizations)
                result['saved'] = save_success
            
            # 5단계: 결과 표시 (선택적)
            if show_results:
                self._show_results(visualizations, detections)
            
            # 처리 성공
            result['success'] = True
            result['processing_time'] = time.time() - start_time
            
            # 통계 업데이트
            self._update_pipeline_stats(result)
            
            logger.info(f"Image processed successfully in {result['processing_time']:.2f}s")
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            logger.error(f"Error processing image: {e}")
        
        return result
    
    def process_batch(self,
                     input_path: Union[str, Path],
                     pattern: str = "*",
                     max_images: int = None) -> Dict[str, Dict]:
        """
        배치 이미지 처리
        
        Args:
            input_path: 입력 디렉토리 또는 이미지 파일
            pattern: 파일 패턴 (디렉토리인 경우)
            max_images: 최대 처리 이미지 수
            
        Returns:
            모든 이미지의 처리 결과
        """
        logger.info(f"Starting batch processing: {input_path}")
        
        # 처리할 이미지 파일 목록 생성
        input_path = Path(input_path)
        
        if input_path.is_file():
            image_files = [input_path]
        elif input_path.is_dir():
            image_files = FileUtils.get_image_files(input_path)
            if pattern != "*":
                image_files = [f for f in image_files if f.match(pattern)]
        else:
            logger.error(f"Invalid input path: {input_path}")
            return {}
        
        # 최대 이미지 수 제한
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
            logger.info(f"Limited to {max_images} images")
        
        if not image_files:
            logger.warning("No image files found")
            return {}
        
        logger.info(f"Processing {len(image_files)} image(s)")
        
        # 각 이미지 처리
        results = {}
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            result = self.process_image(
                image_file,
                save_results=True,
                show_results=False
            )
            
            results[str(image_file)] = result
        
        # 배치 처리 요약
        self._print_batch_summary(results)
        
        return results
    
    def _create_visualizations(self,
                             image: np.ndarray,
                             detections: List[Dict],
                             masks: List[np.ndarray] = None) -> Dict:
        """시각화 생성"""
        visualizations = {}
        
        try:
            # 1. YOLO 감지 결과 시각화
            yolo_viz = self.yolo_detector.visualize_detections(
                image, detections, show_labels=True, show_confidence=True
            )
            visualizations['yolo_detection'] = yolo_viz
            
            # 2. SAM 세그멘테이션 시각화 (있는 경우)
            if masks and self.sam_segmentor:
                # 원본 이미지에 마스크 오버레이
                sam_viz = self.sam_segmentor.visualize_masks(masks)
                visualizations['sam_segmentation'] = sam_viz
                
                # 마스크만 표시
                mask_viz = self._create_mask_visualization(masks)
                visualizations['masks_only'] = mask_viz
                
                # 결합된 시각화 (감지 + 세그멘테이션)
                combined_viz = self._create_combined_visualization(
                    image, detections, masks
                )
                visualizations['combined'] = combined_viz
            
            # 3. 통계 시각화
            if detections:
                stats_viz = VisualizationUtils.plot_detection_stats(detections)
                visualizations['statistics'] = stats_viz
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizations
    
    def _create_mask_visualization(self, masks: List[np.ndarray]) -> np.ndarray:
        """마스크만 표시하는 시각화 생성"""
        if not masks:
            return None
        
        # 모든 마스크를 하나로 결합
        h, w = masks[0].shape
        combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 각 마스크를 다른 색상으로 표시
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            color = (np.array(colors[i][:3]) * 255).astype(np.uint8)
            combined_mask[mask > 0] = color
        
        return combined_mask
    
    def _create_combined_visualization(self,
                                     image: np.ndarray,
                                     detections: List[Dict],
                                     masks: List[np.ndarray]) -> np.ndarray:
        """감지 + 세그멘테이션 결합 시각화"""
        # 먼저 SAM 마스크 오버레이
        if self.sam_segmentor and hasattr(self.sam_segmentor, 'current_image'):
            self.sam_segmentor.current_image = image
        
        combined = image.copy()
        
        # 마스크 오버레이
        if masks and self.sam_segmentor:
            combined = self.sam_segmentor.visualize_masks(masks)
        
        # 바운딩 박스 추가
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            color = detection_config.CLASS_COLORS.get(class_name, (255, 255, 255))
            kr_name = detection_config.CLASS_NAMES_KR.get(class_name, class_name)
            label = f"{kr_name}: {confidence:.2f}"
            
            combined = VisualizationUtils.draw_bbox(combined, bbox, label, color)
        
        return combined
    
    def _save_results(self,
                     image_path: Union[str, Path],
                     detections: List[Dict],
                     masks: List[np.ndarray],
                     visualizations: Dict) -> bool:
        """결과 저장"""
        try:
            image_path = Path(image_path)
            base_name = image_path.stem
            
            # 1. 시각화 이미지들 저장
            for viz_name, viz_image in visualizations.items():
                if viz_image is not None and isinstance(viz_image, np.ndarray):
                    output_path = OUTPUT_DIR / f"{base_name}_{viz_name}.jpg"
                    ImageUtils.save_image(viz_image, output_path)
                elif hasattr(viz_image, 'savefig'):  # matplotlib figure
                    output_path = OUTPUT_DIR / f"{base_name}_{viz_name}.png"
                    viz_image.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # 2. 텍스트 결과 저장
            if detections:
                text_output_path = OUTPUT_DIR / f"{base_name}_results.txt"
                image_info = {'original_path': str(image_path)}
                TextUtils.save_detection_results(
                    detections, text_output_path, image_info
                )
            
            # 3. 마스크 데이터 저장 (NumPy 형식)
            if masks:
                masks_output_path = OUTPUT_DIR / f"{base_name}_masks.npz"
                np.savez_compressed(masks_output_path, *masks)
            
            logger.info(f"Results saved for: {image_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def _show_results(self, visualizations: Dict, detections: List[Dict]):
        """결과 표시"""
        try:
            # 감지 결과 요약 출력
            TextUtils.print_detection_summary(detections)
            
            # matplotlib 백엔드 설정 확인
            import matplotlib
            current_backend = matplotlib.get_backend()
            logger.debug(f"Current matplotlib backend: {current_backend}")
            
            # GUI 백엔드가 없는 경우 처리
            if current_backend == 'Agg':
                logger.warning("GUI backend not available. Saving plots instead of showing.")
                self._save_plots_instead_of_show(visualizations)
                return
            
            # 시각화 표시
            viz_images = {k: v for k, v in visualizations.items() 
                         if isinstance(v, np.ndarray)}
            
            if len(viz_images) == 1:
                # 단일 이미지 표시
                try:
                    fig = plt.figure(figsize=viz_config.SINGLE_FIGURE_SIZE)
                    plt.imshow(list(viz_images.values())[0])
                    plt.title(list(viz_images.keys())[0])
                    plt.axis('off')
                    
                    # 안전한 표시 방법
                    if hasattr(plt, 'show') and current_backend.lower() != 'agg':
                        plt.show(block=False)
                        plt.pause(0.1)  # GUI 업데이트 시간 제공
                    else:
                        # 파일로 저장
                        output_path = OUTPUT_DIR / "visualization_result.png"
                        fig.savefig(output_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Visualization saved to: {output_path}")
                    
                    plt.close(fig)  # 메모리 해제
                    
                except Exception as e:
                    logger.warning(f"Failed to show plot, saving instead: {e}")
                    output_path = OUTPUT_DIR / "visualization_result.png"
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
            elif len(viz_images) > 1:
                # 다중 이미지 표시
                try:
                    n_images = len(viz_images)
                    cols = min(3, n_images)
                    rows = (n_images + cols - 1) // cols
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
                    if n_images == 1:
                        axes = [axes]
                    elif rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    axes = axes.flatten()
                    
                    for i, (name, image) in enumerate(viz_images.items()):
                        axes[i].imshow(image)
                        axes[i].set_title(name.replace('_', ' ').title())
                        axes[i].axis('off')
                    
                    # 빈 subplot 숨기기
                    for i in range(n_images, len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    
                    # 안전한 표시 방법
                    if hasattr(plt, 'show') and current_backend.lower() != 'agg':
                        plt.show(block=False)
                        plt.pause(0.1)
                    else:
                        output_path = OUTPUT_DIR / "visualization_grid.png"
                        fig.savefig(output_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Grid visualization saved to: {output_path}")
                    
                    plt.close(fig)
                    
                except Exception as e:
                    logger.warning(f"Failed to show grid plot, saving instead: {e}")
                    output_path = OUTPUT_DIR / "visualization_grid.png"
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # matplotlib figures 처리
            for name, fig in visualizations.items():
                if hasattr(fig, 'savefig'):
                    try:
                        if hasattr(fig, 'show') and current_backend.lower() != 'agg':
                            fig.show()
                        else:
                            output_path = OUTPUT_DIR / f"{name}_plot.png"
                            fig.savefig(output_path, dpi=300, bbox_inches='tight')
                            logger.info(f"Plot saved to: {output_path}")
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error handling figure {name}: {e}")
            
        except Exception as e:
            logger.error(f"Error showing results: {e}")
    
    def _save_plots_instead_of_show(self, visualizations: Dict):
        """GUI가 없는 경우 플롯을 파일로 저장"""
        logger.info("Saving visualizations to files instead of showing")
        
        for name, viz in visualizations.items():
            try:
                output_path = OUTPUT_DIR / f"{name}_saved.png"
                
                if isinstance(viz, np.ndarray):
                    # numpy 배열인 경우
                    fig = plt.figure(figsize=viz_config.SINGLE_FIGURE_SIZE)
                    plt.imshow(viz)
                    plt.title(name.replace('_', ' ').title())
                    plt.axis('off')
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                elif hasattr(viz, 'savefig'):
                    # matplotlib figure인 경우
                    viz.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(viz)
                
                logger.info(f"Saved visualization: {output_path}")
                
            except Exception as e:
                logger.error(f"Error saving visualization {name}: {e}")
    
    def _update_pipeline_stats(self, result: Dict):
        """파이프라인 통계 업데이트"""
        self.pipeline_stats['total_processed'] += 1
        
        if result['success']:
            self.pipeline_stats['total_detections'] += len(result['detections'])
        
        self.pipeline_stats['total_time'] += result['processing_time']
        self.pipeline_stats['avg_processing_time'] = (
            self.pipeline_stats['total_time'] / self.pipeline_stats['total_processed']
        )
    
    def _print_batch_summary(self, results: Dict[str, Dict]):
        """배치 처리 요약 출력"""
        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful
        total_detections = sum(len(r['detections']) for r in results.values() if r['success'])
        avg_time = np.mean([r['processing_time'] for r in results.values()])
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total detections: {total_detections}")
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"{'='*60}\n")
        
        # 실패한 이미지들 출력
        if failed > 0:
            print("Failed images:")
            for path, result in results.items():
                if not result['success']:
                    print(f"  {Path(path).name}: {result.get('error', 'Unknown error')}")
            print()
    
    def get_pipeline_statistics(self) -> Dict:
        """파이프라인 통계 반환"""
        stats = self.pipeline_stats.copy()
        
        # 추가 통계 계산
        if stats['total_processed'] > 0:
            stats['detection_rate'] = stats['total_detections'] / stats['total_processed']
            stats['success_rate'] = 1.0  # 실패 추적을 위해 나중에 개선 가능
        
        return stats
    
    def reset_statistics(self):
        """통계 초기화"""
        self.pipeline_stats = {
            'total_processed': 0,
            'total_detections': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # 하위 컴포넌트 통계도 초기화
        if self.yolo_detector:
            self.yolo_detector.reset_statistics()
    
    def update_settings(self, **kwargs):
        """파이프라인 설정 업데이트"""
        if 'confidence_threshold' in kwargs:
            new_threshold = kwargs['confidence_threshold']
            if 0 <= new_threshold <= 1:
                self.confidence_threshold = new_threshold
                if self.yolo_detector:
                    self.yolo_detector.update_confidence_threshold(new_threshold)
                logger.info(f"Confidence threshold updated: {new_threshold}")
        
        if 'enable_sam' in kwargs:
            self.enable_sam = kwargs['enable_sam'] and SAM_AVAILABLE
            logger.info(f"SAM segmentation: {'Enabled' if self.enable_sam else 'Disabled'}")

# 편의 함수들
def quick_vehicle_detection(image_path: Union[str, Path],
                          yolo_model: str = 'yolov8n.pt',
                          confidence: float = 0.25,
                          save_result: bool = True,
                          show_result: bool = True) -> Dict:
    """
    빠른 차량 감지 (SAM 없이)
    
    Args:
        image_path: 이미지 경로
        yolo_model: YOLO 모델
        confidence: 신뢰도 임계값
        save_result: 결과 저장 여부
        show_result: 결과 표시 여부
        
    Returns:
        처리 결과
    """
    pipeline = VehicleDetectionPipeline(
        yolo_model=yolo_model,
        confidence_threshold=confidence,
        enable_sam=False
    )
    
    return pipeline.process_image(
        image_path,
        save_results=save_result,
        show_results=show_result
    )

def full_vehicle_analysis(image_path: Union[str, Path],
                         yolo_model: str = 'yolov8n.pt',
                         sam_model: str = 'vit_h',
                         confidence: float = 0.25,
                         save_result: bool = True,
                         show_result: bool = True) -> Dict:
    """
    완전한 차량 분석 (YOLO + SAM)
    
    Args:
        image_path: 이미지 경로
        yolo_model: YOLO 모델
        sam_model: SAM 모델
        confidence: 신뢰도 임계값
        save_result: 결과 저장 여부
        show_result: 결과 표시 여부
        
    Returns:
        처리 결과
    """
    pipeline = VehicleDetectionPipeline(
        yolo_model=yolo_model,
        sam_model=sam_model,
        confidence_threshold=confidence,
        enable_sam=True
    )
    
    return pipeline.process_image(
        image_path,
        save_results=save_result,
        show_results=show_result
    )

# 메인 함수
def main():
    """파이프라인 테스트"""
    print("🚀 Vehicle Detection Pipeline Test")
    print("=" * 50)
    
    # 테스트용 이미지
    test_image = "korean_car.jpg"
    
    try:
        if not Path(test_image).exists():
            print(f"⚠️  Test image '{test_image}' not found!")
            print("Please place a vehicle image in the current directory.")
            print("\nTesting with pipeline initialization only...")
            
            # 파이프라인만 초기화해서 테스트
            pipeline = VehicleDetectionPipeline(enable_sam=False)
            stats = pipeline.get_pipeline_statistics()
            print(f"Pipeline initialized successfully")
            print(f"Statistics: {stats}")
            
            return
        
        print(f"Testing with image: {test_image}")
        
        # 1. 빠른 감지 테스트 (YOLO만)
        print("\n1. Quick Detection Test (YOLO only)...")
        result1 = quick_vehicle_detection(
            test_image,
            save_result=True,
            show_result=False
        )
        
        if result1['success']:
            print(f"✅ Quick detection: {len(result1['detections'])} vehicles found")
            print(f"⏱️  Processing time: {result1['processing_time']:.2f}s")
        else:
            print(f"❌ Quick detection failed: {result1.get('error', 'Unknown error')}")
        
        # 2. 완전한 분석 테스트 (YOLO + SAM)
        if SAM_AVAILABLE:
            print("\n2. Full Analysis Test (YOLO + SAM)...")
            result2 = full_vehicle_analysis(
                test_image,
                save_result=True,
                show_result=False
            )
            
            if result2['success']:
                print(f"✅ Full analysis: {len(result2['detections'])} vehicles, {len(result2['masks'])} masks")
                print(f"⏱️  Processing time: {result2['processing_time']:.2f}s")
            else:
                print(f"❌ Full analysis failed: {result2.get('error', 'Unknown error')}")
        else:
            print("\n2. SAM not available - skipping full analysis test")
        
        print(f"\n📁 Results saved in: {OUTPUT_DIR}")
        print("🎉 Pipeline test completed!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        logger.error(f"Pipeline test error: {e}")

if __name__ == "__main__":
    main()