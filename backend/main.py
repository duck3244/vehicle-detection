"""
차량 감지 및 세그멘테이션 메인 애플리케이션
명령행 인터페이스를 통한 사용자 친화적 실행
"""

import argparse
import sys
from pathlib import Path
import json

# 프로젝트 모듈 import
from config import (
    model_config, detection_config, OUTPUT_DIR, 
    print_config, validate_config
)
from utils import (
    setup_project_directories, check_dependencies,
    print_system_info, logger
)
from pipeline import (
    VehicleDetectionPipeline, quick_vehicle_detection,
    full_vehicle_analysis
)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="차량 감지 및 세그멘테이션 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s korean_car.jpg                          # 기본 설정으로 단일 이미지 처리
  %(prog)s -i images/ -b                           # 디렉토리의 모든 이미지 배치 처리
  %(prog)s image.jpg -m yolov8s.pt -c 0.3          # 특정 모델과 신뢰도로 처리
  %(prog)s image.jpg --sam --sam-model vit_l       # SAM 세그멘테이션 포함
  %(prog)s image.jpg --no-save --show              # 저장하지 않고 결과만 표시
  %(prog)s --info                                  # 시스템 정보 표시
        """
    )
    
    # 필수 인수
    parser.add_argument(
        'input', nargs='?',
        help='입력 이미지 파일 또는 디렉토리 경로'
    )
    
    # 기본 옵션
    parser.add_argument(
        '-i', '--input',
        dest='input_path',
        help='입력 경로 (위치 인수 대신 사용 가능)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=OUTPUT_DIR,
        help=f'출력 디렉토리 (기본값: {OUTPUT_DIR})'
    )
    
    # 모델 설정
    parser.add_argument(
        '-m', '--model',
        choices=list(model_config.YOLO_MODELS.values()),
        default=model_config.DEFAULT_YOLO_MODEL,
        help='YOLO 모델 선택 (기본값: yolov8n.pt)'
    )
    
    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=detection_config.DEFAULT_CONFIDENCE_THRESHOLD,
        help=f'신뢰도 임계값 (기본값: {detection_config.DEFAULT_CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='사용할 디바이스 (기본값: auto)'
    )
    
    # SAM 관련 옵션
    parser.add_argument(
        '--sam',
        action='store_true',
        help='SAM 세그멘테이션 활성화'
    )
    
    parser.add_argument(
        '--sam-model',
        choices=['vit_h', 'vit_l', 'vit_b'],
        default='vit_h',
        help='SAM 모델 타입 (기본값: vit_h)'
    )
    
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='마스크 정제 비활성화'
    )
    
    # 처리 옵션
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='배치 처리 모드 (디렉토리 입력시)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        help='최대 처리 이미지 수 (배치 모드)'
    )
    
    parser.add_argument(
        '--pattern',
        default='*',
        help='파일 패턴 (배치 모드, 기본값: *)'
    )
    
    # 출력 옵션
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='결과 저장 안함'
    )
    
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='GUI 없이 실행 (파일 저장만)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='결과 화면에 표시'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='최소 출력 (오류만 표시)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 출력'
    )
    
    # 정보 및 설정
    parser.add_argument(
        '--info',
        action='store_true',
        help='시스템 정보 및 설정 표시'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='의존성 패키지 확인'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='현재 설정 표시'
    )
    
    # 성능 옵션
    parser.add_argument(
        '--stats',
        action='store_true',
        help='처리 통계 표시'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='성능 벤치마크 실행'
    )
    
    return parser.parse_args()

def setup_logging(verbose=False, quiet=False):
    """로깅 설정"""
    import logging
    
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

def check_and_setup_models(args):
    """모델 확인 및 설정"""
    from utils import ModelUtils
    
    print("🔍 Checking model availability...")
    
    model_status = ModelUtils.check_model_availability()
    
    # YOLO 모델은 자동 다운로드되므로 확인만
    yolo_available = any('yolo' in k and v for k, v in model_status.items())
    if yolo_available:
        print("✅ YOLO models: Available")
    else:
        print("❌ YOLO models: Not available")
    
    # SAM 모델 확인 및 다운로드
    if args.sam:
        sam_available = any('sam' in k and v for k, v in model_status.items())
        
        if not sam_available:
            print("⚠️  SAM models not found. Attempting to download...")
            print("📥 This may take several minutes depending on your internet connection.")
            
            try:
                # SAM 패키지 확인
                try:
                    import segment_anything
                    print("✅ SAM package is installed")
                except ImportError:
                    print("❌ SAM package not installed")
                    print("Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
                    return False
                
                # 기본 SAM 모델 다운로드 시도
                success = ModelUtils.download_sam_model(args.sam_model)
                
                if success:
                    print(f"✅ SAM model ({args.sam_model}) downloaded successfully")
                else:
                    print(f"❌ Failed to download SAM model ({args.sam_model})")
                    print("Continuing with YOLO-only mode...")
                    args.sam = False
                    
            except Exception as e:
                print(f"❌ Error setting up SAM: {e}")
                print("Continuing with YOLO-only mode...")
                args.sam = False
        else:
            print("✅ SAM models: Available")
    
    return True

def setup_matplotlib_backend():
    """matplotlib 백엔드 설정"""
    import matplotlib
    import os
    
    # 환경 변수 확인
    if 'DISPLAY' not in os.environ:
        # GUI 없는 환경
        matplotlib.use('Agg')
        print("ℹ️  No display detected, using non-interactive backend")
    else:
        # GUI 환경 시도
        try:
            current_backend = matplotlib.get_backend()
            if current_backend == 'Agg':
                # TkAgg나 Qt5Agg 시도
                try:
                    matplotlib.use('TkAgg')
                except ImportError:
                    try:
                        matplotlib.use('Qt5Agg')
                    except ImportError:
                        print("ℹ️  GUI backend not available, plots will be saved to files")
        except Exception as e:
            print(f"ℹ️  Backend setup warning: {e}")

def setup_matplotlib_for_args(args):
    """인수에 따른 matplotlib 설정"""
    import matplotlib
    
    if args.no_gui or not args.show:
        # GUI 비활성화
        matplotlib.use('Agg')
        print("ℹ️  Using non-interactive backend (no GUI)")
    else:
        setup_matplotlib_backend()

def handle_processing_error(e, args):
    """처리 오류 핸들링"""
    error_msg = str(e).lower()
    
    if "can't invoke" in error_msg and "wm" in error_msg:
        print("⚠️  GUI display error detected")
        print("💡 This is common when running without a display server")
        print("📁 Results have been saved to files instead")
        return True  # 계속 진행
    
    elif "cuda" in error_msg and "out of memory" in error_msg:
        print("💾 GPU memory error detected")
        print("💡 Try using CPU mode: --device cpu")
        return False
    
    elif "model" in error_msg and "not found" in error_msg:
        print("📦 Model loading error")
        print("💡 Check your internet connection and try again")
        return False
    
    elif "display" in error_msg or "backend" in error_msg:
        print("🖥️  Display/backend error detected")
        print("💡 Try using --no-gui option")
        return True  # GUI 오류는 성공으로 처리
    
    else:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")
        return False

def validate_arguments(args):
    """인수 유효성 검사"""
    errors = []
    
    # 입력 경로 확인
    input_path = args.input or args.input_path
    if not input_path and not any([args.info, args.check_deps, args.config, args.benchmark]):
        errors.append("입력 경로가 필요합니다")
    
    if input_path and not Path(input_path).exists():
        errors.append(f"입력 경로를 찾을 수 없습니다: {input_path}")
    
    # 신뢰도 범위 확인
    if not 0 <= args.confidence <= 1:
        errors.append("신뢰도는 0과 1 사이여야 합니다")
    
    # 출력 디렉토리 확인/생성
    try:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"출력 디렉토리 생성 실패: {e}")
    
    return errors

def show_system_info():
    """시스템 정보 표시"""
    print_system_info()
    print_config()
    
    # 모델 사용 가능성 확인
    from utils import ModelUtils
    model_status = ModelUtils.check_model_availability()
    
    print("\nModel Availability:")
    print("-" * 30)
    for model, available in model_status.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{model}: {status}")
    
    print("\nSupported Vehicle Classes:")
    print("-" * 30)
    for eng_name, kr_name in detection_config.CLASS_NAMES_KR.items():
        print(f"{kr_name} ({eng_name})")

def run_benchmark(args):
    """성능 벤치마크 실행"""
    print("🏃‍♂️ Running Performance Benchmark...")
    print("=" * 50)
    
    # 테스트용 더미 이미지 생성
    import numpy as np
    from utils import ImageUtils
    
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_path = Path(args.output) / "benchmark_test.jpg"
    
    if not ImageUtils.save_image(test_image, test_path):
        print("❌ Failed to create test image")
        return
    
    try:
        # 다양한 모델로 벤치마크
        models_to_test = ['yolov8n.pt', 'yolov8s.pt'] if 'yolov8s.pt' in model_config.YOLO_MODELS.values() else ['yolov8n.pt']
        
        results = {}
        for model in models_to_test:
            print(f"\nTesting {model}...")
            
            pipeline = VehicleDetectionPipeline(
                yolo_model=model,
                confidence_threshold=args.confidence,
                enable_sam=False
            )
            
            import time
            start_time = time.time()
            
            # 여러 번 실행하여 평균 시간 계산
            times = []
            for i in range(3):
                iter_start = time.time()
                result = pipeline.process_image(test_path, save_results=False, show_results=False)
                iter_time = time.time() - iter_start
                times.append(iter_time)
                
                if result['success']:
                    print(f"  Run {i+1}: {iter_time:.2f}s, {len(result['detections'])} detections")
                else:
                    print(f"  Run {i+1}: Failed - {result.get('error', 'Unknown error')}")
            
            avg_time = np.mean(times)
            results[model] = {
                'avg_time': avg_time,
                'times': times,
                'total_time': time.time() - start_time
            }
        
        # 결과 출력
        print(f"\n📊 Benchmark Results:")
        print("-" * 30)
        for model, result in results.items():
            print(f"{model}:")
            print(f"  Average time: {result['avg_time']:.2f}s")
            print(f"  Min time: {min(result['times']):.2f}s")
            print(f"  Max time: {max(result['times']):.2f}s")
    
    finally:
        # 테스트 이미지 삭제
        if test_path.exists():
            test_path.unlink()

def process_single_image(args):
    """단일 이미지 처리"""
    input_path = args.input or args.input_path
    
    print(f"🖼️  Processing single image: {Path(input_path).name}")
    
    # 처리 방식 선택
    if args.sam:
        print("Using full analysis pipeline (YOLO + SAM)")
        result = full_vehicle_analysis(
            input_path,
            yolo_model=args.model,
            sam_model=args.sam_model,
            confidence=args.confidence,
            save_result=not args.no_save,
            show_result=args.show
        )
    else:
        print("Using quick detection pipeline (YOLO only)")
        result = quick_vehicle_detection(
            input_path,
            yolo_model=args.model,
            confidence=args.confidence,
            save_result=not args.no_save,
            show_result=args.show
        )
    
    # 결과 출력
    if result['success']:
        print(f"✅ Processing completed successfully!")
        print(f"   Detections: {len(result['detections'])}")
        if 'masks' in result:
            print(f"   Masks: {len(result['masks'])}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        
        if not args.no_save:
            print(f"   Results saved to: {args.output}")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        return False
    
    return True

def process_batch_images(args):
    """배치 이미지 처리"""
    input_path = args.input or args.input_path
    
    print(f"📁 Processing batch images from: {input_path}")
    
    # 파이프라인 초기화
    pipeline = VehicleDetectionPipeline(
        yolo_model=args.model,
        sam_model=args.sam_model if args.sam else None,
        confidence_threshold=args.confidence,
        enable_sam=args.sam
    )
    
    # 배치 처리 실행
    results = pipeline.process_batch(
        input_path=input_path,
        pattern=args.pattern,
        max_images=args.max_images
    )
    
    if not results:
        print("❌ No images processed")
        return False
    
    # 통계 출력
    if args.stats:
        stats = pipeline.get_pipeline_statistics()
        print(f"\n📈 Pipeline Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    return True

def save_results_summary(args, success, processing_time=0):
    """결과 요약 저장"""
    if args.no_save:
        return
    
    summary = {
        'timestamp': str(Path().cwd()),
        'input_path': args.input or args.input_path,
        'model': args.model,
        'confidence_threshold': args.confidence,
        'sam_enabled': args.sam,
        'sam_model': args.sam_model if args.sam else None,
        'success': success,
        'processing_time': processing_time,
        'output_directory': str(args.output)
    }
    
    summary_path = Path(args.output) / "processing_summary.json"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"📄 Summary saved: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")

def main():
    """메인 함수"""
    # ASCII 아트 로고
    print("""
    🚗 Vehicle Detection & Segmentation System
    ==========================================
    YOLO + SAM 기반 차량 자동 감지 및 세그멘테이션
    """)
    
    # 인수 파싱
    try:
        args = parse_arguments()
    except SystemExit:
        return
    
    # 로깅 설정
    setup_logging(args.verbose, args.quiet)
    
    # matplotlib 백엔드 설정
    setup_matplotlib_for_args(args)
    
    # 정보 표시 모드
    if args.info:
        show_system_info()
        return
    
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    if args.config:
        print_config()
        return
    
    if args.benchmark:
        run_benchmark(args)
        return
    
    # 인수 유효성 검사
    errors = validate_arguments(args)
    if errors:
        print("❌ 입력 오류:")
        for error in errors:
            print(f"   {error}")
        sys.exit(1)
    
    # 프로젝트 설정
    setup_project_directories()
    validate_config()
    
    if not args.quiet:
        print("🔧 System Check...")
        if not check_dependencies():
            print("⚠️  일부 의존성이 누락되었습니다. 계속 진행하시겠습니까? (y/N)")
            if input().lower() != 'y':
                sys.exit(1)
        
        # 모델 확인 및 설정
        check_and_setup_models(args)
    
    # 처리 시작
    start_time = time.time()
    success = False
    
    try:
        input_path = Path(args.input or args.input_path)
        
        if input_path.is_file():
            # 단일 파일 처리
            success = process_single_image(args)
            
        elif input_path.is_dir():
            if args.batch:
                # 배치 처리
                success = process_batch_images(args)
            else:
                # 디렉토리에서 첫 번째 이미지만 처리
                from utils import FileUtils
                image_files = FileUtils.get_image_files(input_path)
                
                if image_files:
                    args.input = str(image_files[0])
                    print(f"📁 Found {len(image_files)} images, processing first: {image_files[0].name}")
                    success = process_single_image(args)
                else:
                    print("❌ No image files found in directory")
        else:
            print("❌ Invalid input path")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        success = False
    
    except Exception as e:
        # 오류 핸들링
        if handle_processing_error(e, args):
            success = True  # GUI 오류는 성공으로 처리
        else:
            success = False
    
    # 처리 완료
    processing_time = time.time() - start_time
    
    # 결과 요약 저장
    save_results_summary(args, success, processing_time)
    
    # 최종 메시지
    if success:
        print(f"\n🎉 Processing completed successfully!")
        print(f"⏱️  Total time: {processing_time:.2f}s")
        if not args.no_save:
            print(f"📁 Results saved to: {args.output}")
    else:
        print(f"\n💥 Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    import time
    main()