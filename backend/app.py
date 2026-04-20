"""
Streamlit 기반 차량 감지 웹 애플리케이션
웹 브라우저에서 이미지를 업로드하고 차량 감지 결과를 확인할 수 있습니다.
"""

import streamlit as st
import numpy as np
import cv2
import time
from pathlib import Path
import tempfile
import zipfile
import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# 프로젝트 모듈 import
try:
    from pipeline import VehicleDetectionPipeline, quick_vehicle_detection, full_vehicle_analysis
    from config import detection_config, model_config, OUTPUT_DIR
    from utils import (
        ImageUtils, TextUtils, KoreanTextUtils, AnalysisUtils,
        system_health_check, check_dependencies, logger
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"모듈 로드 실패: {e}")
    st.error("프로젝트 루트 디렉토리에서 실행해주세요: streamlit run app.py")
    MODULES_AVAILABLE = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="🚗 차량 감지 시스템",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "YOLO + SAM 기반 차량 자동 감지 및 세그멘테이션 시스템"
    }
)

# 사이드바 - 설정 패널
def create_sidebar():
    """사이드바 설정 패널 생성"""
    st.sidebar.title("⚙️ 설정")
    
    # 모델 선택
    st.sidebar.subheader("🤖 모델 설정")
    
    yolo_models = {
        'YOLOv8 Nano (빠름)': 'yolov8n.pt',
        'YOLOv8 Small (균형)': 'yolov8s.pt',
        'YOLOv8 Medium (정확)': 'yolov8m.pt',
        'YOLOv8 Large (고성능)': 'yolov8l.pt'
    }
    
    selected_yolo = st.sidebar.selectbox(
        "YOLO 모델 선택",
        list(yolo_models.keys()),
        index=0,
        help="더 큰 모델일수록 정확하지만 느립니다"
    )
    
    # SAM 모델 설정
    enable_sam = st.sidebar.checkbox(
        "🎯 고급 세그멘테이션 (SAM) 활성화",
        value=False,
        help="정밀한 픽셀 단위 세그멘테이션을 제공하지만 처리 시간이 늘어납니다"
    )
    
    sam_model = None
    if enable_sam:
        sam_options = {
            'ViT-H (최고 품질, 느림)': 'vit_h',
            'ViT-L (균형)': 'vit_l',
            'ViT-B (빠름, 낮은 품질)': 'vit_b'
        }
        selected_sam = st.sidebar.selectbox(
            "SAM 모델 선택",
            list(sam_options.keys()),
            help="ViT-H가 가장 정확하지만 처리 시간이 깁니다"
        )
        sam_model = sam_options[selected_sam]
    
    # 감지 설정
    st.sidebar.subheader("🔍 감지 설정")
    
    confidence_threshold = st.sidebar.slider(
        "신뢰도 임계값",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="높을수록 확실한 객체만 감지합니다"
    )
    
    # 디바이스 선택
    device_options = ["자동 선택", "GPU (CUDA)", "CPU"]
    selected_device = st.sidebar.selectbox(
        "처리 디바이스",
        device_options,
        help="GPU가 사용 가능하면 자동으로 선택됩니다"
    )
    
    device_map = {
        "자동 선택": "auto",
        "GPU (CUDA)": "cuda",
        "CPU": "cpu"
    }
    
    return {
        'yolo_model': yolo_models[selected_yolo],
        'enable_sam': enable_sam,
        'sam_model': sam_model,
        'confidence_threshold': confidence_threshold,
        'device': device_map[selected_device]
    }

# 메인 인터페이스
def create_main_interface():
    """메인 인터페이스 생성"""
    
    # 헤더
    st.title("🚗 차량 감지 및 세그멘테이션 시스템")
    st.markdown("---")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📸 이미지 분석", "📊 시스템 정보", "📚 사용 가이드", "🔧 고급 설정"])
    
    return tab1, tab2, tab3, tab4

def process_uploaded_image(uploaded_file, config):
    """업로드된 이미지 처리"""
    if uploaded_file is None:
        return None
        
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # 이미지 로드 및 표시
        image = ImageUtils.load_image(temp_path)
        if image is None:
            st.error("이미지를 로드할 수 없습니다.")
            return None
        
        # 이미지 정보 표시
        image_info = ImageUtils.get_image_info(image)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 업로드된 이미지")
            st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
        
        with col2:
            st.subheader("📋 이미지 정보")
            st.write(f"**크기**: {image_info['width']} × {image_info['height']}")
            st.write(f"**채널**: {image_info['channels']}")
            st.write(f"**종횡비**: {image_info['aspect_ratio']:.2f}")
            st.write(f"**파일 크기**: {image_info['size_mb']:.1f} MB")
        
        # 처리 시작
        st.markdown("---")
        st.subheader("🔄 처리 중...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 처리 시작
        start_time = time.time()
        
        status_text.text("🤖 AI 모델 초기화 중...")
        progress_bar.progress(20)
        
        try:
            if config['enable_sam']:
                # SAM 포함 처리
                status_text.text("🎯 YOLO + SAM 파이프라인 실행 중...")
                progress_bar.progress(40)
                
                result = full_vehicle_analysis(
                    temp_path,
                    yolo_model=config['yolo_model'],
                    sam_model=config['sam_model'],
                    confidence=config['confidence_threshold'],
                    save_result=False,
                    show_result=False
                )
            else:
                # YOLO만 처리
                status_text.text("🚗 차량 감지 중...")
                progress_bar.progress(60)
                
                result = quick_vehicle_detection(
                    temp_path,
                    yolo_model=config['yolo_model'],
                    confidence=config['confidence_threshold'],
                    save_result=False,
                    show_result=False
                )
            
            progress_bar.progress(80)
            status_text.text("📊 결과 분석 중...")
            
            processing_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text("✅ 처리 완료!")
            
            # 결과 표시
            display_results(result, image, processing_time, config)
            
        except Exception as e:
            st.error(f"처리 중 오류 발생: {e}")
            logger.error(f"Processing error: {e}")
        
        finally:
            # 임시 파일 정리
            try:
                Path(temp_path).unlink()
            except OSError as e:
                logger.warning(f"임시 파일 삭제 실패 ({temp_path}): {e}")

    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")

def display_results(result, original_image, processing_time, config):
    """결과 표시"""
    if not result or not result['success']:
        st.error("처리에 실패했습니다.")
        if result and 'error' in result:
            st.error(f"오류: {result['error']}")
        return
    
    detections = result['detections']
    masks = result.get('masks', [])
    
    st.markdown("---")
    st.subheader("🎉 분석 결과")
    
    # 요약 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("감지된 차량", len(detections))
    
    with col2:
        avg_confidence = np.mean([det['confidence'] for det in detections]) if detections else 0
        st.metric("평균 신뢰도", f"{avg_confidence:.1%}")
    
    with col3:
        st.metric("처리 시간", f"{processing_time:.1f}초")
    
    with col4:
        mask_count = len(masks) if masks else 0
        st.metric("생성된 마스크", mask_count)
    
    if not detections:
        st.warning("감지된 차량이 없습니다. 신뢰도 임계값을 낮춰보세요.")
        return
    
    # 시각화 결과
    st.subheader("🖼️ 시각화 결과")
    
    if 'visualizations' in result and result['visualizations']:
        visualizations = result['visualizations']
        
        # 탭으로 다른 시각화 구분
        viz_tabs = []
        viz_names = []
        
        if 'yolo_detection' in visualizations:
            viz_tabs.append("🔍 차량 감지")
            viz_names.append('yolo_detection')
        
        if 'sam_segmentation' in visualizations:
            viz_tabs.append("🎯 세그멘테이션")
            viz_names.append('sam_segmentation')
        
        if 'combined' in visualizations:
            viz_tabs.append("🔗 통합 결과")
            viz_names.append('combined')
        
        if viz_tabs:
            tabs = st.tabs(viz_tabs)
            
            for i, (tab, viz_name) in enumerate(zip(tabs, viz_names)):
                with tab:
                    if viz_name in visualizations:
                        viz_image = visualizations[viz_name]
                        if isinstance(viz_image, np.ndarray):
                            st.image(viz_image, caption=f"{viz_name} 결과", use_column_width=True)
    
    # 상세 결과
    st.subheader("📋 상세 분석 결과")
    
    # 한국어 리포트
    korean_report = KoreanTextUtils.format_korean_report(detections)
    st.text_area("한국어 요약", korean_report, height=200)
    
    # 감지된 차량 목록
    if detections:
        st.subheader("🚗 감지된 차량 목록")
        
        # 데이터프레임으로 정리
        df_data = []
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            kr_name = detection_config.CLASS_NAMES_KR.get(det['class'], det['class'])
            
            df_data.append({
                '번호': i,
                '차량 종류': kr_name,
                '영문명': det['class'],
                '신뢰도': f"{det['confidence']:.1%}",
                '위치 (X,Y)': f"({bbox[0]}, {bbox[1]})",
                '크기': f"{bbox[2]-bbox[0]} × {bbox[3]-bbox[1]}",
                '면적': f"{(bbox[2]-bbox[0])*(bbox[3]-bbox[1]):,} px²"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    # 통계 차트
    if len(detections) > 0:
        st.subheader("📊 통계 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 차량 종류별 분포
            class_counts = {}
            for det in detections:
                kr_name = detection_config.CLASS_NAMES_KR.get(det['class'], det['class'])
                class_counts[kr_name] = class_counts.get(kr_name, 0) + 1
            
            if class_counts:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(class_counts.keys(), class_counts.values())
                ax.set_title('차량 종류별 분포')
                ax.set_xlabel('차량 종류')
                ax.set_ylabel('개수')
                st.pyplot(fig)
        
        with col2:
            # 신뢰도 분포
            confidences = [det['confidence'] for det in detections]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(confidences, bins=min(10, len(confidences)), alpha=0.7, edgecolor='black')
            ax.set_title('신뢰도 분포')
            ax.set_xlabel('신뢰도')
            ax.set_ylabel('빈도')
            ax.axvline(config['confidence_threshold'], color='red', linestyle='--', label='임계값')
            ax.legend()
            st.pyplot(fig)
    
    # 다운로드 섹션
    st.subheader("💾 결과 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 텍스트 리포트 다운로드"):
            report = TextUtils.format_detection_report(detections, include_korean=True)
            st.download_button(
                label="리포트.txt 다운로드",
                data=report,
                file_name=f"vehicle_detection_report_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    with col2:
        if 'visualizations' in result and 'yolo_detection' in result['visualizations']:
            # 결과 이미지 다운로드
            result_image = result['visualizations']['yolo_detection']
            if isinstance(result_image, np.ndarray):
                is_success, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button(
                        label="🖼️ 결과 이미지 다운로드",
                        data=buffer.tobytes(),
                        file_name=f"vehicle_detection_{int(time.time())}.jpg",
                        mime="image/jpeg"
                    )
    
    with col3:
        # JSON 결과 다운로드
        import json
        json_result = {
            'processing_time': processing_time,
            'total_vehicles': len(detections),
            'average_confidence': avg_confidence,
            'detections': [
                {
                    'type': det['class'],
                    'korean_name': detection_config.CLASS_NAMES_KR.get(det['class'], det['class']),
                    'confidence': float(det['confidence']),
                    'bbox': [int(x) for x in det['bbox']],
                    'area': int((det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1]))
                }
                for det in detections
            ]
        }
        
        st.download_button(
            label="📊 JSON 데이터 다운로드",
            data=json.dumps(json_result, indent=2, ensure_ascii=False),
            file_name=f"vehicle_detection_data_{int(time.time())}.json",
            mime="application/json"
        )

def show_system_info():
    """시스템 정보 표시"""
    st.subheader("💻 시스템 정보")
    
    # 시스템 상태 확인
    with st.spinner("시스템 상태 확인 중..."):
        try:
            from utils import ConfigUtils
            system_info = ConfigUtils.get_system_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**플랫폼 정보**")
                st.write(f"- OS: {system_info.get('platform', 'Unknown')}")
                st.write(f"- Python: {system_info.get('python_version', 'Unknown')}")
                
                st.write("**라이브러리 버전**")
                st.write(f"- OpenCV: {system_info.get('opencv_version', 'Unknown')}")
                st.write(f"- PyTorch: {system_info.get('torch_version', 'Unknown')}")
                st.write(f"- Ultralytics: {system_info.get('ultralytics_version', 'Unknown')}")
            
            with col2:
                st.write("**GPU 정보**")
                if system_info.get('cuda_available', False):
                    st.write(f"- CUDA 사용 가능: ✅")
                    st.write(f"- CUDA 버전: {system_info.get('cuda_version', 'Unknown')}")
                    st.write(f"- GPU 개수: {system_info.get('gpu_count', 0)}")
                    st.write(f"- GPU 이름: {system_info.get('gpu_name', 'Unknown')}")
                else:
                    st.write("- CUDA 사용 불가: ❌")
                
                st.write("**의존성 상태**")
                deps_ok = check_dependencies()
                st.write(f"- 의존성: {'✅' if deps_ok else '❌'}")
        
        except Exception as e:
            st.error(f"시스템 정보를 가져올 수 없습니다: {e}")
    
    # 모델 상태 확인
    st.subheader("🤖 모델 상태")
    
    try:
        from utils import ModelUtils
        model_status = ModelUtils.check_model_availability()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**YOLO 모델**")
            for model_name, available in model_status.items():
                if 'yolo' in model_name:
                    status = "✅" if available else "❌"
                    st.write(f"- {model_name}: {status}")
        
        with col2:
            st.write("**SAM 모델**")
            sam_models = {k: v for k, v in model_status.items() if 'sam' in k}
            
            if sam_models:
                for model_name, available in sam_models.items():
                    status = "✅" if available else "❌"
                    st.write(f"- {model_name}: {status}")
                
                if not any(sam_models.values()):
                    st.warning("SAM 모델이 설치되지 않았습니다.")
                    st.info("고급 세그멘테이션을 사용하려면 SAM을 설치하세요:")
                    st.code("pip install git+https://github.com/facebookresearch/segment-anything.git")
            else:
                st.write("SAM 패키지가 설치되지 않았습니다.")
    
    except Exception as e:
        st.error(f"모델 상태를 확인할 수 없습니다: {e}")

def show_usage_guide():
    """사용 가이드 표시"""
    st.subheader("📚 사용 가이드")
    
    st.markdown("""
    ### 🚀 빠른 시작
    1. **이미지 업로드**: '이미지 분석' 탭에서 차량이 포함된 이미지를 업로드하세요
    2. **설정 조정**: 좌측 사이드바에서 모델과 파라미터를 조정하세요
    3. **분석 실행**: 업로드하면 자동으로 분석이 시작됩니다
    4. **결과 확인**: 감지된 차량과 상세 정보를 확인하세요
    
    ### ⚙️ 설정 가이드
    
    **YOLO 모델 선택**
    - **Nano**: 가장 빠르지만 정확도가 낮음 (실시간 처리용)
    - **Small**: 속도와 정확도의 균형
    - **Medium**: 높은 정확도, 적당한 속도
    - **Large**: 최고 정확도, 느림
    
    **SAM 세그멘테이션**
    - 픽셀 단위로 정확한 차량 모양을 추출
    - 처리 시간이 오래 걸리지만 매우 정밀함
    - ViT-H > ViT-L > ViT-B 순으로 품질이 높음
    
    **신뢰도 임계값**
    - 높을수록: 확실한 차량만 감지 (정밀도 ↑, 재현율 ↓)
    - 낮을수록: 더 많은 차량 감지 (정밀도 ↓, 재현율 ↑)
    
    ### 🎯 최적화 팁
    
    **빠른 처리가 필요한 경우**
    - YOLO Nano 모델 사용
    - SAM 비활성화
    - 신뢰도 임계값을 높게 설정 (0.4-0.6)
    
    **높은 정확도가 필요한 경우**
    - YOLO Large 모델 사용
    - SAM ViT-H 모델 활성화
    - 신뢰도 임계값을 낮게 설정 (0.1-0.3)
    
    ### 📋 지원하는 차량 종류
    - 🚗 **자동차** (car): 일반 승용차
    - 🏍️ **오토바이** (motorcycle): 이륜차
    - 🚌 **버스** (bus): 대형 버스
    - 🚚 **트럭** (truck): 화물차
    - 🚲 **자전거** (bicycle): 자전거
    
    ### 🔧 문제 해결
    
    **차량이 감지되지 않는 경우**
    - 신뢰도 임계값을 낮춰보세요 (0.1-0.2)
    - 더 큰 YOLO 모델을 사용해보세요
    - 이미지 품질과 해상도를 확인하세요
    
    **처리가 너무 느린 경우**
    - 작은 YOLO 모델을 사용하세요 (Nano/Small)
    - SAM을 비활성화하세요
    - CPU 대신 GPU를 사용하세요
    
    **메모리 부족 오류**
    - CPU 모드로 전환하세요
    - 이미지 크기를 줄여보세요
    - 브라우저를 새로고침하세요
    """)

def show_advanced_settings():
    """고급 설정 표시"""
    st.subheader("🔧 고급 설정")
    
    # 배치 처리
    st.markdown("### 📁 배치 처리")
    st.info("여러 이미지를 한 번에 처리할 수 있습니다.")
    
    uploaded_files = st.file_uploader(
        "여러 이미지 선택",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="최대 10개의 이미지를 선택할 수 있습니다"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("최대 10개의 이미지만 선택할 수 있습니다.")
        else:
            st.success(f"{len(uploaded_files)}개의 이미지가 선택되었습니다.")
            
            if st.button("🚀 배치 처리 시작"):
                process_batch_images(uploaded_files)
    
    # 성능 모니터링
    st.markdown("### 📊 성능 모니터링")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 메모리 사용량 확인"):
            try:
                from utils import PerformanceUtils
                memory_info = PerformanceUtils.get_memory_usage()
                
                if memory_info:
                    st.metric("RSS 메모리", f"{memory_info['rss']:.1f} MB")
                    st.metric("메모리 사용률", f"{memory_info['percent']:.1f}%")
                else:
                    st.warning("메모리 정보를 가져올 수 없습니다.")
            except Exception as e:
                st.error(f"메모리 정보 오류: {e}")
    
    with col2:
        if st.button("🏥 시스템 상태 점검"):
            with st.spinner("시스템 상태 점검 중..."):
                try:
                    # 시스템 상태 점검 실행
                    health_ok = system_health_check()
                    
                    if health_ok:
                        st.success("✅ 시스템이 정상 작동 중입니다!")
                    else:
                        st.warning("⚠️ 일부 구성 요소에 문제가 있습니다.")
                        
                except Exception as e:
                    st.error(f"상태 점검 중 오류: {e}")

def process_batch_images(uploaded_files):
    """배치 이미지 처리"""
    config = create_sidebar()  # 현재 설정 가져오기
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"처리 중: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            # 처리 실행
            if config['enable_sam']:
                result = full_vehicle_analysis(
                    temp_path,
                    yolo_model=config['yolo_model'],
                    sam_model=config['sam_model'],
                    confidence=config['confidence_threshold'],
                    save_result=False,
                    show_result=False
                )
            else:
                result = quick_vehicle_detection(
                    temp_path,
                    yolo_model=config['yolo_model'],
                    confidence=config['confidence_threshold'],
                    save_result=False,
                    show_result=False
                )
            
            results.append({
                'filename': uploaded_file.name,
                'success': result['success'],
                'detections': len(result['detections']) if result['success'] else 0,
                'processing_time': result.get('processing_time', 0),
                'error': result.get('error', None)
            })
            
            # 임시 파일 정리
            try:
                Path(temp_path).unlink()
            except OSError as e:
                logger.warning(f"임시 파일 삭제 실패 ({temp_path}): {e}")

        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'success': False,
                'detections': 0,
                'processing_time': 0,
                'error': str(e)
            })
    
    # 배치 결과 표시
    progress_bar.progress(1.0)
    status_text.text("✅ 배치 처리 완료!")
    
    # 결과 요약
    successful = sum(1 for r in results if r['success'])
    total_detections = sum(r['detections'] for r in results if r['success'])
    avg_time = np.mean([r['processing_time'] for r in results if r['success']]) if successful > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("처리된 이미지", len(results))
    with col2:
        st.metric("성공률", f"{successful}/{len(results)}")
    with col3:
        st.metric("총 감지 차량", total_detections)
    with col4:
        st.metric("평균 처리 시간", f"{avg_time:.1f}초")
    
    # 상세 결과 테이블
    df_results = pd.DataFrame(results)
    st.subheader("📋 배치 처리 상세 결과")
    st.dataframe(df_results, use_container_width=True)
    
    # 결과 다운로드
    csv_data = df_results.to_csv(index=False)
    st.download_button(
        label="📊 배치 결과 CSV 다운로드",
        data=csv_data,
        file_name=f"batch_results_{int(time.time())}.csv",
        mime="text/csv"
    )

# 메인 애플리케이션
def main():
    """메인 애플리케이션 실행"""
    
    # 모듈 사용 불가능한 경우 종료
    if not MODULES_AVAILABLE:
        st.stop()
    
    # matplotlib 백엔드 설정 (Streamlit용)
    import matplotlib
    matplotlib.use('Agg')
    
    # 사이드바 설정
    config = create_sidebar()
    
    # 메인 인터페이스
    tab1, tab2, tab3, tab4 = create_main_interface()
    
    # 탭별 콘텐츠
    with tab1:
        st.subheader("📸 이미지 업로드 및 분석")
        
        uploaded_file = st.file_uploader(
            "차량이 포함된 이미지를 선택하세요",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="지원 형식: JPG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            process_uploaded_image(uploaded_file, config)
        else:
            # 샘플 이미지 제공
            st.info("💡 이미지를 업로드하면 자동으로 차량 감지가 시작됩니다.")
            
            # 사용 예시 이미지들
            st.subheader("🖼️ 사용 예시")
            
            example_col1, example_col2, example_col3 = st.columns(3)
            
            with example_col1:
                st.image("https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=Car+Example", 
                        caption="승용차 감지 예시")
            
            with example_col2:
                st.image("https://via.placeholder.com/300x200/2196F3/FFFFFF?text=Traffic+Scene", 
                        caption="교통 상황 분석")
            
            with example_col3:
                st.image("https://via.placeholder.com/300x200/FF9800/FFFFFF?text=Parking+Lot", 
                        caption="주차장 모니터링")
            
            st.markdown("""
            ### 📋 지원하는 이미지 유형
            - **교통 상황**: 도로, 교차로, 고속도로의 차량들
            - **주차장**: 주차된 차량들의 개수 및 위치 파악
            - **감시 카메라**: 보안 카메라 영상에서 차량 감지
            - **개별 차량**: 단일 차량의 상세 분석
            
            ### 🎯 최적의 결과를 위한 팁
            - **해상도**: 640×640 픽셀 이상 권장
            - **조명**: 충분한 밝기와 선명도
            - **각도**: 차량이 명확히 보이는 각도
            - **크기**: 10MB 이하의 파일 크기
            """)
    
    with tab2:
        show_system_info()
    
    with tab3:
        show_usage_guide()
    
    with tab4:
        show_advanced_settings()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 20px;'>
        <p>🚗 Vehicle Detection & Segmentation System</p>
        <p>YOLO + SAM 기반 차량 자동 감지 및 세그멘테이션</p>
        <p>Made with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 사이드바에 추가 정보
def add_sidebar_info():
    """사이드바에 추가 정보 표시"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ 정보")
    
    # 현재 상태 표시
    try:
        import torch
        if torch.cuda.is_available():
            st.sidebar.success("🔥 GPU 가속 사용 가능")
            gpu_name = torch.cuda.get_device_name(0)
            st.sidebar.info(f"GPU: {gpu_name}")
        else:
            st.sidebar.warning("💻 CPU 모드")
    except ImportError:
        st.sidebar.error("❌ PyTorch 없음")
    
    # 메모리 사용량 (실시간)
    try:
        from utils import PerformanceUtils
        memory_info = PerformanceUtils.get_memory_usage()
        
        if memory_info:
            st.sidebar.metric("메모리 사용량", f"{memory_info['rss']:.0f} MB")
    except ImportError:
        # psutil 미설치 환경에서는 메모리 표시를 건너뜀
        pass
    except Exception as e:
        logger.debug(f"메모리 사용량 조회 실패: {e}")
    
    # 도움말 링크
    st.sidebar.markdown("### 🔗 유용한 링크")
    st.sidebar.markdown("- [YOLOv8 공식 문서](https://docs.ultralytics.com/)")
    st.sidebar.markdown("- [SAM 논문](https://arxiv.org/abs/2304.02643)")
    st.sidebar.markdown("- [Streamlit 문서](https://docs.streamlit.io/)")

# CSS 스타일링
def add_custom_css():
    """커스텀 CSS 스타일 추가"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stProgress .st-bo {
        background-color: #e0e0e0;
    }
    
    .stProgress .st-bp {
        background-color: #4CAF50;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #fafafa;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 애플리케이션 실행
if __name__ == "__main__":
    # CSS 스타일 적용
    add_custom_css()
    
    # 사이드바 정보 추가
    add_sidebar_info()
    
    # 메인 애플리케이션 실행
    main()
