#!/bin/bash

echo "🚗 Vehicle Detection System Setup"
echo "================================="

# 기본 패키지 설치
echo "📦 Installing basic requirements..."
pip install -r requirements.txt

# SAM 설치 옵션
echo ""
echo "🤖 Do you want to install SAM for advanced segmentation? (y/N)"
read -r install_sam

if [[ $install_sam =~ ^[Yy]$ ]]; then
    echo "📥 Installing SAM (this may take a few minutes)..."
    pip install git+https://github.com/facebookresearch/segment-anything.git
    
    if [ $? -eq 0 ]; then
        echo "✅ SAM installed successfully"
        
        echo "📥 Do you want to download SAM models now? (y/N)"
        read -r download_models
        
        if [[ $download_models =~ ^[Yy]$ ]]; then
            echo "📥 Downloading SAM models..."
            mkdir -p models
            
            echo "Downloading ViT-H model (largest, best quality)..."
            wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
            
            echo "✅ SAM models downloaded"
        fi
    else
        echo "❌ SAM installation failed"
    fi
else
    echo "ℹ️  SAM not installed - only YOLO detection will be available"
fi

# 테스트 실행
echo ""
echo "🧪 Running system test..."
python utils.py

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Usage examples:"
echo "  python main.py image.jpg                    # Basic detection"
echo "  python main.py image.jpg --sam              # With segmentation"  
echo "  python main.py image.jpg --show --no-gui    # Safe display mode"
echo "  python main.py --info                       # System information"
