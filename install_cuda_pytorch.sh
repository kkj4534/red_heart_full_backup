#!/bin/bash
# CUDA 지원 PyTorch 설치 스크립트

echo "🔥 CUDA 지원 PyTorch 설치 시작..."

# 현재 PyTorch CPU 버전 제거
echo "🗑️ 기존 CPU PyTorch 제거..."
pip uninstall -y torch torchvision torchaudio

# CUDA 11.8 지원 PyTorch 설치 (WSL에서 안정적)
echo "🚀 CUDA 지원 PyTorch 설치..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 툴킷 설치 (필요시)
echo "🛠️ CUDA 개발 도구 설치..."
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# 설치 확인
echo "✅ 설치 확인..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('CUDA devices:', torch.cuda.device_count())
    print('Device name:', torch.cuda.get_device_name(0))
    print('Device memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
else:
    print('⚠️ CUDA 사용 불가 - WSL GPU 패스스루 확인 필요')
"

echo "🎉 CUDA PyTorch 설치 완료!"
echo "📝 WSL에서 GPU 사용을 위해서는 Windows에서 NVIDIA 드라이버와 WSL CUDA 지원이 필요합니다."