#!/bin/bash
set -e

echo "=== Lattice GPU Setup: ROCm + HIP ==="
echo ""

# Detect GPU vendor
if nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "Detected NVIDIA GPU: $GPU_NAME"
elif rocminfo &>/dev/null; then
    GPU_VENDOR="amd"
    echo "Detected AMD GPU"
else
    echo "No GPU detected. Install GPU drivers first."
    exit 1
fi

# 1. Install CUDA toolkit (NVIDIA only)
if [ "$GPU_VENDOR" = "nvidia" ]; then
    echo ""
    echo "[1/3] Installing CUDA toolkit..."
    sudo apt update
    sudo apt install -y nvidia-cuda-toolkit
fi

# 2. Install ROCm + HIP headers
echo ""
echo "[2/3] Installing ROCm + HIP dev tools..."
if ! dpkg -l | grep -q amdgpu-install; then
    wget -q https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb -O /tmp/amdgpu-install.deb
    sudo apt install -y /tmp/amdgpu-install.deb
fi
sudo apt install -y hip-dev rocm-hip-sdk

# 3. Set environment variables
echo ""
echo "[3/3] Setting environment..."
if [ "$GPU_VENDOR" = "nvidia" ]; then
    # Add to current shell
    export HIP_PLATFORM=nvidia
    export CUDA_PATH=/usr

    # Add to .bashrc if not already there
    if ! grep -q "HIP_PLATFORM=nvidia" ~/.bashrc 2>/dev/null; then
        echo '' >> ~/.bashrc
        echo '# Lattice GPU setup (HIP on NVIDIA)' >> ~/.bashrc
        echo 'export HIP_PLATFORM=nvidia' >> ~/.bashrc
        echo 'export CUDA_PATH=/usr' >> ~/.bashrc
        echo "Added HIP_PLATFORM and CUDA_PATH to ~/.bashrc"
    fi
fi

# Verify
echo ""
echo "=== Verification ==="
echo "nvcc: $(nvcc --version 2>&1 | grep 'release')"
echo "HIP headers: $(ls /opt/rocm/include/hip/hip_runtime.h 2>/dev/null && echo 'OK' || echo 'MISSING')"
if [ "$GPU_VENDOR" = "nvidia" ]; then
    echo "HIP_PLATFORM: $HIP_PLATFORM"
    echo "CUDA_PATH: $CUDA_PATH"
fi

echo ""
echo "=== Done! Build with: ==="
echo "  make clean && make BACKEND=gpu"
echo ""
echo "Run inspiral:"
echo "  nohup build/test_binary_inspiral > inspiral.log 2>&1 &"
echo "  tail -f inspiral.log"
