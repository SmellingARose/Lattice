#!/bin/bash
set -e

echo "=== Lattice GPU Setup ==="
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
    echo "[1/3] Checking CUDA toolkit..."
    if nvcc --version &>/dev/null; then
        echo "  nvcc already installed: $(nvcc --version 2>&1 | grep 'release')"
    else
        echo "  Installing nvidia-cuda-toolkit..."
        sudo apt update
        sudo apt install -y nvidia-cuda-toolkit
    fi
fi

# 2. Install HIP headers
echo ""
echo "[2/3] Installing HIP headers..."

# Check if HIP headers already exist
if [ -f /opt/rocm/include/hip/hip_runtime.h ]; then
    echo "  HIP headers already installed at /opt/rocm/include/hip/"
else
    # Try apt first (works on some distros)
    if sudo apt install -y hip-dev 2>/dev/null; then
        echo "  Installed hip-dev via apt"
    else
        # Fallback: download HIP headers from ROCm GitHub (NVIDIA only needs headers)
        echo "  hip-dev not available via apt, downloading HIP headers from GitHub..."
        HIP_VERSION="6.2.4"
        HIP_TAR="/tmp/hip-headers.tar.gz"
        wget -q "https://github.com/ROCm/HIP/archive/refs/tags/rocm-${HIP_VERSION}.tar.gz" -O "$HIP_TAR"
        sudo mkdir -p /opt/rocm/include
        # Extract just the include/hip directory
        sudo tar xzf "$HIP_TAR" -C /opt/rocm/include --strip-components=2 "HIP-rocm-${HIP_VERSION}/include/hip"
        # HIP headers reference hip/hip_version.h which we create
        if [ ! -f /opt/rocm/include/hip/hip_version.h ]; then
            sudo tee /opt/rocm/include/hip/hip_version.h > /dev/null << 'HIPEOF'
#ifndef HIP_VERSION_H
#define HIP_VERSION_H
#define HIP_VERSION_MAJOR 6
#define HIP_VERSION_MINOR 2
#define HIP_VERSION_PATCH 4
#define HIP_VERSION 60200004
#endif
HIPEOF
        fi
        rm -f "$HIP_TAR"
        echo "  HIP headers installed to /opt/rocm/include/hip/"
    fi
fi

# 3. Set environment variables
echo ""
echo "[3/3] Setting environment..."
if [ "$GPU_VENDOR" = "nvidia" ]; then
    # Detect CUDA_PATH
    if [ -d /usr/local/cuda ]; then
        DETECTED_CUDA_PATH=/usr/local/cuda
    elif [ -d /usr ]; then
        DETECTED_CUDA_PATH=/usr
    fi

    export HIP_PLATFORM=nvidia
    export CUDA_PATH="${DETECTED_CUDA_PATH}"

    # Add to .bashrc if not already there
    if ! grep -q "HIP_PLATFORM=nvidia" ~/.bashrc 2>/dev/null; then
        echo '' >> ~/.bashrc
        echo '# Lattice GPU setup (HIP on NVIDIA)' >> ~/.bashrc
        echo "export HIP_PLATFORM=nvidia" >> ~/.bashrc
        echo "export CUDA_PATH=${DETECTED_CUDA_PATH}" >> ~/.bashrc
        echo "  Added HIP_PLATFORM and CUDA_PATH to ~/.bashrc"
    fi
    echo "  HIP_PLATFORM=$HIP_PLATFORM"
    echo "  CUDA_PATH=$CUDA_PATH"
fi

# Verify
echo ""
echo "=== Verification ==="
echo "nvcc:        $(nvcc --version 2>&1 | grep 'release' || echo 'NOT FOUND')"
echo "HIP headers: $(ls /opt/rocm/include/hip/hip_runtime.h 2>/dev/null && echo 'OK' || echo 'MISSING')"
echo "gcc:         $(gcc --version 2>&1 | head -1)"

echo ""
echo "=== Done! Build with: ==="
echo "  make clean && make BACKEND=gpu"
echo ""
echo "Run inspiral:"
echo "  nohup ./build/test_binary_inspiral > inspiral.log 2>&1 &"
echo "  tail -f inspiral.log"
