#!/bin/bash
set -e

# Lattice GPU setup for vast.ai containers (any template, Ubuntu 22.04).
# Template: vastai/base-image:stock-ubuntu22.04
#
# Key constraint: vast.ai bind-mounts NVIDIA drivers into containers.
# Any apt package that installs nvidia driver files (nvidia-persistenced,
# libnvidia-compute, etc.) will fail with "Invalid cross-device link".
# This rules out the `cuda` metapackage and `hip-runtime-nvidia` via apt.
#
# Solution: cuda-toolkit-12-4 (toolkit only, no drivers) + git clone for
# HIP headers (no apt dependency chain).

echo "=== Lattice GPU Setup ==="
echo ""

# 1. Build tools
echo "[1/4] Installing build tools..."
sudo apt-get update -qq
sudo apt-get install -y gcc g++ make git wget

# 2. Detect GPU
echo ""
echo "[2/4] Detecting GPU..."
if nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  NVIDIA GPU: $GPU_NAME"
elif rocminfo &>/dev/null; then
    GPU_VENDOR="amd"
    echo "  AMD GPU detected"
else
    echo "  No GPU detected. Install drivers first."
    exit 1
fi

# 3. GPU toolkit + HIP headers
echo ""
echo "[3/4] Installing GPU toolkit..."
if [ "$GPU_VENDOR" = "nvidia" ]; then
    # CUDA toolkit (no driver packages — safe in containers)
    if nvcc --version &>/dev/null; then
        echo "  nvcc already installed: $(nvcc --version 2>&1 | grep 'release')"
    else
        echo "  Installing CUDA 12.4 toolkit..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
            -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        sudo apt-get update -qq
        # cuda-toolkit-12-4, NOT cuda (which pulls in driver packages)
        sudo apt-get install -y cuda-toolkit-12-4
    fi

    # HIP headers via git clone (apt is not viable in containers — hip-runtime-nvidia
    # depends on the `cuda` metapackage which installs driver files and crashes dpkg)
    if [ -f /opt/rocm/include/hip/nvidia_detail/nvidia_hip_runtime.h ]; then
        echo "  HIP headers already installed"
    else
        echo "  Installing HIP headers from GitHub..."
        ROCM_TAG="rocm-6.2.4"
        TMPDIR=$(mktemp -d)

        git clone --depth 1 --branch "$ROCM_TAG" https://github.com/ROCm/HIP.git "$TMPDIR/HIP"
        git clone --depth 1 --branch "$ROCM_TAG" https://github.com/ROCm/hipother.git "$TMPDIR/hipother"

        sudo mkdir -p /opt/rocm/include/hip
        sudo cp -r "$TMPDIR/HIP/include/hip/"* /opt/rocm/include/hip/
        sudo cp -r "$TMPDIR/hipother/hipnv/include/hip/nvidia_detail" /opt/rocm/include/hip/

        rm -rf "$TMPDIR"
        echo "  HIP headers installed (ROCm $ROCM_TAG)"
    fi
else
    echo "  Installing ROCm HIP SDK..."
    sudo apt-get install -y hip-dev rocm-hip-sdk
fi

# 4. Environment variables
echo ""
echo "[4/4] Setting environment..."
if [ "$GPU_VENDOR" = "nvidia" ]; then
    if [ -d /usr/local/cuda ]; then
        CUDA_PATH=/usr/local/cuda
    else
        CUDA_PATH=/usr
    fi

    if ! grep -q "HIP_PLATFORM=nvidia" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << EOF

# Lattice GPU (HIP on NVIDIA)
export HIP_PLATFORM=nvidia
export CUDA_PATH=${CUDA_PATH}
export PATH=${CUDA_PATH}/bin:\$PATH
EOF
        echo "  Added to ~/.bashrc"
    fi

    export HIP_PLATFORM=nvidia
    export CUDA_PATH="${CUDA_PATH}"
    export PATH="${CUDA_PATH}/bin:$PATH"
    echo "  HIP_PLATFORM=nvidia"
    echo "  CUDA_PATH=${CUDA_PATH}"
fi

# Verify
echo ""
echo "=== Verification ==="
echo "gcc:           $(gcc --version 2>&1 | head -1)"
echo "nvcc:          $(nvcc --version 2>&1 | grep 'release' || echo 'NOT FOUND')"
echo "HIP headers:   $(test -f /opt/rocm/include/hip/hip_runtime.h && echo 'OK' || echo 'MISSING')"
if [ "$GPU_VENDOR" = "nvidia" ]; then
    echo "nvidia_detail: $(test -f /opt/rocm/include/hip/nvidia_detail/nvidia_hip_runtime.h && echo 'OK' || echo 'MISSING')"
fi

echo ""
echo "=== Build ==="
echo "  source ~/.bashrc"
echo "  make clean && make BACKEND=gpu"
echo ""
echo "=== Run inspiral ==="
echo "  nohup ./build/test_binary_inspiral > inspiral.log 2>&1 &"
echo "  tail -f inspiral.log"
