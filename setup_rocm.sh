#!/bin/bash
set -euo pipefail

# Lattice GPU setup for vast.ai containers (Ubuntu 22.04/24.04, x86_64).
# Template: vastai/base-image:stock-ubuntu22.04
#
# Key constraint: vast.ai bind-mounts NVIDIA drivers into containers.
# Any apt package that installs nvidia driver files (nvidia-persistenced,
# libnvidia-compute, etc.) will fail with "Invalid cross-device link".
# This rules out the `cuda` metapackage and `hip-runtime-nvidia` via apt.
#
# Solution: cuda-toolkit-12-4 (toolkit only, no drivers) + git clone for
# HIP headers (no apt dependency chain).
#
# Override ROCm tag: ROCM_TAG=rocm-7.2.0 ./setup_rocm.sh

echo "=== Lattice GPU Setup ==="
echo ""

# Sanity checks
if ! command -v apt-get &>/dev/null; then
    echo "ERROR: apt-get not found. This script requires Ubuntu/Debian."
    exit 1
fi
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "ERROR: Only x86_64 is supported (detected: $ARCH)."
    exit 1
fi

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
    # CUDA toolkit (no driver packages — safe in containers).
    # Check common paths since nvcc may not be in PATH yet.
    NVCC_BIN=""
    for p in /usr/local/cuda/bin/nvcc /usr/bin/nvcc; do
        if [ -x "$p" ]; then NVCC_BIN="$p"; break; fi
    done
    if [ -n "$NVCC_BIN" ]; then
        echo "  nvcc already installed: $($NVCC_BIN --version 2>&1 | grep 'release')"
    else
        echo "  Installing CUDA 12.4 toolkit..."
        # Remove conflicting keyring configs (vast.ai containers may have stale ones)
        sudo rm -f /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true
        sudo rm -f /usr/share/keyrings/cuda-*.gpg /usr/share/keyrings/nvidia-cuda.gpg 2>/dev/null || true
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
        ROCM_TAG="${ROCM_TAG:-rocm-6.2.4}"
        CLONE_DIR=$(mktemp -d)
        trap 'rm -rf "$CLONE_DIR"' EXIT

        # Need headers from 3 repos: HIP (main), clr (amd_detail), hipother (nvidia_detail)
        git clone --depth 1 --branch "$ROCM_TAG" https://github.com/ROCm/HIP.git "$CLONE_DIR/HIP"
        git clone --depth 1 --branch "$ROCM_TAG" https://github.com/ROCm/clr.git "$CLONE_DIR/clr"
        git clone --depth 1 --branch "$ROCM_TAG" https://github.com/ROCm/hipother.git "$CLONE_DIR/hipother"

        sudo mkdir -p /opt/rocm/include/hip
        sudo cp -r "$CLONE_DIR/HIP/include/hip/"* /opt/rocm/include/hip/
        sudo cp -r "$CLONE_DIR/clr/hipamd/include/hip/amd_detail" /opt/rocm/include/hip/
        sudo cp -r "$CLONE_DIR/hipother/hipnv/include/hip/nvidia_detail" /opt/rocm/include/hip/

        # Generate hip_version.h (normally created by CMake during ROCm build)
        sudo tee /opt/rocm/include/hip/hip_version.h > /dev/null << 'VEOF'
#ifndef HIP_VERSION_H
#define HIP_VERSION_H

#define HIP_VERSION_MAJOR 6
#define HIP_VERSION_MINOR 2
#define HIP_VERSION_PATCH 41134
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
#define HIP_VERSION_GITHASH "0"
#define HIP_VERSION_BUILD_NAME ""

#endif
VEOF

        rm -rf "$CLONE_DIR"
        trap - EXIT
        echo "  HIP headers installed (ROCm $ROCM_TAG)"
    fi
else
    echo "  Installing ROCm HIP SDK..."
    # Add ROCm apt repository if not already present
    if ! apt-cache policy 2>/dev/null | grep -q "repo.radeon.com"; then
        echo "  Adding ROCm apt repository..."
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest jammy main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
        sudo apt-get update -qq
    fi
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
if [ "$GPU_VENDOR" = "nvidia" ]; then
    echo "nvcc:          $(nvcc --version 2>&1 | grep 'release' || echo 'NOT FOUND')"
    echo "HIP headers:   $(test -f /opt/rocm/include/hip/hip_runtime.h && echo 'OK' || echo 'MISSING')"
    echo "nvidia_detail: $(test -f /opt/rocm/include/hip/nvidia_detail/nvidia_hip_runtime.h && echo 'OK' || echo 'MISSING')"
    echo "hip_version.h: $(test -f /opt/rocm/include/hip/hip_version.h && echo 'OK' || echo 'MISSING')"
else
    echo "hipcc:         $(hipcc --version 2>&1 | head -1 || echo 'NOT FOUND')"
    echo "HIP headers:   $(test -f /opt/rocm/include/hip/hip_runtime.h && echo 'OK' || echo 'MISSING')"
fi

echo ""
echo "=== Build ==="
echo "  source ~/.bashrc"
echo "  make clean && make BACKEND=gpu"
echo ""
echo "=== Run inspiral ==="
echo "  # Option A: foreground"
echo "  make BACKEND=gpu test-inspiral"
echo ""
echo "  # Option B: background (for long runs / unstable SSH)"
echo "  nohup make BACKEND=gpu test-inspiral > inspiral.log 2>&1 &"
echo "  tail -f inspiral.log"
