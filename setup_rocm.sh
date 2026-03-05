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

HEADERS_OK=0
if [ -f /opt/rocm/include/hip/hip_runtime.h ]; then
    if [ "$GPU_VENDOR" = "nvidia" ] && [ ! -f /opt/rocm/include/hip/nvidia_detail/nvidia_hip_runtime.h ]; then
        echo "  HIP headers found but nvidia_detail missing — reinstalling..."
        sudo rm -rf /opt/rocm/include/hip
    else
        echo "  HIP headers already installed at /opt/rocm/include/hip/"
        HEADERS_OK=1
    fi
fi

if [ "$HEADERS_OK" -eq 0 ]; then
    # Try native apt first (works when ROCm repo matches distro)
    if sudo apt install -y hip-dev 2>/dev/null; then
        echo "  Installed hip-dev via apt"
    else
        # Fallback: add ROCm jammy repo (headers are distro-independent)
        echo "  hip-dev not in default repos, adding ROCm apt repo (jammy)..."
        ROCM_VERSION="6.2.4"

        # Add ROCm GPG key
        wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
            | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg 2>/dev/null

        # Add ROCm jammy repo (works on any Ubuntu — headers are arch-independent)
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" \
            | sudo tee /etc/apt/sources.list.d/rocm-jammy.list > /dev/null

        sudo apt update

        if [ "$GPU_VENDOR" = "nvidia" ]; then
            # NVIDIA: install hip-runtime-nvidia (includes nvidia_detail headers)
            sudo apt install -y hip-runtime-nvidia hip-dev || {
                # If hip-dev still fails, just get the runtime (has the headers we need)
                sudo apt install -y hip-runtime-nvidia
            }
        else
            # AMD: full HIP SDK
            sudo apt install -y hip-dev rocm-hip-sdk
        fi

        echo "  HIP headers installed from ROCm ${ROCM_VERSION} repo"
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
echo "nvidia_detail: $(ls /opt/rocm/include/hip/nvidia_detail/nvidia_hip_runtime.h 2>/dev/null && echo 'OK' || echo 'MISSING')"
echo "gcc:         $(gcc --version 2>&1 | head -1)"

echo ""
echo "=== Done! Build with: ==="
echo "  export HIP_PLATFORM=nvidia CUDA_PATH=/usr"
echo "  make clean && make BACKEND=gpu"
echo ""
echo "Run inspiral:"
echo "  nohup ./build/test_binary_inspiral > inspiral.log 2>&1 &"
echo "  tail -f inspiral.log"
