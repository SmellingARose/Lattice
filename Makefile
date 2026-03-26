# Lattice — 3D Numerical Relativity Simulator
# Build system
#
# Usage:
#   make                    # optimized CPU build (OpenMP threads)
#   make BACKEND=gpu        # GPU build (HIP — AMD + NVIDIA)
#   make debug              # debug build with sanitizers
#   make test               # run all tests
#   make clean
#
# GPU build requires ROCm (AMD) or HIP CUDA backend (NVIDIA).
# Set HIPCC to your hipcc path if not in PATH.

BACKEND ?= cpu
FD_ORDER ?= 6
EM ?= off
HDF5 ?= off

# Platform detection: macOS vs Linux
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    CC ?= clang
else
    CC ?= gcc
endif

# Source files (common to all backends)
CORE_SRC    = src/core/grid.c
EVOLUTION_SRC = src/evolution/ccz4_rhs.c src/evolution/dissipation.c src/evolution/maxwell_rhs.c
NUMERICS_SRC  = src/numerics/rk4.c
INITIAL_SRC   = src/initial_data/puncture.c src/initial_data/bowen_york.c src/initial_data/jfnk_solver.c src/initial_data/kerr_quasi_isotropic.c
DIAG_SRC      = src/diagnostics/constraints.c src/diagnostics/ah_finder.c src/diagnostics/psi4.c src/diagnostics/bh_tracker.c
BOUNDARY_SRC  = src/boundary/sommerfeld.c
IO_SRC        = src/io/output.c src/io/checkpoint.c
AMR_SRC       = src/amr/block.c src/amr/mesh.c src/amr/meshblock_pack.c src/amr/ghost_exchange.c \
                src/amr/prolongation.c src/amr/restriction.c src/amr/criterion.c src/amr/refine.c
MAIN_SRC      = src/main.c

# Backend selection
ifeq ($(BACKEND),cpu)
    BACKEND_SRC = src/backend/backend_cpu.c
    LTO_FLAGS = -flto
    ifeq ($(UNAME),Darwin)
        # macOS: clang needs libomp from Homebrew
        OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
        BACKEND_FLAGS = -Xclang -fopenmp -I$(OMP_PREFIX)/include
        BACKEND_LIBS = -L$(OMP_PREFIX)/lib -lomp
    else
        # Linux: gcc has OpenMP built in
        BACKEND_FLAGS = -fopenmp
        BACKEND_LIBS = -lgomp
    endif
else ifeq ($(BACKEND),gpu)
    # HIP GPU backend — works on AMD (ROCm) and NVIDIA (HIP CUDA backend).
    # Two-phase compilation: gcc compiles host C files, hipcc compiles
    # device-side C files (as C++) and backend_hip.cpp, then link all.
    #
    # NVIDIA: set HIP_PLATFORM=nvidia CUDA_PATH=/usr (or wherever nvcc lives)
    # AMD:    default (HIP_PLATFORM=amd)
    HIPCC ?= hipcc
    BACKEND_SRC = src/backend/backend_hip.cpp
    HIP_DEVICE_SRC = src/evolution/ccz4_rhs.c src/evolution/maxwell_rhs.c \
                     src/evolution/dissipation.c src/boundary/sommerfeld.c \
                     src/diagnostics/constraints.c src/diagnostics/psi4.c
    LTO_FLAGS =
    HIP_FLAGS = -DLATTICE_HIP -DLATTICE_GPU
    BACKEND_FLAGS = -fopenmp
    BACKEND_LIBS =
    # NVIDIA platform: hipcc needs CUDA_PATH; host linker needs -lcudart
    # Auto-detect ROCm path for HIP headers
    ROCM_PATH ?= $(wildcard /opt/rocm)
    ifeq ($(ROCM_PATH),)
        ROCM_PATH := $(firstword $(wildcard /opt/rocm-*))
    endif
    ifeq ($(HIP_PLATFORM),nvidia)
        CUDA_PATH ?= /usr/local/cuda
        BACKEND_LIBS += -L$(CUDA_PATH)/lib64 -lcudart
    endif
else
    $(error Unknown BACKEND=$(BACKEND). Use cpu or gpu)
endif

ALL_SRC = $(CORE_SRC) $(EVOLUTION_SRC) $(NUMERICS_SRC) $(INITIAL_SRC) \
          $(DIAG_SRC) $(BOUNDARY_SRC) $(IO_SRC) $(AMR_SRC) $(BACKEND_SRC)

# For HIP GPU builds, separate device sources from host sources.
# Device sources are compiled with hipcc; host sources with gcc.
ifeq ($(BACKEND),gpu)
    HOST_SRC = $(filter-out $(HIP_DEVICE_SRC) $(BACKEND_SRC),$(ALL_SRC))
endif

# HDF5 support (optional, for CCE worldtube output).
# make HDF5=on  → adds -DLATTICE_HDF5, links libhdf5, compiles cce_worldtube.c.
# make HDF5=off → CCE code excluded via #ifdef, zero impact on build.
ifeq ($(HDF5),on)
    HDF5_FLAGS = -DLATTICE_HDF5 $(shell pkg-config --cflags hdf5)
    HDF5_LIBS = $(shell pkg-config --libs hdf5)
    ALL_SRC += src/diagnostics/cce_worldtube.c
else
    HDF5_FLAGS =
    HDF5_LIBS =
endif

# EM compile-time dispatch: EM=on adds -DLATTICE_EM_ENABLED.
# When off (default), COMPILED_NUM_FIELDS=25 lets the compiler eliminate EM
# field iterations from hot loops. When on, COMPILED_NUM_FIELDS=31.
ifeq ($(EM),on)
    EM_FLAGS = -DLATTICE_EM_ENABLED
else
    EM_FLAGS =
endif

# Compiler flags
INCLUDES = -I src
CFLAGS_BASE = -std=c17 -Wall -Wextra -Werror -D_GNU_SOURCE -Wno-unused-but-set-variable -DFD_ORDER=$(FD_ORDER) $(EM_FLAGS) $(HDF5_FLAGS) $(INCLUDES) $(BACKEND_FLAGS)
HOST_OPT ?= -O3
CFLAGS_OPT  = $(CFLAGS_BASE) $(HOST_OPT) -ffast-math -march=native $(LTO_FLAGS)
CFLAGS_DBG  = $(CFLAGS_BASE) -O0 -g -fsanitize=address,undefined -DDEBUG

LDFLAGS = $(BACKEND_LIBS) $(HDF5_LIBS) -lm $(LTO_FLAGS)

# Build directory
BUILD = build

# Targets
.PHONY: all debug test test-single-bh test-convergence test-constraints test-head-on test-amr-mesh test-amr-ghost test-amr-prolong test-amr-refine test-amr-evolve test-pack-evolve test-subcycle test-bowen-york test-hispid test-maxwell test-ah test-psi4 test-cce test-cp-bc test-inspiral test-inspiral-smoke test-inspiral-solver test-inspiral-convergence test-jfnk test-checkpoint test-nbody-track test-gpu-tracker clean

all: $(BUILD)/lattice

ifeq ($(BACKEND),gpu)
# HIP GPU build: two-phase compilation.
# Phase 1: gcc compiles host-only C files to .o
# Phase 2: hipcc compiles device C files (as C++) and backend_hip.cpp to .o
# Phase 3: hipcc links all objects
HOST_OBJS = $(patsubst %.c,$(BUILD)/host/%.o,$(HOST_SRC) $(MAIN_SRC))
DEVICE_C_OBJS = $(patsubst %.c,$(BUILD)/device/%.o,$(HIP_DEVICE_SRC))
DEVICE_CXX_OBJS = $(BUILD)/device/src/backend/backend_hip.o

# NVIDIA hipcc wraps nvcc which has different flag syntax
ifeq ($(HIP_PLATFORM),nvidia)
    HIP_CXXFLAGS = -std=c++17 -O3 -DFD_ORDER=$(FD_ORDER) $(EM_FLAGS) $(HDF5_FLAGS) $(INCLUDES) $(HIP_FLAGS) \
                   -I$(ROCM_PATH)/include -D__HIP_PLATFORM_NVIDIA__ -rdc=true \
                   --compiler-options -Wall,-Wextra,-Wno-unused-parameter,-Wno-unused-but-set-variable
    HIP_LANG_FLAG = -x cu
    # Use nvcc directly (hipcc wrapper adds flags that break with Ubuntu CUDA toolkit)
    HIPCC := nvcc
else
    HIP_CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -DFD_ORDER=$(FD_ORDER) $(EM_FLAGS) $(HDF5_FLAGS) $(INCLUDES) $(HIP_FLAGS) \
                   -Wno-unused-parameter -Wno-unused-but-set-variable
    HIP_LANG_FLAG = -x hip
endif

$(BUILD)/host/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) -c -o $@ $<

$(BUILD)/device/%.o: %.c
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIP_CXXFLAGS) $(HIP_LANG_FLAG) -c -o $@ $<

$(BUILD)/device/src/backend/backend_hip.o: src/backend/backend_hip.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIP_CXXFLAGS) $(HIP_LANG_FLAG) -c -o $@ $<

$(BUILD)/lattice: $(HOST_OBJS) $(DEVICE_C_OBJS) $(DEVICE_CXX_OBJS)
	@mkdir -p $(BUILD)
ifeq ($(HIP_PLATFORM),nvidia)
	$(HIPCC) -rdc=true -o $@ $^ $(HDF5_LIBS) -lm -lgomp -lcudadevrt -lcudart
else
	$(HIPCC) -fopenmp -o $@ $^ $(HDF5_LIBS) -lm
endif
	@echo "Built: $@ (HIP GPU)"
else
# CPU build: single-phase gcc compilation.
$(BUILD)/lattice: $(ALL_SRC) $(MAIN_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ $(ALL_SRC) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built: $@"
endif

debug: $(BUILD)/lattice_debug

$(BUILD)/lattice_debug: $(ALL_SRC) $(MAIN_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_DBG) -o $@ $(ALL_SRC) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built: $@ (debug)"

# CCE test variables (shared between CPU and GPU)
CCE_HDF5_FLAGS = -DLATTICE_HDF5 $(shell pkg-config --cflags hdf5)
CCE_HDF5_LIBS = $(shell pkg-config --libs hdf5)
ALL_SRC_BASE = $(CORE_SRC) $(EVOLUTION_SRC) $(NUMERICS_SRC) $(INITIAL_SRC) \
               $(DIAG_SRC) $(BOUNDARY_SRC) $(IO_SRC) $(AMR_SRC) $(BACKEND_SRC)

# Test build rules — GPU needs two-phase compilation, CPU uses single-phase.
ifeq ($(BACKEND),gpu)
# GPU test builds: compile test source with gcc (host), link with hipcc
# against pre-built library objects (host .o + device .o).
LIB_HOST_OBJS = $(patsubst %.c,$(BUILD)/host/%.o,$(HOST_SRC))
ALL_LIB_OBJS = $(LIB_HOST_OBJS) $(DEVICE_C_OBJS) $(DEVICE_CXX_OBJS)

# Default pattern rule: handles all standard tests
# NVIDIA: nvcc needs -Xcompiler -fopenmp; AMD: hipcc takes -fopenmp directly
ifeq ($(HIP_PLATFORM),nvidia)
    GPU_LINK_FLAGS = -rdc=true -lgomp -lcudadevrt -lcudart
else
    GPU_LINK_FLAGS = -fopenmp
endif

$(BUILD)/test_%: tests/test_%.c $(ALL_LIB_OBJS)
	@mkdir -p $(BUILD)/host/tests
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) -c -o $(BUILD)/host/tests/test_$*.o $<
	$(HIPCC) $(GPU_LINK_FLAGS) -o $@ $(BUILD)/host/tests/test_$*.o $(ALL_LIB_OBJS) $(HDF5_LIBS) -lm

# EM tests: recompile device + host sources with -DLATTICE_EM_ENABLED
DEVICE_C_OBJS_EM = $(patsubst %.c,$(BUILD)/device_em/%.o,$(HIP_DEVICE_SRC))
$(BUILD)/device_em/%.o: %.c
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIP_CXXFLAGS) -DLATTICE_EM_ENABLED $(HIP_LANG_FLAG) -c -o $@ $<

LIB_HOST_OBJS_EM = $(patsubst %.c,$(BUILD)/host_em/%.o,$(HOST_SRC))
$(BUILD)/host_em/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) -DLATTICE_EM_ENABLED -c -o $@ $<

ALL_LIB_OBJS_EM = $(LIB_HOST_OBJS_EM) $(DEVICE_C_OBJS_EM) $(DEVICE_CXX_OBJS)

$(BUILD)/test_maxwell: tests/test_maxwell.c $(ALL_LIB_OBJS_EM)
	@mkdir -p $(BUILD)/host/tests
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) -DLATTICE_EM_ENABLED -c -o $(BUILD)/host/tests/test_maxwell.o $<
	$(HIPCC) $(GPU_LINK_FLAGS) -o $@ $(BUILD)/host/tests/test_maxwell.o $(ALL_LIB_OBJS_EM) $(HDF5_LIBS) -lm

$(BUILD)/test_maxwell_debug: tests/test_maxwell_debug.c $(ALL_LIB_OBJS_EM)
	@mkdir -p $(BUILD)/host/tests
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) -DLATTICE_EM_ENABLED -c -o $(BUILD)/host/tests/test_maxwell_debug.o $<
	$(HIPCC) $(GPU_LINK_FLAGS) -o $@ $(BUILD)/host/tests/test_maxwell_debug.o $(ALL_LIB_OBJS_EM) $(HDF5_LIBS) -lm

# CCE test: needs HDF5 flags + extra source file
$(BUILD)/test_cce_worldtube: tests/test_cce_worldtube.c $(ALL_LIB_OBJS) src/diagnostics/cce_worldtube.c
	@mkdir -p $(BUILD)/host/tests $(BUILD)/host/src/diagnostics
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) $(CCE_HDF5_FLAGS) -c -o $(BUILD)/host/tests/test_cce_worldtube.o $<
	$(CC) $(CFLAGS_OPT) $(HIP_FLAGS) $(CCE_HDF5_FLAGS) -c -o $(BUILD)/host/src/diagnostics/cce_worldtube.o src/diagnostics/cce_worldtube.c
	$(HIPCC) $(GPU_LINK_FLAGS) -o $@ $(BUILD)/host/tests/test_cce_worldtube.o $(BUILD)/host/src/diagnostics/cce_worldtube.o $(ALL_LIB_OBJS) $(CCE_HDF5_LIBS) -lm

else
# CPU test builds: single-phase gcc compilation.
# Default pattern rule: handles all standard tests.
$(BUILD)/test_%: tests/test_%.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ $< $(ALL_SRC) $(LDFLAGS)

# EM tests: need -DLATTICE_EM_ENABLED on all sources
$(BUILD)/test_maxwell: tests/test_maxwell.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -DLATTICE_EM_ENABLED -o $@ $< $(ALL_SRC) $(LDFLAGS)

$(BUILD)/test_maxwell_debug: tests/test_maxwell_debug.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -DLATTICE_EM_ENABLED -o $@ $< $(ALL_SRC) $(LDFLAGS)

# CCE test: needs HDF5 flags + extra source file
$(BUILD)/test_cce_worldtube: tests/test_cce_worldtube.c $(ALL_SRC_BASE) src/diagnostics/cce_worldtube.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) $(CCE_HDF5_FLAGS) -o $@ $< $(ALL_SRC_BASE) src/diagnostics/cce_worldtube.c $(LDFLAGS) $(CCE_HDF5_LIBS)
endif

# Test run targets (shared between CPU and GPU)
test: $(BUILD)/test_flat $(BUILD)/test_convergence $(BUILD)/test_amr_evolve $(BUILD)/test_maxwell
	@echo "=== Running tests ==="
	$(BUILD)/test_flat
	$(BUILD)/test_convergence
	$(BUILD)/test_amr_evolve
	$(BUILD)/test_maxwell

test-single-bh: $(BUILD)/test_single_bh
	@echo "=== Running single BH test ==="
	$(BUILD)/test_single_bh

test-convergence: $(BUILD)/test_convergence
	@echo "=== Running convergence test ==="
	$(BUILD)/test_convergence

test-constraints: $(BUILD)/test_constraints
	@echo "=== Running constraint tests ==="
	$(BUILD)/test_constraints

test-head-on: $(BUILD)/test_head_on
	@echo "=== Running head-on binary test ==="
	$(BUILD)/test_head_on

test-amr-mesh: $(BUILD)/test_amr_mesh
	@echo "=== Running AMR mesh test ==="
	$(BUILD)/test_amr_mesh

test-amr-ghost: $(BUILD)/test_amr_ghost
	@echo "=== Running AMR ghost exchange test ==="
	$(BUILD)/test_amr_ghost

test-amr-prolong: $(BUILD)/test_amr_prolong
	@echo "=== Running AMR prolongation + noise reduction test ==="
	$(BUILD)/test_amr_prolong

test-amr-refine: $(BUILD)/test_amr_refine
	@echo "=== Running AMR refinement + multi-level ghost test ==="
	$(BUILD)/test_amr_refine

test-amr-evolve: $(BUILD)/test_amr_evolve
	@echo "=== Running AMR evolution test ==="
	$(BUILD)/test_amr_evolve

test-pack-evolve: $(BUILD)/test_pack_evolve
	@echo "=== Running packed evolution test ==="
	$(BUILD)/test_pack_evolve

test-subcycle: $(BUILD)/test_subcycle
	@echo "=== Running subcycling test ==="
	$(BUILD)/test_subcycle

test-bowen-york: $(BUILD)/test_bowen_york
	@echo "=== Running Bowen-York test ==="
	$(BUILD)/test_bowen_york

test-hispid: $(BUILD)/test_hispid
	@echo "=== Running HiSpID test ==="
	$(BUILD)/test_hispid

test-maxwell: $(BUILD)/test_maxwell
	@echo "=== Running Maxwell test ==="
	$(BUILD)/test_maxwell

test-maxwell-debug: $(BUILD)/test_maxwell_debug
	@echo "=== Running Maxwell debug test ==="
	$(BUILD)/test_maxwell_debug

test-ah: $(BUILD)/test_ah_finder
	@echo "=== Running AH finder test ==="
	$(BUILD)/test_ah_finder

test-psi4: $(BUILD)/test_psi4
	@echo "=== Running Psi4 test ==="
	$(BUILD)/test_psi4

test-cce: $(BUILD)/test_cce_worldtube
	@echo "=== Running CCE worldtube test ==="
	$(BUILD)/test_cce_worldtube

test-cp-bc: $(BUILD)/test_cp_bc
	@echo "=== Running CP-BC test ==="
	$(BUILD)/test_cp_bc

test-solver-multiroot: $(BUILD)/test_solver_multiroot
	@echo "=== Running multi-root solver test ==="
	$(BUILD)/test_solver_multiroot

test-inspiral: $(BUILD)/test_binary_inspiral
	@echo "=== Running binary inspiral test ==="
	$(BUILD)/test_binary_inspiral

test-inspiral-smoke: $(BUILD)/test_inspiral_smoke
	@echo "=== Running inspiral smoke test ==="
	$(BUILD)/test_inspiral_smoke

test-inspiral-solver: $(BUILD)/test_inspiral_solver
	@echo "=== Running inspiral solver smoke test ==="
	$(BUILD)/test_inspiral_solver

test-inspiral-convergence: $(BUILD)/test_inspiral_convergence
	@echo "=== Running inspiral convergence test ==="
	$(BUILD)/test_inspiral_convergence

test-jfnk: $(BUILD)/test_jfnk
	@echo "=== Running JFNK solver test ==="
	$(BUILD)/test_jfnk

test-amr-accuracy: $(BUILD)/test_amr_accuracy
	@echo "=== Running AMR accuracy comparison ==="
	$(BUILD)/test_amr_accuracy

test-checkpoint: $(BUILD)/test_checkpoint
	@echo "=== Running checkpoint/restart test ==="
	$(BUILD)/test_checkpoint

test-gpu-debug: $(BUILD)/test_gpu_debug
	@echo "=== Running GPU kernel debug test ==="
	$(BUILD)/test_gpu_debug

test-deep-amr-nan: $(BUILD)/test_deep_amr_nan
	@echo "=== Running deep AMR NaN isolation test ==="
	$(BUILD)/test_deep_amr_nan

test-nbody-track: $(BUILD)/test_nbody_track
	@echo "=== Running N-body BH tracker test ==="
	$(BUILD)/test_nbody_track

test-gpu-tracker: $(BUILD)/test_gpu_tracker
	@echo "=== Running GPU vs CPU BH tracker test ==="
	$(BUILD)/test_gpu_tracker

test-qnm: $(BUILD)/test_qnm_ringdown
	@echo "=== Running Schwarzschild QNM ringdown test ==="
	$(BUILD)/test_qnm_ringdown

test-qnm-pub: $(BUILD)/test_qnm_publication
	@echo "=== Running publication-quality QNM ringdown test ==="
	$(BUILD)/test_qnm_publication

clean:
	rm -rf $(BUILD)
