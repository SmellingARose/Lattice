# Lattice — 3D Numerical Relativity Simulator
# Build system
#
# Usage:
#   make                    # optimized CPU build (OpenMP threads)
#   make BACKEND=gpu        # GPU build (OpenMP target, requires GCC 15+)
#   make BACKEND=gpu GPU_ARCH=amdgcn-amdhsa  # AMD GPU target
#   make debug              # debug build with sanitizers
#   make test               # run all tests
#   make clean
#
# GPU runtime: export GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=16384

BACKEND ?= cpu
FD_ORDER ?= 6
EM ?= off

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
INITIAL_SRC   = src/initial_data/puncture.c src/initial_data/bowen_york.c src/initial_data/relaxation.c src/initial_data/relaxation_amr.c src/initial_data/kerr_quasi_isotropic.c
DIAG_SRC      = src/diagnostics/constraints.c src/diagnostics/ah_finder.c
BOUNDARY_SRC  = src/boundary/sommerfeld.c
IO_SRC        = src/io/output.c
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
    BACKEND_SRC = src/backend/backend_gpu.c
    # GPU offloading via OpenMP target. Requires GCC 14+ for nvptx.
    # Set GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=16384 at runtime.
    # GPU_ARCH: nvptx-none (NVIDIA, default) or amdgcn-amdhsa (AMD)
    #
    # Host -O2 required: GCC -O3 generates broken nvptx IR for large
    # inlined functions (ccz4_rhs_point ~3KB of locals). Offload-side
    # -O3 is fine — the bug is in host-side IR generation. -O3 offload
    # via -foffload-options keeps GPU kernel optimization aggressive.
    GPU_ARCH ?= nvptx-none
    LTO_FLAGS =
    GPU_OPT = -O2
    BACKEND_FLAGS = -fopenmp -foffload=$(GPU_ARCH) -fcf-protection=none \
                    -fno-stack-protector \
                    -foffload-options="-lm -fno-stack-protector -O3" -DLATTICE_GPU
    BACKEND_LIBS =
else
    $(error Unknown BACKEND=$(BACKEND). Use cpu or gpu)
endif

ALL_SRC = $(CORE_SRC) $(EVOLUTION_SRC) $(NUMERICS_SRC) $(INITIAL_SRC) \
          $(DIAG_SRC) $(BOUNDARY_SRC) $(IO_SRC) $(AMR_SRC) $(BACKEND_SRC)

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
CFLAGS_BASE = -std=c17 -Wall -Wextra -Werror -D_GNU_SOURCE -Wno-unused-but-set-variable -DFD_ORDER=$(FD_ORDER) $(EM_FLAGS) $(INCLUDES) $(BACKEND_FLAGS)
HOST_OPT ?= -O3
ifeq ($(BACKEND),gpu)
    HOST_OPT = $(GPU_OPT)
endif
CFLAGS_OPT  = $(CFLAGS_BASE) $(HOST_OPT) -ffast-math -march=native $(LTO_FLAGS)
CFLAGS_DBG  = $(CFLAGS_BASE) -O0 -g -fsanitize=address,undefined -DDEBUG

LDFLAGS = $(BACKEND_LIBS) -lm $(LTO_FLAGS)

# Build directory
BUILD = build

# Targets
.PHONY: all debug test test-single-bh test-convergence test-constraints test-head-on test-amr-mesh test-amr-ghost test-amr-prolong test-amr-refine test-amr-evolve test-pack-evolve test-subcycle test-bowen-york test-hispid test-maxwell test-ah test-inspiral-convergence test-relaxation-amr clean

all: $(BUILD)/lattice

$(BUILD)/lattice: $(ALL_SRC) $(MAIN_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ $(ALL_SRC) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built: $@"

debug: $(BUILD)/lattice_debug

$(BUILD)/lattice_debug: $(ALL_SRC) $(MAIN_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_DBG) -o $@ $(ALL_SRC) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built: $@ (debug)"

# Tests
$(BUILD)/test_flat: tests/test_flat.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_flat.c $(ALL_SRC) $(LDFLAGS)

$(BUILD)/test_single_bh: tests/test_single_bh.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_single_bh.c $(ALL_SRC) $(LDFLAGS)

test: $(BUILD)/test_flat $(BUILD)/test_convergence $(BUILD)/test_amr_evolve $(BUILD)/test_maxwell
	@echo "=== Running tests ==="
	$(BUILD)/test_flat
	$(BUILD)/test_convergence
	$(BUILD)/test_amr_evolve
	$(BUILD)/test_maxwell

test-single-bh: $(BUILD)/test_single_bh
	@echo "=== Running single BH test ==="
	$(BUILD)/test_single_bh

$(BUILD)/test_convergence: tests/test_convergence.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_convergence.c $(ALL_SRC) $(LDFLAGS)

test-convergence: $(BUILD)/test_convergence
	@echo "=== Running convergence test ==="
	$(BUILD)/test_convergence

$(BUILD)/test_constraints: tests/test_constraints.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_constraints.c $(ALL_SRC) $(LDFLAGS)

test-constraints: $(BUILD)/test_constraints
	@echo "=== Running constraint tests ==="
	$(BUILD)/test_constraints

$(BUILD)/test_head_on: tests/test_head_on.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_head_on.c $(ALL_SRC) $(LDFLAGS)

test-head-on: $(BUILD)/test_head_on
	@echo "=== Running head-on binary test ==="
	$(BUILD)/test_head_on

$(BUILD)/test_amr_mesh: tests/test_amr_mesh.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_amr_mesh.c $(ALL_SRC) $(LDFLAGS)

test-amr-mesh: $(BUILD)/test_amr_mesh
	@echo "=== Running AMR mesh test ==="
	$(BUILD)/test_amr_mesh

$(BUILD)/test_amr_ghost: tests/test_amr_ghost.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_amr_ghost.c $(ALL_SRC) $(LDFLAGS)

test-amr-ghost: $(BUILD)/test_amr_ghost
	@echo "=== Running AMR ghost exchange test ==="
	$(BUILD)/test_amr_ghost

$(BUILD)/test_amr_prolong: tests/test_amr_prolong.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_amr_prolong.c $(ALL_SRC) $(LDFLAGS)

test-amr-prolong: $(BUILD)/test_amr_prolong
	@echo "=== Running AMR prolongation + noise reduction test ==="
	$(BUILD)/test_amr_prolong

$(BUILD)/test_amr_refine: tests/test_amr_refine.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_amr_refine.c $(ALL_SRC) $(LDFLAGS)

test-amr-refine: $(BUILD)/test_amr_refine
	@echo "=== Running AMR refinement + multi-level ghost test ==="
	$(BUILD)/test_amr_refine

$(BUILD)/test_amr_evolve: tests/test_amr_evolve.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_amr_evolve.c $(ALL_SRC) $(LDFLAGS)

test-amr-evolve: $(BUILD)/test_amr_evolve
	@echo "=== Running AMR evolution test ==="
	$(BUILD)/test_amr_evolve

$(BUILD)/test_pack_evolve: tests/test_pack_evolve.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_pack_evolve.c $(ALL_SRC) $(LDFLAGS)

test-pack-evolve: $(BUILD)/test_pack_evolve
	@echo "=== Running packed evolution test ==="
	$(BUILD)/test_pack_evolve

$(BUILD)/test_subcycle: tests/test_subcycle.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_subcycle.c $(ALL_SRC) $(LDFLAGS)

test-subcycle: $(BUILD)/test_subcycle
	@echo "=== Running subcycling test ==="
	$(BUILD)/test_subcycle

$(BUILD)/test_bowen_york: tests/test_bowen_york.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_bowen_york.c $(ALL_SRC) $(LDFLAGS)

test-bowen-york: $(BUILD)/test_bowen_york
	@echo "=== Running Bowen-York test ==="
	$(BUILD)/test_bowen_york

$(BUILD)/test_hispid: tests/test_hispid.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_hispid.c $(ALL_SRC) $(LDFLAGS)

test-hispid: $(BUILD)/test_hispid
	@echo "=== Running HiSpID test ==="
	$(BUILD)/test_hispid

$(BUILD)/test_maxwell: tests/test_maxwell.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -DLATTICE_EM_ENABLED -o $@ tests/test_maxwell.c $(ALL_SRC) $(LDFLAGS)

test-maxwell: $(BUILD)/test_maxwell
	@echo "=== Running Maxwell test ==="
	$(BUILD)/test_maxwell

$(BUILD)/test_maxwell_debug: tests/test_maxwell_debug.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -DLATTICE_EM_ENABLED -o $@ tests/test_maxwell_debug.c $(ALL_SRC) $(LDFLAGS)

test-maxwell-debug: $(BUILD)/test_maxwell_debug
	@echo "=== Running Maxwell debug test ==="
	$(BUILD)/test_maxwell_debug

$(BUILD)/test_ah_finder: tests/test_ah_finder.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_ah_finder.c $(ALL_SRC) $(LDFLAGS)

test-ah: $(BUILD)/test_ah_finder
	@echo "=== Running AH finder test ==="
	$(BUILD)/test_ah_finder

$(BUILD)/test_inspiral_convergence: tests/test_inspiral_convergence.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_inspiral_convergence.c $(ALL_SRC) $(LDFLAGS)

test-inspiral-convergence: $(BUILD)/test_inspiral_convergence
	@echo "=== Running inspiral convergence test ==="
	$(BUILD)/test_inspiral_convergence

$(BUILD)/test_relaxation_amr: tests/test_relaxation_amr.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_relaxation_amr.c $(ALL_SRC) $(LDFLAGS)

test-relaxation-amr: $(BUILD)/test_relaxation_amr
	@echo "=== Running AMR relaxation solver test ==="
	$(BUILD)/test_relaxation_amr

$(BUILD)/test_amr_accuracy: tests/test_amr_accuracy.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_amr_accuracy.c $(ALL_SRC) $(LDFLAGS)

test-amr-accuracy: $(BUILD)/test_amr_accuracy
	@echo "=== Running AMR accuracy comparison ==="
	$(BUILD)/test_amr_accuracy

$(BUILD)/test_gpu_debug: tests/test_gpu_debug.c $(ALL_SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_OPT) -o $@ tests/test_gpu_debug.c $(ALL_SRC) $(LDFLAGS)

test-gpu-debug: $(BUILD)/test_gpu_debug
	@echo "=== Running GPU kernel debug test ==="
	CUDA_VISIBLE_DEVICES=0 GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=65536 $(BUILD)/test_gpu_debug

clean:
	rm -rf $(BUILD)
