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

# Platform detection: macOS vs Linux
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    CC ?= clang
else
    CC ?= gcc
endif

# Source files (common to all backends)
CORE_SRC    = src/core/grid.c
EVOLUTION_SRC = src/evolution/ccz4_rhs.c src/evolution/dissipation.c
NUMERICS_SRC  = src/numerics/rk4.c
INITIAL_SRC   = src/initial_data/puncture.c
DIAG_SRC      = src/diagnostics/constraints.c
BOUNDARY_SRC  = src/boundary/sommerfeld.c
IO_SRC        = src/io/output.c
MAIN_SRC      = src/main.c

# Backend selection
ifeq ($(BACKEND),cpu)
    BACKEND_SRC = src/backend/backend_cpu.c
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
    # GPU offloading via OpenMP target. Requires GCC 15+ for stack size control.
    # Set GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=16384 at runtime.
    # GPU_ARCH: nvptx-none (NVIDIA, default) or amdgcn-amdhsa (AMD)
    GPU_ARCH ?= nvptx-none
    BACKEND_FLAGS = -fopenmp -foffload=$(GPU_ARCH) -fcf-protection=none \
                    -fno-stack-protector \
                    -foffload-options="-lm -fno-stack-protector" -DLATTICE_GPU
    BACKEND_LIBS =
else
    $(error Unknown BACKEND=$(BACKEND). Use cpu or gpu)
endif

ALL_SRC = $(CORE_SRC) $(EVOLUTION_SRC) $(NUMERICS_SRC) $(INITIAL_SRC) \
          $(DIAG_SRC) $(BOUNDARY_SRC) $(IO_SRC) $(BACKEND_SRC)

# Compiler flags
INCLUDES = -I src
CFLAGS_BASE = -std=c17 -Wall -Wextra -Werror -D_GNU_SOURCE -Wno-unused-but-set-variable $(INCLUDES) $(BACKEND_FLAGS)
CFLAGS_OPT  = $(CFLAGS_BASE) -O3 -ffast-math -march=native
CFLAGS_DBG  = $(CFLAGS_BASE) -O0 -g -fsanitize=address,undefined -DDEBUG

LDFLAGS = $(BACKEND_LIBS) -lm

# Build directory
BUILD = build

# Targets
.PHONY: all debug test test-single-bh test-convergence test-constraints clean

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

test: $(BUILD)/test_flat $(BUILD)/test_convergence
	@echo "=== Running tests ==="
	$(BUILD)/test_flat
	$(BUILD)/test_convergence

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

clean:
	rm -rf $(BUILD)
