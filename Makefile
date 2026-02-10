# Lattice — 3D Numerical Relativity Simulator
#
# Usage:
#   make                  Optimized build with OpenMP parallelism
#   make PARALLEL=0       Build without OpenMP (single-threaded)
#   make debug            Debug build with sanitizers (no OMP)
#   make test-flat        Run flat spacetime test
#   make test-bh          Run single BH test
#   make test             Run all tests
#   make test-convergence Convergence verification
#   make clean            Remove build artifacts

BACKEND  ?= cpu
CC       ?= cc
PARALLEL ?= 1

CFLAGS_BASE = -std=c17 -Wall -Wextra -Werror

# Optimized flags
CFLAGS_OPT = $(CFLAGS_BASE) -O3 -ffast-math -march=native
# Debug flags (no OMP — sanitizers + OMP don't mix well)
CFLAGS_DBG = $(CFLAGS_BASE) -O0 -g -fsanitize=address,undefined

# Default to optimized
CFLAGS ?= $(CFLAGS_OPT)

LDFLAGS += -lm

# --- OpenMP / Parallel ---
ifeq ($(PARALLEL),1)
    CFLAGS += -DLATTICE_USE_OMP
    ifeq ($(shell uname),Darwin)
        # macOS: clang needs libomp from Homebrew
        ifneq ($(wildcard /opt/homebrew/opt/libomp),)
            CFLAGS  += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
            LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
        else ifneq ($(wildcard /usr/local/opt/libomp),)
            CFLAGS  += -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
            LDFLAGS += -L/usr/local/opt/libomp/lib -lomp
        else
            $(warning OpenMP requested but libomp not found. Install: brew install libomp)
            CFLAGS := $(filter-out -DLATTICE_USE_OMP,$(CFLAGS))
        endif
    else
        # Linux: GCC has built-in OpenMP
        CFLAGS  += -fopenmp
        LDFLAGS += -fopenmp
    endif
endif

# Source files
CORE_SRC    = src/core/grid.c
BACKEND_SRC = src/backend/backend_$(BACKEND).c
NUMERICS_SRC = src/numerics/rk4.c
BOUNDARY_SRC = src/boundary/sommerfeld.c
IO_SRC       = src/io/output.c
DIAG_SRC     = src/diagnostics/constraints.c
GEOM_SRC     = src/geometry/christoffel.c src/geometry/ricci.c
EVOL_SRC     = src/evolution/ccz4_rhs.c src/evolution/gauge_rhs.c \
               src/evolution/dissipation.c
MAIN_SRC     = src/main.c

ALL_SRC = $(CORE_SRC) $(BACKEND_SRC) $(NUMERICS_SRC) $(BOUNDARY_SRC) \
          $(IO_SRC) $(DIAG_SRC) $(GEOM_SRC) $(EVOL_SRC) $(MAIN_SRC)

# Object files
BUILDDIR = build
ALL_OBJ = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(ALL_SRC))

# Library objects (everything except main.c)
LIB_SRC = $(filter-out src/main.c,$(ALL_SRC))
LIB_OBJ = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(LIB_SRC))

# Binary names
BIN = lattice
TEST_FLAT_BIN = build/test_flat
TEST_BH_BIN   = build/test_bh

.PHONY: all debug test test-flat test-bh test-convergence clean

all: $(BIN)

$(BIN): $(ALL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Debug build (single-threaded, sanitizers)
debug: CFLAGS = $(CFLAGS_DBG)
debug: clean $(BIN)

# Compile rule
$(BUILDDIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Test binaries
$(TEST_FLAT_BIN): $(LIB_OBJ) $(BUILDDIR)/test_flat.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/test_flat.o: tests/test_flat.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

$(TEST_BH_BIN): $(LIB_OBJ) $(BUILDDIR)/test_single_bh.o $(BUILDDIR)/initial_data/puncture.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/test_single_bh.o: tests/test_single_bh.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Test runners
test-flat: $(TEST_FLAT_BIN)
	./$(TEST_FLAT_BIN)

test-bh: $(TEST_BH_BIN)
	./$(TEST_BH_BIN)

test: test-flat test-bh

test-convergence: $(TEST_FLAT_BIN)
	./tests/convergence.sh

clean:
	rm -rf $(BUILDDIR) $(BIN)
	rm -f scalars.dat
