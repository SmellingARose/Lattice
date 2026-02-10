#!/bin/bash
#
# convergence.sh — Verify 4th-order convergence of flat spacetime evolution
#
# Runs test_flat at 3 resolutions (N=16, 32, 64) and checks that the error
# ratio E(h)/E(h/2) ≈ 2^4 = 16 (4th-order convergence).
#
# Usage: ./tests/convergence.sh

set -e

TEST_BIN="./build/test_flat"
STEPS=100  # Fewer steps for convergence test (speed)

if [ ! -x "$TEST_BIN" ]; then
    echo "Building test_flat..."
    make build/test_flat
fi

echo "=== Convergence Test ==="
echo ""

# Run at 3 resolutions
declare -a ERRORS

for N in 16 32 64; do
    echo "Running N=${N}..."
    OUTPUT=$($TEST_BIN --nx $N --steps $STEPS 2>&1)
    HAM=$(echo "$OUTPUT" | grep "ham_l2" | head -1 | awk '{print $3}')
    echo "  N=${N}: ham_l2 = ${HAM}"
    ERRORS+=("$HAM")
done

E1=${ERRORS[0]}
E2=${ERRORS[1]}
E3=${ERRORS[2]}

echo ""

# For flat spacetime, errors should be exactly 0 (or machine precision).
# Convergence order is computed from: order = log2(E1/E2) / log2(h1/h2) = log2(E1/E2)
# Since h1/h2 = 2 when N doubles.
#
# If all errors are 0 (flat spacetime is exact), we just report that.

if [ "$E1" = "0.000000e+00" ] && [ "$E2" = "0.000000e+00" ] && [ "$E3" = "0.000000e+00" ]; then
    echo "All errors are exactly zero — flat spacetime preserved to machine precision."
    echo "Convergence order: EXACT (∞)"
    echo ""
    echo "PASS: convergence test"
    exit 0
fi

# Compute convergence order using awk
ORDER=$(echo "$E1 $E2" | awk '{
    if ($2 > 0 && $1 > 0) {
        printf "%.2f", log($1/$2)/log(2.0)
    } else {
        printf "inf"
    }
}')

RATIO=$(echo "$E1 $E2" | awk '{
    if ($2 > 0) printf "%.2f", $1/$2
    else printf "inf"
}')

echo "Error ratio E(h)/E(h/2) = ${RATIO} (expect ~16 for 4th order)"
echo "Convergence order = ${ORDER} (expect ~4.0)"
echo ""

# Check if order is reasonable (> 3.5)
PASS=$(echo "$ORDER" | awk '{ print ($1 > 3.5 || $1 == "inf") ? 1 : 0 }')

if [ "$PASS" = "1" ]; then
    echo "PASS: convergence test"
    exit 0
else
    echo "FAIL: convergence order ${ORDER} < 3.5"
    exit 1
fi
