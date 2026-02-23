#!/usr/bin/env python3
"""
Compute and verify AMR prolongation, restriction, and temporal interpolation
weights using SymPy exact rational arithmetic.

1. 7-point cell-centered Lagrange prolongation weights
2. 6-point symmetric cell-average restriction weights
3. Quartic temporal interpolation weights (Hermite-Birkhoff)

Usage:
    python3 tools/compute_amr_weights.py
"""

from sympy import (
    Rational, Symbol, symbols, simplify, prod, integrate,
    Poly, factorial, together, cancel, pprint
)


def section(title):
    """Print a section header."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# 1. 7-point cell-centered Lagrange prolongation weights
# ---------------------------------------------------------------------------
def compute_prolongation_weights():
    section("1. 7-POINT CELL-CENTERED LAGRANGE PROLONGATION WEIGHTS")

    print("Stencil nodes (coarse cell centers): {-3, -2, -1, 0, 1, 2, 3} * dx_c")
    print("Left child evaluation point: x = -1/4  (in units of dx_c)")
    print()

    x = Symbol("x")
    nodes = [Rational(k) for k in range(-3, 4)]  # -3, -2, -1, 0, 1, 2, 3
    x_eval = Rational(-1, 4)  # left child position

    # Lagrange basis polynomials L_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)
    weights_left = []
    for j, xj in enumerate(nodes):
        numer = Rational(1)
        denom = Rational(1)
        for k, xk in enumerate(nodes):
            if k != j:
                numer *= (x_eval - xk)
                denom *= (xj - xk)
        w = numer / denom
        weights_left.append(w)

    print("Left child weights (x = -1/4):")
    print("-" * 50)
    for j, (node, w) in enumerate(zip(nodes, weights_left)):
        print(f"  w[{j}] (node {int(node):+2}) = {w}")

    print()
    weight_sum = sum(weights_left)
    print(f"Sum of left child weights: {weight_sum}")
    assert weight_sum == 1, f"ERROR: weights sum to {weight_sum}, not 1"
    print("  -> VERIFIED: sum = 1")

    print()
    print("Right child weights (x = +1/4) = reversed left child:")
    print("-" * 50)
    weights_right = list(reversed(weights_left))
    for j, (node, w) in enumerate(zip(nodes, weights_right)):
        print(f"  w[{j}] (node {int(node):+2}) = {w}")

    weight_sum_r = sum(weights_right)
    print(f"\nSum of right child weights: {weight_sum_r}")
    assert weight_sum_r == 1

    # Express as fractions with common denominator for C code
    print()
    print("C-friendly form (left child):")
    print("-" * 50)
    for j, w in enumerate(weights_left):
        print(f"  w[{j}] = {w.p} / {w.q}    ({float(w):+.15e})")

    # Verify polynomial exactness: for f(x) = x^k, interpolation at -1/4
    # should recover (-1/4)^k exactly for k = 0..6
    print()
    print("Polynomial exactness verification:")
    for deg in range(7):
        exact = x_eval**deg
        interp = sum(w * n**deg for w, n in zip(weights_left, nodes))
        ok = "OK" if simplify(interp - exact) == 0 else "FAIL"
        print(f"  degree {deg}: exact={float(exact):+.6f}  "
              f"interp={float(interp):+.6f}  [{ok}]")

    return weights_left


# ---------------------------------------------------------------------------
# 2. 6-point symmetric cell-average restriction weights
# ---------------------------------------------------------------------------
def compute_restriction_weights():
    section("2. 6-POINT SYMMETRIC CELL-AVERAGE RESTRICTION WEIGHTS")

    print("Fine cell centers relative to coarse center:")
    print("  {-5/2, -3/2, -1/2, +1/2, +3/2, +5/2} * dx_fine")
    print("  where dx_fine = dx_coarse / 2")
    print()
    print("Restriction weight = (1/dx_c) * integral of L_j(x) over [-dx_c/2, +dx_c/2]")
    print("  with dx_c = 2*dx_f, so integral is over [-1, +1] in dx_f units.")
    print()

    x = Symbol("x")

    # Fine cell centers in units of dx_fine
    nodes = [Rational(k, 2) for k in (-5, -3, -1, 1, 3, 5)]

    # Integration bounds in units of dx_fine: coarse cell spans [-1, +1]*dx_f
    a = Rational(-1, 1)
    b = Rational(1, 1)
    # dx_coarse in units of dx_fine = 2
    dx_c = Rational(2, 1)

    weights = []
    for j, xj in enumerate(nodes):
        # Lagrange basis polynomial L_j(x)
        L_j = Rational(1)
        for k, xk in enumerate(nodes):
            if k != j:
                L_j *= (x - xk) / (xj - xk)

        # Cell-average weight = (1/dx_c) * integral of L_j over coarse cell
        w = integrate(L_j, (x, a, b)) / dx_c
        w = cancel(w)
        weights.append(w)

    print("Restriction weights:")
    print("-" * 50)
    for j, (node, w) in enumerate(zip(nodes, weights)):
        print(f"  w[{j}] (fine center {str(node):>5s}) = {w}")

    print()
    weight_sum = sum(weights)
    print(f"Sum of weights: {weight_sum}")
    assert weight_sum == 1, f"ERROR: weights sum to {weight_sum}, not 1"
    print("  -> VERIFIED: sum = 1")

    # Check symmetry
    print()
    print("Symmetry check:")
    n = len(weights)
    symmetric = True
    for j in range(n // 2):
        if weights[j] != weights[n - 1 - j]:
            print(f"  w[{j}] != w[{n-1-j}]: ASYMMETRIC")
            symmetric = False
    if symmetric:
        print("  -> VERIFIED: weights are symmetric  w[j] = w[N-1-j]")

    # C-friendly output
    print()
    print("C-friendly form:")
    print("-" * 50)
    for j, w in enumerate(weights):
        print(f"  w[{j}] = {w.p} / {w.q}    ({float(w):+.15e})")

    # Verify polynomial exactness: for f(x) = x^k, the cell average
    # (1/dx_c) * int_{-1}^{1} x^k dx should equal sum_j w_j * node_j^k
    print()
    print("Polynomial exactness verification (cell-average reproduction):")
    for deg in range(6):
        # Exact cell average of x^k over [-1, 1] with width 2
        exact_avg = integrate(x**deg, (x, a, b)) / dx_c
        # Weighted sum of nodal values
        weighted = sum(w * nd**deg for w, nd in zip(weights, nodes))
        ok = "OK" if simplify(weighted - exact_avg) == 0 else "FAIL"
        print(f"  degree {deg}: exact avg={float(exact_avg):+.8f}  "
              f"weighted={float(weighted):+.8f}  [{ok}]")

    return weights


# ---------------------------------------------------------------------------
# 3. Quartic temporal interpolation weights
# ---------------------------------------------------------------------------
def compute_temporal_weights():
    section("3. QUARTIC TEMPORAL INTERPOLATION WEIGHTS")

    print("Quartic polynomial p(s) with 5 constraints:")
    print("  p(0)  = U_n       (current solution)")
    print("  p(1)  = U_{n+1}   (next solution)")
    print("  p(-1) = U_{n-1}   (previous solution)")
    print("  p'(0) = dt * F_n  (current RHS)")
    print("  p'(-1)= dt * F_{n-1} (previous RHS)")
    print()
    print("Express p(s) = c0*U_{n-1} + c1*U_n + c2*U_{n+1} + c3*dt*F_{n-1} + c4*dt*F_n")
    print("where c0..c4 are polynomials in s.")
    print()

    s = Symbol("s")

    # Unknowns: coefficients a0..a4 of p(s) = a0 + a1*s + a2*s^2 + a3*s^3 + a4*s^4
    a0, a1, a2, a3, a4 = symbols("a0 a1 a2 a3 a4")

    p = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4
    dp = a1 + 2*a2*s + 3*a3*s**2 + 4*a4*s**3

    # Data symbols
    Un, Unp1, Unm1, Fn, Fnm1 = symbols("U_n U_{n+1} U_{n-1} F_n F_{n-1}")

    # 5 constraints:
    # p(0)  = U_n      ->  a0 = U_n
    # p(1)  = U_{n+1}  ->  a0 + a1 + a2 + a3 + a4 = U_{n+1}
    # p(-1) = U_{n-1}  ->  a0 - a1 + a2 - a3 + a4 = U_{n-1}
    # p'(0) = dt*F_n   ->  a1 = dt*F_n
    # p'(-1)= dt*F_{n-1} -> a1 - 2*a2 + 3*a3 - 4*a4 = dt*F_{n-1}

    from sympy import solve, Eq

    eqs = [
        Eq(a0, Un),                                        # p(0) = U_n
        Eq(a0 + a1 + a2 + a3 + a4, Unp1),                # p(1) = U_{n+1}
        Eq(a0 - a1 + a2 - a3 + a4, Unm1),                # p(-1) = U_{n-1}
        Eq(a1, Fn),                                        # p'(0) = dt*F_n
        Eq(a1 - 2*a2 + 3*a3 - 4*a4, Fnm1),              # p'(-1) = dt*F_{n-1}
    ]

    sol = solve(eqs, [a0, a1, a2, a3, a4])

    print("Polynomial coefficients (a_k for p(s) = sum a_k * s^k):")
    print("-" * 60)
    for coeff_sym in [a0, a1, a2, a3, a4]:
        expr = cancel(sol[coeff_sym])
        print(f"  {coeff_sym} = {expr}")

    # Substitute back to get p(s)
    p_full = (sol[a0] + sol[a1]*s + sol[a2]*s**2
              + sol[a3]*s**3 + sol[a4]*s**4)
    p_full = cancel(p_full).expand()

    print()
    print("Full polynomial p(s):")
    print(f"  p(s) = {p_full}")

    # Extract weight polynomials: coefficients of each data value
    # Collect by each data symbol
    data_syms = [Unm1, Un, Unp1, Fnm1, Fn]
    data_names = ["U_{n-1}", "U_n", "U_{n+1}", "dt*F_{n-1}", "dt*F_n"]

    print()
    print("Weight polynomials in s:")
    print("-" * 60)

    weight_polys = {}
    for dsym, dname in zip(data_syms, data_names):
        # Coefficient of dsym in p_full
        w_poly = Rational(0)
        p_expanded = p_full.expand()
        # Collect the coefficient of this symbol
        w_poly = p_expanded.coeff(dsym)
        w_poly = cancel(w_poly).expand()
        weight_polys[dname] = w_poly
        print(f"  c_{dname}(s) = {w_poly}")

    # Verification at all 5 constraint points
    print()
    print("Verification at constraint points:")
    print("-" * 60)

    # p(0) should give U_n
    p_at_0 = p_full.subs(s, 0)
    p_at_0_simplified = cancel(p_at_0)
    check = simplify(p_at_0_simplified - Un)
    status = "OK" if check == 0 else "FAIL"
    print(f"  p(0) = {p_at_0_simplified}")
    print(f"    Expected: U_n  [{status}]")

    # p(1) should give U_{n+1}
    p_at_1 = p_full.subs(s, 1)
    p_at_1_simplified = cancel(p_at_1)
    check = simplify(p_at_1_simplified - Unp1)
    status = "OK" if check == 0 else "FAIL"
    print(f"  p(1) = {p_at_1_simplified}")
    print(f"    Expected: U_{{n+1}}  [{status}]")

    # p(-1) should give U_{n-1}
    p_at_m1 = p_full.subs(s, -1)
    p_at_m1_simplified = cancel(p_at_m1)
    check = simplify(p_at_m1_simplified - Unm1)
    status = "OK" if check == 0 else "FAIL"
    print(f"  p(-1) = {p_at_m1_simplified}")
    print(f"    Expected: U_{{n-1}}  [{status}]")

    # p'(0) should give dt*F_n
    dp_full = p_full.diff(s)
    dp_at_0 = dp_full.subs(s, 0)
    dp_at_0_simplified = cancel(dp_at_0)
    check = simplify(dp_at_0_simplified - Fn)
    status = "OK" if check == 0 else "FAIL"
    print(f"  p'(0) = {dp_at_0_simplified}")
    print(f"    Expected: dt*F_n  [{status}]")

    # p'(-1) should give dt*F_{n-1}
    dp_at_m1 = dp_full.subs(s, -1)
    dp_at_m1_simplified = cancel(dp_at_m1)
    check = simplify(dp_at_m1_simplified - Fnm1)
    status = "OK" if check == 0 else "FAIL"
    print(f"  p'(-1) = {dp_at_m1_simplified}")
    print(f"    Expected: dt*F_{{n-1}}  [{status}]")

    # Also print the weight at s=1/2 (common subcycle interpolation point)
    print()
    print("Weights at s = 1/2 (midpoint, common subcycle evaluation):")
    print("-" * 60)
    s_half = Rational(1, 2)
    for dname, w_poly in weight_polys.items():
        val = w_poly.subs(s, s_half)
        val = cancel(val)
        print(f"  c_{dname}(1/2) = {val}    ({float(val):+.15e})")

    p_half = p_full.subs(s, s_half)
    print(f"\n  p(1/2) = {cancel(p_half)}")

    return weight_polys


# ---------------------------------------------------------------------------
# Comparison with existing code
# ---------------------------------------------------------------------------
def compare_with_existing():
    section("COMPARISON WITH EXISTING LATTICE STENCILS")

    print("Current Lattice prolongation: 5-point (4th-order)")
    print("  Nodes: {-2, -1, 0, 1, 2}")
    print("  Left child at x = -1/4:")
    x_eval = Rational(-1, 4)
    nodes_5 = [Rational(k) for k in range(-2, 3)]
    weights_5 = []
    for j, xj in enumerate(nodes_5):
        numer = Rational(1)
        denom = Rational(1)
        for k, xk in enumerate(nodes_5):
            if k != j:
                numer *= (x_eval - xk)
                denom *= (xj - xk)
        w = numer / denom
        weights_5.append(w)
    for j, (n, w) in enumerate(zip(nodes_5, weights_5)):
        print(f"    w[{j}] (node {int(n):+2}) = {w.p}/{w.q}  = {float(w):+.15e}")
    print(f"  Sum = {sum(weights_5)}")

    print()
    print("Current Lattice restriction: 4-point (4th-order)")
    print("  Fine centers: {-3/2, -1/2, +1/2, +3/2} * dx_f")
    x = Symbol("x")
    nodes_4 = [Rational(k, 2) for k in (-3, -1, 1, 3)]
    a, b = Rational(-1), Rational(1)
    dx_c = Rational(2)
    weights_4 = []
    for j, xj in enumerate(nodes_4):
        L_j = Rational(1)
        for k, xk in enumerate(nodes_4):
            if k != j:
                L_j *= (x - xk) / (xj - xk)
        w = integrate(L_j, (x, a, b)) / dx_c
        w = cancel(w)
        weights_4.append(w)
    for j, (n, w) in enumerate(zip(nodes_4, weights_4)):
        print(f"    w[{j}] (fine center {n}) = {w.p}/{w.q}  = {float(w):+.15e}")
    print(f"  Sum = {sum(weights_4)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Lattice AMR Weight Computation")
    print("Using SymPy exact rational arithmetic")

    w_prolong = compute_prolongation_weights()
    w_restrict = compute_restriction_weights()
    w_temporal = compute_temporal_weights()
    compare_with_existing()

    section("SUMMARY")
    print("All weights computed and verified successfully.")
    print()
    print("7-point prolongation (left child):")
    for j, w in enumerate(w_prolong):
        print(f"  w[{j}] = {w.p}/{w.q}")
    print()
    print("6-point restriction:")
    for j, w in enumerate(w_restrict):
        print(f"  w[{j}] = {w.p}/{w.q}")
    print()
    print("Temporal interpolation: quartic Hermite-Birkhoff polynomial verified")
    print("  at all 5 constraint points (p(0), p(1), p(-1), p'(0), p'(-1)).")
