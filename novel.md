# Novel Contributions

Things Lattice does differently from existing NR codes (BAM, GRChombo,
Einstein Toolkit, SpEC, Lean, etc.).

---

## 1. Equidistribution-Optimal AMR Refinement Ratio (β = 1.516)

**What every other code does:** Halve the refinement radius at each level
(β = 2). Level k covers a region of radius r/2^k. This is the universal
default in BAM, GRChombo, Einstein Toolkit, and every AMR NR code we've
found. Nobody derives it — it's just "the obvious choice."

**What we do:** Use β = 2^(3/5) ≈ 1.516 instead of 2.

**Why it's better:** β = 2 is only optimal if you're resolving a *constant*
field. But the fields near a black hole aren't constant — Riemann curvature
and extrinsic curvature fall off as 1/r³. The truncation error of a p-th
order finite difference scheme on a 1/r^α field at a level boundary is:

    ε ~ dx^p / r^(α+p+1)

At an AMR level boundary, dx halves (dx → dx/2) and the radius changes by β
(r → r/β). For equal error at every boundary:

    (dx/2)^p / (r/β)^(α+p+1) = dx^p / r^(α+p+1)

Solving: β = 2^(p/(α+p+1))

For 6th-order FD (p=6) on 1/r³ curvature (α=3):

    β = 2^(6/10) = 2^(3/5) ≈ 1.516

**Concrete effect:** Compared to standard halving:
- Fine levels get *larger* radii → better coverage near the puncture
- Coarse levels get *smaller* radii → fewer blocks, less wasted work
- For the D10 inspiral: finest level covers 1.94M (vs 0.375M with β=2)
  and coarsest level covers 124M (vs 384M with β=2)

**Per-puncture mass scaling:** The radius is r_k = 4·M·β^k, where M is each
puncture's individual mass. A 10M black hole gets 10x larger refinement regions
than a 1M black hole. Standard codes use the same radius for all punctures.

**Status:** Implemented and tested. No other published NR code uses an
optimized refinement ratio.

Full derivation: [docs/amr_refinement_ratio.html](docs/amr_refinement_ratio.html)
How it works: [docs/amr_refinement_howto.html](docs/amr_refinement_howto.html)
Code: `src/initial_data/relaxation_amr.c`, `refine_mesh_near_punctures()`
