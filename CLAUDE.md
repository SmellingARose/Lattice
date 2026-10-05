# Lattice — ADM Numerical Relativity, Built From Scratch

## What this is

A rewrite from zero. The previous CCZ4/AMR/GPU code was deleted; it lives in
git history (last commit before the rewrite: `5c98407`). Do not restore or copy
from it.

Goal: evolve Einstein's equations in the plain ADM (3+1) formulation on a
uniform 3D grid in C, with the owner writing every line and understanding every
equation. ADM is a stepping stone: once it works and we have watched it fail on
a black hole, we reformulate to BSSN.

## How we work (teaching mode) — follow strictly

1. **Claude shows the math first.** Derive or state each equation before any
   code is written. Define every symbol. Cite sources (see References).
2. **The owner writes the code.** Claude does not write implementation code
   for the owner, not even "just a starting sketch", unless explicitly asked.
3. **Claude reviews, then commits.** Check the submitted code against the math
   first. Point out bugs with the line and the reason, and let the owner fix
   them. Don't silently correct anything. Once it is right, Claude puts it in
   the repo, builds, tests, and commits.
4. **One step at a time.** Don't jump ahead to the next topic until the
   current step compiles, passes its test, and the owner says to move on.
5. Plain notation in chat (unicode / ASCII math). Keep explanations tight; the
   owner is technically strong but rusty on GR.

## Roadmap

0. GR refresher: metric, Christoffel symbols, covariant derivative, Riemann/Ricci
1. 3+1 split: foliation, lapse α, shift β^i, spatial metric γ_ij, extrinsic curvature K_ij
2. ADM evolution equations + Hamiltonian/momentum constraints
3. Discretization: uniform 3D grid, finite differences, time integrator
4. Tests: flat spacetime → gauge wave → linearized wave → Schwarzschild (expect blow-up)
5. Why ADM fails (weak hyperbolicity) → BSSN

## Decisions so far

- Language: C. 3D from day one, uniform grid, no AMR, no GPU.
- Tests that are effectively 1D (e.g. gauge wave along x) run on the 3D code.

Record every new decision (memory layout, FD order, integrator, etc.) here
when it is made.

## References

- Baumgarte & Shapiro, *Numerical Relativity* (2010), Ch. 2–4 — 3+1 and ADM
- Alcubierre, *Introduction to 3+1 Numerical Relativity* (2008), Ch. 2
- Gourgoulhon, *3+1 Formalism in General Relativity*, arXiv:gr-qc/0703035
- Alcubierre et al., "Towards standard testbeds for numerical relativity"
  (Apples-with-Apples), arXiv:gr-qc/0305023 — gauge wave / linear wave tests
