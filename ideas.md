# Ideas

## GPU-accelerated multigrid solver

The FAS multigrid constraint solver (relaxation.c, relaxation_amr.c) currently
runs on CPU. It uses 8-color Gauss-Seidel smoothing which is inherently
parallelizable — each color can be updated independently across all grid points.

### What would need GPU porting:
- Gauss-Seidel relaxation sweeps (8-color parallel)
- Restriction operator (fine → coarse)
- Prolongation operator (coarse → fine)
- Residual computation and L2 norm reduction
- Ghost exchange between solver iterations

### Current impact:
- Solver runs once at initialization (~2 minutes on CPU)
- Evolution is 1400+ steps (~2-3 hours on GPU)
- Solver is <2% of total runtime for production runs

### When it matters:
- N-body initial data with many punctures (N=10+) where solver cost grows
- Higher resolution grids where solver iterations become expensive
- Repeated solves (e.g., constraint re-solving during evolution)
- Interactive/exploratory runs where startup latency matters
