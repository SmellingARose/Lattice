#!/usr/bin/env python3
"""
Parameter sweep for level-dependent per-field sigma.

Generates Latin Hypercube samples over 4 sigma parameters and runs them
in parallel. Ranks results by fitness (higher = better; crashed runs
are penalized).

Usage:
  python3 tools/sigma_sweep/sweep.py [N_SAMPLES] [N_PARALLEL]

Default: 64 samples, 28 parallel jobs (all cores).

Output: CSV of sorted results printed to stdout, plus sweep_results.csv
"""
import subprocess
import multiprocessing as mp
import sys
import os
import random

# Parameter ranges (min, max)
RANGES = {
    "sg_c": (0.30, 1.00),  # gauge coarse
    "sg_f": (0.10, 0.99),  # gauge fine
    "sp_c": (0.05, 0.80),  # phys coarse
    "sp_f": (0.05, 0.50),  # phys fine
}

# Seed configurations we want to test regardless of LHS (hand-picked for context)
SEED_CONFIGS = [
    # (sg_c,  sg_f,  sp_c,  sp_f)   # label
    (0.50,  0.50,  0.50,  0.50),   # uniform 0.5 (AthenaK)
    (1.00,  1.00,  1.00,  1.00),   # uniform 1.0 (GRChombo)
    (0.99,  0.99,  0.30,  0.30),   # Etienne per-field (no level-dep)
    (0.50,  0.10,  0.50,  0.10),   # BAM-ish (level-dep only, no per-field)
    (0.99,  0.99,  0.50,  0.10),   # combined: per-field + level-dep
    (0.99,  0.50,  0.30,  0.10),   # asymmetric: gauge fades, phys fades
]


def latin_hypercube(n_samples, ranges):
    """Simple LHS: divide each dim into n intervals, sample one from each
    interval, then permute across dimensions to decorrelate."""
    rng = random.Random(42)
    samples = []
    cols = []
    for key, (lo, hi) in ranges.items():
        # n uniformly-spaced intervals
        bins = [lo + (hi - lo) * (i + rng.random()) / n_samples
                for i in range(n_samples)]
        rng.shuffle(bins)
        cols.append(bins)
    for i in range(n_samples):
        samples.append([cols[d][i] for d in range(len(ranges))])
    return samples


def run_config(config):
    sg_c, sg_f, sp_c, sp_f = config
    try:
        result = subprocess.run(
            ["./tools/sigma_sweep/run_one.sh",
             f"{sg_c:.4f}", f"{sg_f:.4f}", f"{sp_c:.4f}", f"{sp_f:.4f}"],
            capture_output=True, text=True, timeout=180)
        line = result.stdout.strip()
        if not line:
            return (sg_c, sg_f, sp_c, sp_f, -9999, "FAIL", 0)
        parts = line.split()
        return (float(parts[0]), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]), parts[5], int(parts[6]))
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return (sg_c, sg_f, sp_c, sp_f, -9999, "ERR", 0)


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    n_parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 28

    # Combine LHS samples with seed configs
    lhs = latin_hypercube(n_samples, RANGES)
    all_configs = SEED_CONFIGS + [tuple(x) for x in lhs]

    print(f"# Running {len(all_configs)} configs, {n_parallel} parallel", file=sys.stderr)
    print(f"# {len(SEED_CONFIGS)} seed + {n_samples} LHS", file=sys.stderr)

    import time
    start = time.time()

    with mp.Pool(n_parallel) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_config, all_configs)):
            results.append(r)
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(all_configs)}] {elapsed:5.1f}s  fit={r[4]:8.3f}  "
                  f"ham={r[5]:>14s}  step={r[6]}  "
                  f"({r[0]:.2f},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f})",
                  file=sys.stderr)

    # Sort by fitness (descending - higher is better)
    results.sort(key=lambda r: -r[4])

    # Write results
    with open("sweep_results.csv", "w") as f:
        f.write("sg_c,sg_f,sp_c,sp_f,fitness,ham,last_step\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n")

    # Print top 15
    print("\n# Top 15 (sorted by fitness)")
    print("# sg_c   sg_f   sp_c   sp_f   fitness   ham              step")
    for r in results[:15]:
        print(f"  {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  "
              f"{r[4]:8.3f}  {r[5]:>14s}  {r[6]}")

    total = time.time() - start
    print(f"\n# Total wall time: {total:.1f}s", file=sys.stderr)
    print(f"# Results written to sweep_results.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
