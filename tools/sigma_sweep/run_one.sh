#!/usr/bin/env bash
# Run a single sigma sweep configuration.
# Args: sg_c sg_f sp_c sp_f
# Outputs: "<sg_c> <sg_f> <sp_c> <sp_f> <fitness> <last_ham> <last_step>"
set -e

sg_c=$1
sg_f=$2
sp_c=$3
sp_f=$4

LATTICE=${LATTICE:-./build/lattice}

output=$(OMP_NUM_THREADS=1 $LATTICE \
    --amr --L 16 --N_block 16 --max_level 1 \
    --refine-c 1.5 --refine-beta 2.0 \
    --steps 20 --regrid_every 0 --amr-levels 1 \
    --puncture 1.0,0,0,0 \
    --level_dep_sigma \
    --sigma_gauge_coarse $sg_c --sigma_gauge_fine $sg_f \
    --sigma_phys_coarse $sp_c --sigma_phys_fine $sp_f \
    2>&1)

# Find the last "step N ... Ham L2=VALUE" line
last_line=$(echo "$output" | grep -E "^  step" | tail -1)
last_step=$(echo "$last_line" | awk '{print $2}')
last_ham=$(echo "$last_line" | grep -oE "Ham L2=[^ ]+" | sed 's/Ham L2=//')

if [[ -z "$last_ham" || "$last_ham" == "-nan" || "$last_ham" == "nan" ]]; then
    fitness=$(awk -v s=$last_step 'BEGIN { printf "%.6f", -1000 + s }')
    last_ham="NaN"
else
    fitness=$(awk -v h=$last_ham 'BEGIN { printf "%.6f", -log(h) / log(10) }')
fi

printf "%s %s %s %s %s %s %s\n" "$sg_c" "$sg_f" "$sp_c" "$sp_f" "$fitness" "$last_ham" "$last_step"
