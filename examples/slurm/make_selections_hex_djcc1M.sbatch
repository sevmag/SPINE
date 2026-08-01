#!/bin/bash
# Build the SPINE reproduction selections for the hexagon djcc-1M pretraining
# split (val 4000 / train 500k, seed-0 shuffle, filtered to SPINE-splittable).
#SBATCH --job-name=spine_mksel
#SBATCH --partition=shared
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --output=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/mksel_%j.out
#SBATCH --error=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/mksel_%j.err
set -euo pipefail
ENV=/n/home12/smagel/data/envs/graphnet_torch26_cpu_dev_dirdist__unstable
export LD_LIBRARY_PATH="${ENV}/lib"
SPINE=/n/home12/smagel/data/spine
export PYTHONPATH="${SPINE}/src"
"${ENV}/bin/python" "${SPINE}/examples/make_pretrain_selections.py" \
  --db /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/nu_bench_data/subsets/hexagon_djcc_1000000.db \
  --geo /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/om_adapter_bench/occupancy/hexagon_geometry.npz \
  --heldout-parquet /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/nu_bench_data/sqlite/hexagon/selections/train_selection.parquet \
  --n-heldout 70000 --n-val-events 4000 --n-train 500000 --seed 0 \
  --expect-pool 1000000 --slice-marks 100000 --workers 32 \
  --out-dir /n/home12/smagel/data/om_adapter_bench/spine_repro/selections
