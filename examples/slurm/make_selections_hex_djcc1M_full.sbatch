#!/bin/bash
# Build the 1M-slice pretraining selections (val 4000 / train 1M candidates,
# seed-0 shuffle, can_always_split filter). 100k/500k runs slice prefixes.
#SBATCH --job-name=spine_mksel1M
#SBATCH --partition=shared
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --output=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/mksel1M_%j.out
#SBATCH --error=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/mksel1M_%j.err
set -euo pipefail
ENV=/n/home12/smagel/data/envs/graphnet_torch26_cpu_dev_dirdist__unstable
export LD_LIBRARY_PATH="${ENV}/lib"
export PYTHONPATH="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/spine-worktrees/graphnet-split/src"
"${ENV}/bin/python" "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/spine-worktrees/graphnet-split/examples/make_pretrain_selections.py" \
  --db /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/nu_bench_data/subsets/hexagon_djcc_1000000.db \
  --heldout-parquet /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/nu_bench_data/sqlite/hexagon/selections/train_selection.parquet \
  --n-heldout 70000 --n-val-events 4000 --n-train 1000000 --seed 0 \
  --expect-pool 1000000 --slice-marks 100000 500000 --workers 32 \
  --out-dir /n/home12/smagel/data/om_adapter_bench/spine_repro/selections_1M
