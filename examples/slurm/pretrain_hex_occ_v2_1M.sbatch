#!/bin/bash
# SPINE v2 (occupancy+dt) pretraining at 1M events; same recipe as the v1
# reproduction runs, uniform dt_scale across the series. DB staged to scratch.
#SBATCH --job-name=spine_v2_1M
#SBATCH --partition=arguelles_delgado_h100
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/v2_1M_%j.out
#SBATCH --error=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/v2_1M_%j.err
set -euo pipefail

ENV=/n/home12/smagel/data/envs/graphnet_torch26_cu128_dev_dirdist__unstable
GWT=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/graphnet-worktrees/per-detector-adapters
SPINE=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/spine-worktrees/graphnet-split
SUBSET_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/nu_bench_data/subsets

export LD_LIBRARY_PATH="${ENV}/lib"
export PYTHONPATH="${GWT}/src:${SPINE}/src:${SPINE}/integrations"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DB_NAME="hexagon_djcc_1000000.db"
LOCAL="${SCRATCH_DIR:-/scratch/${USER}}/om_bench"
mkdir -p "${LOCAL}"
df -h "${LOCAL}" | tail -1
SRC="${SUBSET_DIR}/${DB_NAME}"
DST="${LOCAL}/${DB_NAME}"
exec 200>"${LOCAL}/.stage_${DB_NAME}.lock"
flock 200
if [ ! -s "${DST}" ] || \
   [ "$(stat -c%s "${DST}" 2>/dev/null || echo 0)" != "$(stat -c%s "${SRC}")" ]; then
  echo "staging ${SRC} -> ${DST} ($(date))"
  cp "${SRC}" "${DST}.tmp" && mv "${DST}.tmp" "${DST}"
fi
flock -u 200
echo "reading hexagon from ${DST} ($(date))"

"${ENV}/bin/python" "${SPINE}/examples/train_curtain.py" \
  +experiment=hex_occ_v2cwm_1M \
  data.db="${DST}"
