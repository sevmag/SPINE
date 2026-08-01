#!/bin/bash
# SPINE reproduction of the v1 occupancy 100k pretraining (fp32, cwm time,
# hexagon djcc-1M, batch 32, 1x H200). DB staged to node-local scratch.
#SBATCH --job-name=spine_occ100k
#SBATCH --partition=gpu_h200
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/occ100k_%j.out
#SBATCH --error=/n/home12/smagel/data/om_adapter_bench/spine_repro/logs/occ100k_%j.err
set -euo pipefail

ENV=/n/home12/smagel/data/envs/graphnet_torch26_cu128_dev_dirdist__unstable
WT=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/graphnet-worktrees/per-detector-adapters
SPINE=/n/home12/smagel/data/spine
SUBSET_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel/nu_bench_data/subsets

export LD_LIBRARY_PATH="${ENV}/lib"
export PYTHONPATH="${WT}/src:${SPINE}/src:${SPINE}/examples"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DB_NAME="hexagon_djcc_1000000.db"
LOCAL="${SCRATCH_DIR:-/scratch/${USER}}/om_bench"
mkdir -p "${LOCAL}"
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

cd "${SPINE}/examples"
"${ENV}/bin/python" train_curtain.py \
  +experiment=hex_occ_v1cwm_100k \
  data.db="${DST}"
