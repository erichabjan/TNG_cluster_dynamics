#!/bin/bash
#SBATCH --job-name=tng99_chunks
#SBATCH --output=/scratch/habjan.e/tng_chunk_logs/tng_chunks_%A_%a.out
#SBATCH --error=/scratch/habjan.e/tng_chunk_logs/tng_chunks_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --array=0-27

# Snapshot-99 chunk downloads for every simulation in SIMS, in ONE array: the
# simulations download side by side, each strided over its own TASKS[i] tasks.
# The array size must equal the sum of TASKS (16 + 8 + 4 = 28 -> 0-27).
#
# What each simulation fetches is decided in tng_cutout_config.py, chunk by
# chunk and per particle type -- nothing about fields or chunk counts is set
# here:
#   TNG300-1 : gas is already on disk (mcdonough.b's full gas chunks), so only
#              stars + BHs are downloaded (600 chunks)
#   TNG300-2 : gas (X-ray fields) + stars + BHs (100 chunks)
#   TNG300-3 : gas (X-ray fields) + stars + BHs (16 chunks)
# Re-submitting is safe: chunks already cached with the requested fields, or
# satisfied by an existing source, are skipped.

SIMS=(TNG300-1 TNG300-2 TNG300-3)
TASKS=(16 8 4)          # tasks per simulation, roughly by download volume

# ---- map the array index onto (simulation, task within it) ----
idx=${SLURM_ARRAY_TASK_ID}
SIM=""
for i in "${!SIMS[@]}"; do
  if (( idx < TASKS[i] )); then
    SIM=${SIMS[$i]}
    N_TASKS=${TASKS[$i]}
    TASK_ID=$idx
    break
  fi
  idx=$(( idx - TASKS[i] ))
done
if [[ -z "${SIM}" ]]; then
  echo "[error] array index ${SLURM_ARRAY_TASK_ID} exceeds sum(TASKS); fix --array"
  exit 2
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn
# `conda activate` leaves the shared Anaconda ahead of the env in PATH, so call
# the env's interpreter directly.
PYTHON="$CONDA_PREFIX/bin/python"

mkdir -p /scratch/habjan.e/tng_chunk_logs

export TNG_API_KEY="4486a4dd0aa3dd16f1d5fe60b5a22687"

echo "[info] array ${SLURM_ARRAY_TASK_ID} -> sim=${SIM} task ${TASK_ID}/${N_TASKS}"

cd /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/tng_cutout_scripts
"$PYTHON" tng_chunking.py \
  --sim "${SIM}" \
  --task-id "${TASK_ID}" \
  --n-tasks "${N_TASKS}"
