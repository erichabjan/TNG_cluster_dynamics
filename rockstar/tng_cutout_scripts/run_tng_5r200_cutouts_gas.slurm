#!/bin/bash
#SBATCH --job-name=tng_5r200_gas
#SBATCH --output=/scratch/habjan.e/tng_cutout_logs/gas_%A_%a.out
#SBATCH --error=/scratch/habjan.e/tng_cutout_logs/gas_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --array=0-299%16

# Adds the X-ray gas fields (Density, InternalEnergy, ElectronAbundance,
# StarFormationRate) to the existing gas_within_5r200_<halo>.hdf5 cutouts of
# every simulation in SIMS, in ONE array: index -> (SIMS[i / N_HALOS],
# halo on line i % N_HALOS + 1 of HALO_LIST). --array must be
# 0-(len(SIMS) * N_HALOS - 1).
#
# Where each chunk's fields come from is decided in tng_cutout_config.py:
#   TNG300-1        : the existing full gas chunks under mcdonough.b (no download needed)
#   TNG300-2 / -3   : the tng_chunking.slurm cache on /scratch
# A task whose chunks are not all available stops at once and writes nothing,
# so the TNG300-2/-3 tasks need tng_chunking.slurm to have finished, e.g.
#   sbatch --dependency=afterok:<chunking job id> run_tng_5r200_cutouts_gas.slurm
# Re-submitting is safe: cutouts already carrying the fields are skipped.
#
# The previous TNG-Cluster settings of this file (fresh pos/vel/mass/ID gas
# cutouts) are still available as `TNG_5r200_cutout_gas.py --fresh`.

SIMS=(TNG300-1 TNG300-2 TNG300-3)
N_HALOS=100
HALO_LIST="/home/habjan.e/TNG/Sandbox_notebooks/rockstar_sandboxes/halo_ids.txt"

SIM=${SIMS[$(( SLURM_ARRAY_TASK_ID / N_HALOS ))]}
HALO_ID=$(sed -n "$(( SLURM_ARRAY_TASK_ID % N_HALOS + 1 ))p" "${HALO_LIST}")
if [[ -z "${SIM}" || -z "${HALO_ID}" ]]; then
  echo "[error] array index ${SLURM_ARRAY_TASK_ID}: no simulation/halo (check --array, SIMS, N_HALOS)"
  exit 2
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn
PYTHON="$CONDA_PREFIX/bin/python"      # the env's interpreter, not the shared Anaconda

mkdir -p /scratch/habjan.e/tng_cutout_logs
export TNG_API_KEY="4486a4dd0aa3dd16f1d5fe60b5a22687"

echo "[info] task=${SLURM_ARRAY_TASK_ID} sim=${SIM} halo_id=${HALO_ID}"

cd /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/tng_cutout_scripts
"$PYTHON" TNG_5r200_cutout_gas.py --sim "${SIM}" --halo-id "${HALO_ID}"
