#!/bin/bash
#SBATCH --job-name=tng_5r200_stars
#SBATCH --output=/scratch/habjan.e/tng_cutout_logs/stars_%A_%a.out
#SBATCH --error=/scratch/habjan.e/tng_cutout_logs/stars_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --array=0-299%16

# Builds stars_within_5r200_<halo>.hdf5 star (PartType4) cutouts for every simulation in
# SIMS, in ONE array: index -> (SIMS[i / N_HALOS], halo on line
# i % N_HALOS + 1 of HALO_LIST). --array must be 0-(len(SIMS) * N_HALOS - 1).
#
# Reads the tng_chunking.slurm cache chunk by chunk (fields from
# tng_cutout_config.FIELDS['stars']). A task whose chunks are not all cached stops
# at once and writes nothing, so submit after the chunking has finished, e.g.
#   sbatch --dependency=afterok:<chunking job id> run_tng_5r200_cutouts_stars.slurm
# Re-submitting is safe: finished cutouts are skipped (files are written as
# .part and renamed only when complete).

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
"$PYTHON" TNG_5r200_cutout_stars.py --sim "${SIM}" --halo-id "${HALO_ID}"
