#!/bin/bash
#SBATCH --job-name=tng_5r200_cutouts
#SBATCH --output=/scratch/habjan.e/tng_cutout_logs/cutout_%A_%a.out
#SBATCH --error=/scratch/habjan.e/tng_cutout_logs/cutout_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --array=0-99

# ---- user settings ----
SIM="TNG300-1"
SNAP=99
RMULT=5.0
RDEF="Group_R_Crit200"
CENTER_FIELD="GroupPos"

CHUNKS_DIR="/scratch/habjan.e/_tmp_tng_chunks/${SIM}/snap_${SNAP}/snapshot_chunks"
OUT_DIR="/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/5r200_data"

HALO_LIST="halo_ids.txt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

mkdir -p /scratch/habjan.e/tng_cutout_logs
mkdir -p "${OUT_DIR}"

export TNG_API_KEY="Your API Key"

# ---- pick halo for this task ----
HALO_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${HALO_LIST}")

if [[ -z "${HALO_ID}" ]]; then
  echo "[error] No halo ID found for array index ${SLURM_ARRAY_TASK_ID} in ${HALO_LIST}"
  exit 2
fi

OUTFILE="${OUT_DIR}/gas_within_5r200_${HALO_ID}.hdf5"

echo "[info] task=${SLURM_ARRAY_TASK_ID} halo_id=${HALO_ID}"
echo "[info] chunks_dir=${CHUNKS_DIR}"
echo "[info] outfile=${OUTFILE}"

python TNG_5r200_cutout_gas.py \
  --halo-id "${HALO_ID}" \
  --sim "${SIM}" \
  --snap "${SNAP}" \
  --rmult "${RMULT}" \
  --rdef "${RDEF}" \
  --center-field "${CENTER_FIELD}" \
  --chunks-dir "${CHUNKS_DIR}" \
  --out "${OUTFILE}" \
  --overwrite
