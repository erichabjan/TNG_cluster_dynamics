#!/bin/bash
#SBATCH --job-name=tng99_chunks
#SBATCH --output=/scratch/habjan.e/tng_chunk_logs/tng_chunks_%A_%a.out
#SBATCH --error=/scratch/habjan.e/tng_chunk_logs/tng_chunks_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --array=0-27

# (Optional) load python/conda as you do on your system
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

mkdir -p /scratch/habjan.e/tng_chunk_logs

export TNG_API_KEY="Your API key"

python tng_chunking.py \
  --sim TNG300-1 \
  --snap 99 \
  --cache-dir /scratch/habjan.e/_tmp_tng_chunks \
  --use-slurm-array \
  --array-ntasks 28 \
  --default-min 0 \
  --default-max 599