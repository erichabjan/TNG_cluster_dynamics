#!/bin/bash
#SBATCH --job-name=tng_matching
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/rockstar_slurm_logs/tng_matching_%A_%a.log
#SBATCH --array=0-100%28
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

cluster_id=$SLURM_ARRAY_TASK_ID
python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/tng_subhalo_matching.py "$cluster_id"