#!/bin/bash
#SBATCH --job-name=run_rockstar
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/rockstar_slurm_logs/run_rockstar_%A_%a.log
#SBATCH --array=0-100%28
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

cd /projects/mccleary_group/habjan.e/TNG/Data/rockstar_output/tng_rockstar_output

cluster_id=$SLURM_ARRAY_TASK_ID
python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/tng_sub_finder.py "$cluster_id"