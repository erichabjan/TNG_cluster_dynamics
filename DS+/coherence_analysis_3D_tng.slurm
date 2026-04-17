#!/bin/bash
#SBATCH --job-name=coherence_3D_tng
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/DS+_slurm_logs/coherence_3D_tng_%A_%a.log
#SBATCH --array=0-100%101
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

source ~/miniconda3/etc/profile.d/conda.sh
source activate cl_dyn

cd /projects/mccleary_group/habjan.e/TNG/Data

cluster_id=$SLURM_ARRAY_TASK_ID
srun --cpu-bind=cores python3 /home/habjan.e/TNG/TNG_cluster_dynamics/DS+/coherence_analysis_3D_tng.py "$cluster_id"
