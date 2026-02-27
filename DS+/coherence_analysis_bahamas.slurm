#!/bin/bash
#SBATCH --job-name=coherence_analysis
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/DS+_slurm_logs/coherence_analysis_%A_%a.log
#SBATCH --array=0-499%28
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=10G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

source ~/miniconda3/etc/profile.d/conda.sh
source activate cl_dyn 

cd /projects/mccleary_group/habjan.e/TNG/Data

models=(CDMb SIDM0.1b SIDM0.3b SIDM1b vdSIDMb)

model_id=$(( SLURM_ARRAY_TASK_ID / 100 ))
sim_id=$(( SLURM_ARRAY_TASK_ID % 100 + 1 ))

dm_model=${models[$model_id]}
cluster_id=$(printf "%03d" $sim_id)

echo "Running model: $dm_model | cluster: $cluster_id"

srun --cpu-bind=cores python3 /home/habjan.e/TNG/TNG_cluster_dynamics/DS+/coherence_analysis_bahamas.py "$cluster_id" "$dm_model"