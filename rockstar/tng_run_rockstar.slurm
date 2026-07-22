#!/bin/bash
#SBATCH --job-name=run_rockstar
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/rockstar_slurm_logs/run_rockstar_%A_%a.log
#SBATCH --array=0-299%28
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

sims=(TNG300-1 TNG300-2 TNG300-3)

sim_index=$(( SLURM_ARRAY_TASK_ID / 100 ))
cluster_id=$(( SLURM_ARRAY_TASK_ID % 100 ))

sim=${sims[$sim_index]}

echo "Running sim: $sim | cluster: $cluster_id"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

mkdir -p "/projects/mccleary_group/habjan.e/TNG/Data/rockstar_output/${sim}_rockstar_output"
cd "/projects/mccleary_group/habjan.e/TNG/Data/rockstar_output/${sim}_rockstar_output"

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/tng_sub_finder.py "$cluster_id" "$sim"