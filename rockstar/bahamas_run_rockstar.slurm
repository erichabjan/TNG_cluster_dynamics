#!/bin/bash
#SBATCH --job-name=run_rockstar
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/rockstar_slurm_logs/bahamas_run_rockstar_%A_%a.log
#SBATCH --array=0-499%28
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=10G

#SBATCH -p reservation
#SBATCH --reservation=m_testing

# SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn
cd /projects/mccleary_group/habjan.e/TNG/Data/rockstar_output

models=(CDMb SIDM0.1b SIDM0.3b SIDM1b vdSIDMb)

model_id=$(( SLURM_ARRAY_TASK_ID / 100 ))
sim_id=$(( SLURM_ARRAY_TASK_ID % 100 + 1 ))

dm_model=${models[$model_id]}
cluster_id=$(printf "%03d" $sim_id)

echo "Running model: $dm_model | cluster: $cluster_id"

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/bahamas_sub_finder.py "$cluster_id" "$dm_model"