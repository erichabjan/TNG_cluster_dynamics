#!/bin/bash
#SBATCH --job-name=run_rockstar
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/rockstar_slurm_logs/run_rockstar_%j.log
#SBATCH --array=50
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
source activate cluster_dyn

cd /projects/mccleary_group/habjan.e/TNG/Data/rockstar_output

cluster_id=$SLURM_ARRAY_TASK_ID
python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/tng_sub_finder.py "$cluster_id"