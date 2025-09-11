#!/bin/bash
#SBATCH --job-name=run_rockstar
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/rockstar_slurm_logs/bahamas_run_rockstar_%j.log
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=10G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
source activate cluster_dyn
cd /projects/mccleary_group/habjan.e/TNG/Data/rockstar_output

models=(CDMb SIDM0.1b SIDM0.3b SIDM1b vdSIDMb)
dm_model=${models[$SLURM_ARRAY_TASK_ID]}

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/bahamas_sub_finder.py 10 "$dm_model"