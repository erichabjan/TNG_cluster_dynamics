#!/bin/bash
#SBATCH --job-name=run_rockstar_100
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/run_rockstar_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
source activate cluster_dyn

cd /home/habjan.e/TNG/Data

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/sub_finder.py