#!/bin/bash
#SBATCH --job-name=run_rockstar
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/run_rockstar_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=00:30:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source activate sb_metacal

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/sub_finder.py