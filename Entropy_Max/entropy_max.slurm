#!/bin/bash
#SBATCH --job-name=entropy_max_cluster_3
#SBATCH --output=entropy_max_cluster_3.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate sb_metacal  

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/entropy_max.py