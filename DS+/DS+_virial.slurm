#!/bin/bash
#SBATCH --job-name=DS+_virial_cluster_3
#SBATCH --output=DS+_virial_cluster_3.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=30:00:00
#SBATCH --mem=100G
#SBATCH --partition=short

source activate sb_metacal  

python3 DS+_virial.py