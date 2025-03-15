#!/bin/bash
#SBATCH --job-name=DS+_virial_cluster_3
#SBATCH --output=DS+_virial_cluster_3.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=300
#SBATCH --time=16:00:00
#SBATCH --mem=300G
#SBATCH --partition=long

source activate sb_metacal  

python3 DS+_virial.py