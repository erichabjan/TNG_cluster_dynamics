#!/bin/bash
#SBATCH --job-name=graph_maker
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/gnn_slurm_logs/graph_maker_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate cluster_dyn

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/GNN/graph_maker.py