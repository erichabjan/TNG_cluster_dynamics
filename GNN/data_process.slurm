#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/gnn_slurm_logs/data_process_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10G

#SBATCH -p reservation
#SBATCH --reservation=m_testing

# SBATCH --partition=short

source activate cl_dyn

python3 /home/habjan.e/TNG/TNG_cluster_dynamics/GNN/data.py