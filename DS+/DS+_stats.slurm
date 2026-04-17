#!/bin/bash
#SBATCH --job-name=DS+_virial_cluster
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/DS+_slurm_logs/DS+_stats_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

cd /projects/mccleary_group/habjan.e/TNG/Data

srun --cpu-bind=cores python3 /home/habjan.e/TNG/TNG_cluster_dynamics/DS+/DS+_stats.py