#!/bin/bash
#SBATCH --job-name=coherence_tng
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/DS+_slurm_logs/coherence_tng_%A_%a.log
#SBATCH --array=0-299%101
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --partition=short

# 2D coherence over random projections (100 by default), one task per (TNG300
# resolution, cluster). Reads the SPH smoothing lengths written by the 3D script
# and builds any that are missing. Scaled from 1000-projection runs: TNG300-1
# median 26 min, max 87 min, peak 20 GB; TNG300-2/3 under 20 min.
#
# Clusters already run with 1000 projections are skipped; their first 100
# projections are the same directions a 100-projection run would use.
# The deposit defaults to SPH; to rerun with CIC:
#   sbatch --export=ALL,DEPOSIT=cic coherence_analysis_tng.slurm
# A task whose outputs already exist exits immediately; to recompute anyway:
#   sbatch --export=ALL,FORCE=1 coherence_analysis_tng.slurm

DEPOSIT=${DEPOSIT:-sph}
FORCE=${FORCE:-0}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

sims=(TNG300-1 TNG300-2 TNG300-3)

sim_index=$(( SLURM_ARRAY_TASK_ID / 100 ))
cluster_id=$(( SLURM_ARRAY_TASK_ID % 100 ))

sim=${sims[$sim_index]}

out_dir=/projects/mccleary_group/habjan.e/TNG/Data/$( [ "$DEPOSIT" = sph ] && echo coherence_data_sph || echo coherence_data_cic )/$sim
# power_cross_error is the last file the script saves
if [ "$FORCE" != 1 ] && [ -e "$out_dir/power_cross_error_${cluster_id}.npy" ]; then
    echo "$sim cluster $cluster_id: 2D outputs already in $out_dir, skipping"
    exit 0
fi

echo "Running 2D | sim: $sim | cluster: $cluster_id | deposit: $DEPOSIT"

cd /projects/mccleary_group/habjan.e/TNG/Data

# Called directly rather than through srun: jobs submitted from an OOD session
# inherit SLURM_EXPORT_ENV=NONE, which strips the environment from srun steps.
/home/habjan.e/miniconda3/envs/cl_dyn/bin/python3 -u \
    /home/habjan.e/TNG/TNG_cluster_dynamics/power_ratios/coherence_analysis_tng.py "$cluster_id" "$sim" --deposit "$DEPOSIT"
