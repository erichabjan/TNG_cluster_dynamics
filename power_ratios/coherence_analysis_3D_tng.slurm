#!/bin/bash
#SBATCH --job-name=coherence_3d_tng
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/DS+_slurm_logs/coherence_3d_tng_%A_%a.log
#SBATCH --array=0-299%101
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --partition=short

DEPOSIT=${DEPOSIT:-sph}
FORCE=${FORCE:-0}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

sims=(TNG300-1 TNG300-2 TNG300-3)

sim_index=$(( SLURM_ARRAY_TASK_ID / 100 ))
cluster_id=$(( SLURM_ARRAY_TASK_ID % 100 ))

sim=${sims[$sim_index]}

out_dir=/projects/mccleary_group/habjan.e/TNG/Data/$( [ "$DEPOSIT" = sph ] && echo coherence_data_3D_sph || echo coherence_data_3D )/$sim
# power_cross_error is the last file the script saves
if [ "$FORCE" != 1 ] && [ -e "$out_dir/power_cross_error_${cluster_id}.npy" ]; then
    echo "$sim cluster $cluster_id: 3D outputs already in $out_dir, skipping"
    exit 0
fi

echo "Running 3D | sim: $sim | cluster: $cluster_id | deposit: $DEPOSIT"

cd /projects/mccleary_group/habjan.e/TNG/Data

# Called directly rather than through srun: jobs submitted from an OOD session
# inherit SLURM_EXPORT_ENV=NONE, which strips the environment from srun steps.
/home/habjan.e/miniconda3/envs/cl_dyn/bin/python3 -u \
    /home/habjan.e/TNG/TNG_cluster_dynamics/power_ratios/coherence_analysis_3D_tng.py "$cluster_id" "$sim" --deposit "$DEPOSIT"
