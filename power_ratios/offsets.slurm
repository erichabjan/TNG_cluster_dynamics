#!/bin/bash
#SBATCH --job-name=offsets
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/DS+_slurm_logs/offsets_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=short

# One process pool of 28 workers over all 800 clusters (8 simulations x 100).
# The job is dominated by reading the TNG300-1 cutouts (~62 GB, ~26 s/cluster
# single-threaded), so it is I/O bound rather than CPU bound; 4 hours is well
# clear of the ~1 h this needs even with no parallel speed-up on the filesystem.
#
# The TNG particle arrays are streamed in chunks inside offsets.py, so memory is
# a few hundred MB per worker regardless of cluster size -- 64G is headroom, not
# a requirement.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

cd /projects/mccleary_group/habjan.e/TNG/Data

srun python3 /home/habjan.e/TNG/TNG_cluster_dynamics/power_ratios/offsets.py \
    --nproc "$SLURM_CPUS_PER_TASK" \
    --aperture 1.0
