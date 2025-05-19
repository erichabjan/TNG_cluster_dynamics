#!/bin/bash
#SBATCH --job-name=rockstar_halo
#SBATCH --output=rockstar_halo_%j.out
#SBATCH --error=rockstar_halo_%j.err
#SBATCH --time=24:00:00         # Request 24 hours runtime
#SBATCH --mem=32G               # Request 32GB of RAM for 41 million particles
#SBATCH --cpus-per-task=4       # Request 4 CPU cores

# Print job information
echo "Starting job at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

# Set input file path
INPUT_FILE="/home/habjan.e/TNG/Data/TNG_data/halo_cutouts_dm.hdf5"

# Create output directory
OUTPUT_DIR="/home/habjan.e/TNG/Data/rockstar_output/rockstar_results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Change to the directory containing your compiled executable
cd /home/habjan.e/TNG/Codes/rockstar

# Set library path to include current directory
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

# Set library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libhdf5.so

# Run the rockstar halo finder
./rockstar_interface "$INPUT_FILE" > $OUTPUT_DIR/rockstar_output.log 2>&1

# Check if the binary file was created
if [ -f "$OUTPUT_DIR/halos.bin" ]; then
    echo "Success! Halo catalog created at: $OUTPUT_DIR/halos.bin"
    ls -lh "$OUTPUT_DIR/halos.bin"
else
    echo "ERROR: Halo catalog file not found at expected location: $OUTPUT_DIR/halos.bin"
    # Check if it might have been created in the current directory instead
    if [ -f "halos.bin" ]; then
        echo "Found halos.bin in current directory. Moving to output directory..."
        mv halos.bin "$OUTPUT_DIR/"
    fi
fi

echo "Job completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"