#!/bin/bash
#SBATCH --job-name=bone_thresh           # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8                # Number of CPUs per task (increase if needed)
#SBATCH --mem=64G                        # Memory per node (increase based on your model size)
#SBATCH --time=23:00:00                  # Time limit hrs:min:sec
#SBATCH --output=bone_thresh_%j.out     # Standard output log
#SBATCH --error=bone_thresh_%j.err      # Standard error log

# Load Conda module
module load conda

# Define environment name
ENV_NAME="my_env"

# Define script path
SCRIPT_PATH="/users/rsriramb/brain_extraction/python/quantitative/calibrate_bone_threshold.py"

# Activate Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
conda activate $ENV_NAME

# Run the Python script with explicit args
echo "Running Python script '$SCRIPT_PATH' with args..."
python -u "$SCRIPT_PATH" \
  --out "/users/rsriramb/brain_extraction/results/quantitative/bone_hu_threshold" \
  --dice-thresh 0.90 \
  --format both \
  --bootstrap 1000

# Deactivate Conda environment
echo "Deactivating Conda environment '$ENV_NAME'..."
conda deactivate

echo "Job completed."
