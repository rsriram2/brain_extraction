#!/bin/bash
#SBATCH --job-name=rim_metrics           # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8                # Number of CPUs per task (increase if needed)
#SBATCH --mem=512G                        # Memory per node (increase based on your model size)
#SBATCH --time=23:00:00                  # Time limit hrs:min:sec
#SBATCH --output=rim_metrics_%j.out     # Standard output log
#SBATCH --error=rim_metrics_%j.err      # Standard error log

# Load Conda module
module load conda

# Define environment name
ENV_NAME="my_env"

# Define script path
SCRIPT_PATH="/users/rsriramb/brain_extraction/python/quantitative/get_rim_metrics.py"

# Activate Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
conda activate $ENV_NAME

# Run the Python script
echo "Running Python script '$SCRIPT_PATH'..."
python -u $SCRIPT_PATH

# Deactivate Conda environment
echo "Deactivating Conda environment '$ENV_NAME'..."
conda deactivate

echo "Job completed."