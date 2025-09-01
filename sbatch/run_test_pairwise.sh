#!/bin/bash
#SBATCH --job-name=test_pairwise_metrics           # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8                # Number of CPUs per task (increase if needed)
#SBATCH --mem=512G                        # Memory per node (increase based on your model size)
#SBATCH --time=23:00:00                  # Time limit hrs:min:sec
#SBATCH --output=test_pairwise_metrics_%j.out     # Standard output log
#SBATCH --error=test_pairwise_metrics_%j.err      # Standard error log

# Load Conda module
module load conda

# Define environment name
ENV_NAME="my_env"

# Define script path
SCRIPT_PATH="/users/rsriramb/brain_extraction/python/quantitative/get_pairwise_metrics.py"

# Activate Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
conda activate $ENV_NAME

# Run the Python script (limit to 30 scans)
echo "Running Python script '$SCRIPT_PATH' with --max-scans 30..."
python -u $SCRIPT_PATH --max-scans 30

# Deactivate Conda environment
echo "Deactivating Conda environment '$ENV_NAME'..."
conda deactivate

echo "Job completed."