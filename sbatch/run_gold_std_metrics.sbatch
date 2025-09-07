#!/bin/bash
#SBATCH --job-name=gold_std_metrics           # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8                # Number of CPUs per task (increase if needed)
#SBATCH --mem=128G                        # Memory per node (increase based on your model size)
#SBATCH --time=23:00:00                  # Time limit hrs:min:sec
#SBATCH --output=gold_std_metrics_%j.out     # Standard output log
#SBATCH --error=gold_std_metrics_%j.err      # Standard error log

# Load Conda module (adjust if your cluster uses a different module system)
module load conda

# Define environment name
ENV_NAME="my_env"

# Path to the Python script (ensure this path exists on the cluster)
SCRIPT_PATH="/users/rsriramb/brain_extraction/python/quantitative/get_gold_std_metrics.py"

# Activate Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
conda activate $ENV_NAME

# Run the Python script with the flags it expects
# The script accepts: --out-dir, --threshold, and --save-csv
OUT_DIR="/users/rsriramb/brain_extraction"
THRESH=0.95

echo "Running Python script '$SCRIPT_PATH' with args..."
python -u "$SCRIPT_PATH" \
  --out-dir "$OUT_DIR" \
  --threshold $THRESH \
  --save-csv

# Deactivate Conda environment
echo "Deactivating Conda environment '$ENV_NAME'..."
conda deactivate

echo "Job completed."
