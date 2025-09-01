#!/bin/bash
#SBATCH --job-name=pairwise_shard           # Job name
#  Running 100 shards (IDs 0..99); scheduler will decide concurrency
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=pairwise_shard_%A_%a.out
#SBATCH --error=pairwise_shard_%A_%a.err

# Load Conda module
module load conda

# Define environment name
ENV_NAME="my_env"

# Define script path
SCRIPT_PATH="/users/rsriramb/brain_extraction/python/quantitative/get_pairwise_metrics.py"

echo "Activating Conda environment '$ENV_NAME'..."
conda activate $ENV_NAME

# Configure shard parameters
# Adjust NUM_SHARDS to the number of shards you want and update the SBATCH --array directive above accordingly
NUM_SHARDS=100
SHARD_ID=${SLURM_ARRAY_TASK_ID}

# Output directory for partial shard CSVs (should be a fast shared location or scratch)
OUT_DIR="/scratch/$USER/pairwise_shards"
mkdir -p "$OUT_DIR"
OUT_PARTIAL="${OUT_DIR}/pairwise_shard_${SHARD_ID}.csv"

echo "Running shard ${SHARD_ID} of ${NUM_SHARDS} -> ${OUT_PARTIAL}"
python -u "$SCRIPT_PATH" --num-shards ${NUM_SHARDS} --shard-id ${SHARD_ID} --out-partial "$OUT_PARTIAL"

echo "Deactivating Conda environment '$ENV_NAME'..."
conda deactivate

echo "Shard ${SHARD_ID} completed."