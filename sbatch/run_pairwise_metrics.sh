#!/bin/bash
#SBATCH --job-name=pairwise_shard           # Job name
#  Run a single test shard (change back to 0-49 or desired range for full run)
#SBATCH --array=0-0
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

# Configure shard parameters (set to 1 for test; revert to 50 for full run)
# Adjust NUM_SHARDS to the number of shards you want and update the SBATCH --array directive above accordingly
NUM_SHARDS=1
SHARD_ID=${SLURM_ARRAY_TASK_ID}

# Output directory for partial shard CSVs (should be a fast shared location or scratch)
OUT_DIR="/users/rsriramb/brain_extraction/results/quantitative/pairwise_shards_remaining"
mkdir -p "$OUT_DIR"
OUT_PARTIAL="${OUT_DIR}/pairwise_shard_${SHARD_ID}.csv"

# Process only the stems in this CSV and skip the CTbet_Docker method (per recent script changes)
STEMS_CSV="/users/rsriramb/brain_extraction/results/quantitative/skipped_ctdock_stems_list.csv"
SKIP_METHOD="CTbet_Docker"

echo "Running shard ${SHARD_ID} of ${NUM_SHARDS} -> ${OUT_PARTIAL}"
python -u "$SCRIPT_PATH" --num-shards ${NUM_SHARDS} --shard-id ${SHARD_ID} --out-partial "$OUT_PARTIAL" --stems-csv "$STEMS_CSV" --skip-method "$SKIP_METHOD"

echo "Deactivating Conda environment '$ENV_NAME'..."
conda deactivate

echo "Shard ${SHARD_ID} completed."