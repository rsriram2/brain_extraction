#!/bin/bash
#SBATCH --job-name=pairwise_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=pairwise_aggregate_%j.out
#SBATCH --error=pairwise_aggregate_%j.err

module load conda
conda activate my_env

SHARD_DIR="/users/rsriramb/brain_extraction/results/quantitative/pairwise_comparison/pairwise_shards"
MERGED_CSV="/users/rsriramb/brain_extraction/results/quantitative/pairwise_comparison/pairwise_2x2_metrics_all_scans.csv"

PYTHON_SCRIPT="/users/rsriramb/brain_extraction/python/quantitative/merge_pairwise_shards.py"

echo "Merging shard CSVs from $SHARD_DIR -> $MERGED_CSV"

# run merge + aggregate (SBATCH will capture stdout/stderr to the job files)
python "$PYTHON_SCRIPT" --shard-dir "$SHARD_DIR" --out "$MERGED_CSV" --run-aggregate

echo "Aggregation job finished"
