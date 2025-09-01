#!/bin/bash
#SBATCH --job-name=pairwise_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=pairwise_aggregate_%j.out

module load conda
conda activate my_env

SHARD_DIR="/scratch/$USER/pairwise_shards"
MERGED_CSV="/users/rsriramb/brain_extraction/results/quantitative/pairwise_2x2_metrics_all_scans.csv"

python /users/rsriramb/brain_extraction/python/quantitative/merge_pairwise_shards.py \
  --shard-dir "$SHARD_DIR" --out "$MERGED_CSV" --run-aggregate

echo "Aggregation job finished"
