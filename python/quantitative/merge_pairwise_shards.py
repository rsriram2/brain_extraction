#!/usr/bin/env python3
"""Merge partial shard CSVs into the raw CSV and optionally run aggregation by
calling get_pairwise_metrics.py with --merge-only.

Usage:
  python merge_pairwise_shards.py --shard-dir /scratch/$USER/pairwise_shards \
      --out /path/to/results/pairwise_2x2_metrics_all_scans.csv --run-aggregate
"""
import argparse
import glob
import os
import pandas as pd
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--shard-dir", required=True, help="Directory containing shard CSVs (pattern pairwise_shard_*.csv)")
parser.add_argument("--out", required=True, help="Output merged CSV path")
parser.add_argument("--run-aggregate", action="store_true", help="If set, call get_pairwise_metrics.py --merge-only after merging")
args = parser.parse_args()

parts = sorted(glob.glob(os.path.join(args.shard_dir, "pairwise_shard_*.csv")))
if not parts:
    raise SystemExit(f"No shard CSVs found in {args.shard_dir}")

print(f"Found {len(parts)} shard files; merging...")
dfs = [pd.read_csv(p) for p in parts]
df = pd.concat(dfs, ignore_index=True)
# optional: drop duplicate rows if reruns occurred
df = df.drop_duplicates()
# write merged file
os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
print(f"Writing merged CSV to {args.out} ({len(df)} rows)")
df.to_csv(args.out, index=False)

if args.run_aggregate:
    script = os.path.join(os.path.dirname(__file__), "get_pairwise_metrics.py")
    cmd = ["python", script, "--merge-only"]
    print("Running aggregation/plots via:", " ".join(cmd))
    subprocess.check_call(cmd)

print("Done.")
