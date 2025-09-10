import argparse
import pandas as pd
import numpy as np
import os
import glob
import itertools
import nibabel as nib
from typing import Tuple, Dict, List
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from medpy.metric import binary as medpy_binary
from medpy.metric.binary import hd95 as medpy_hd95

DATA_DIR = "/users/rsriramb/brain_extraction/results/quantitative/"
METHOD_DIRS = {
    'SynthStrip': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_synth',
    'Robust-CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask',
    'Brainchop': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_brainchop',
    'HD-CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_hdctbet',
    'CTbet_Docker': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_dockerctbet',
    'CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_original',
    'CT_BET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_ctbet',
}
OUT_CSV_RAW   = os.path.join(DATA_DIR, "pairwise_2x2_metrics_all_scans.csv")

# ===== helpers for geometry and metrics =====
def stem(p: str) -> str:
    base = os.path.basename(p)
    for suf in (".nii.gz", ".nii"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base

def patient_id_from_stem(s: str) -> str:
    # e.g., "6506-324_20170906_1752_ct" -> "6506-324"
    return s.split("_")[0]

def load_mask_bool(path: str):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32) > 0  # binarize
    return img, arr

def confusion_2x2(a, b):
    tp = np.count_nonzero(a &  b)
    fp = np.count_nonzero(a & ~b)
    fn = np.count_nonzero(~a &  b)
    tn = np.count_nonzero(~a & ~b)
    return tp, fp, fn, tn

def dice(tp, fp, fn):
    den = 2*tp + fp + fn
    return (2*tp/den) if den else np.nan

def iou(tp, fp, fn):
    den = tp + fp + fn
    return (tp/den) if den else np.nan

def accuracy(tp, fp, fn, tn):
    den = tp+fp+fn+tn
    return (tp+tn)/den if den else np.nan

def sensitivity(tp, fn):
    den = tp+fn
    return tp/den if den else np.nan

def specificity(tn, fp):
    den = tn+fp
    return tn/den if den else np.nan

def kappa(tp, fp, fn, tn):
    n = tp+fp+fn+tn
    if n == 0: return np.nan
    po = (tp+tn)/n
    pe = ((tp+fp)/n)*((tp+fn)/n) + ((fn+tn)/n)*((fp+tn)/n)
    den = 1 - pe
    return (po - pe)/den if den else np.nan


def voxel_volume_ml_from_img(img: nib.Nifti1Image) -> float:
    sx, sy, sz = img.header.get_zooms()[:3]
    return float((sx * sy * sz) / 1000.0)

def bootstrap_ci(vals: np.ndarray, B: int = 2000, alpha: float = 0.05, use_median: bool = True):
    if len(vals) == 0:
        return np.nan, np.nan
    
    stat_fun = np.median if use_median else np.mean
    
    res = bootstrap(
        data=(vals,),               # data as a tuple
        statistic=stat_fun,         # function to apply
        confidence_level=1-alpha,   # e.g. 0.95
        n_resamples=B,
        method="percentile",        # matches your old function
        random_state=0              # reproducible
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)

# ===== discover stems and compute per-scan pairwise rows =====
method_files = {m: glob.glob(os.path.join(d, "*.nii*")) for m,d in METHOD_DIRS.items()}
method_maps  = {m: {stem(p): p for p in files} for m,files in method_files.items()}

# stems present in ≥2 methods
all_stems: Dict[str, set] = {}
for m, mp in method_maps.items():
    for s in mp.keys():
        all_stems.setdefault(s, set()).add(m)
candidate_stems = sorted([s for s, ms in all_stems.items() if len(ms) >= 2])

# ----- CLI / sharding support -----
parser = argparse.ArgumentParser(description="Compute pairwise metrics (supports sharding)")
parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (split stems into this many parts)")
parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard id to process (0..num_shards-1)")
parser.add_argument("--out-partial", type=str, default=None, help="If set, write partial CSV for this shard to the given path and exit (skips aggregation)")
parser.add_argument("--merge-only", action="store_true", help="Load existing raw CSV and run aggregation/plots only (no per-scan compute)")
parser.add_argument("--stems-csv", type=str, default=None, help="Optional CSV (one column 'stem' or plain list) with stems to process (limits candidate stems)")
parser.add_argument("--skip-method", type=str, default="CTbet_Docker", help="Method name to skip when processing these stems (default: CTbet_Docker)")
args = parser.parse_args()

if args.num_shards > 1:
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"shard-id must be in [0, {args.num_shards-1}]")
    stems_sorted = candidate_stems
    # deterministic split: take every Nth stem starting at shard_id
    candidate_stems = stems_sorted[args.shard_id::args.num_shards]
    print(f"Shard mode: num_shards={args.num_shards} shard_id={args.shard_id} stems={len(candidate_stems)}")
else:
    print(f"Running single-job mode on {len(candidate_stems)} stems")

if args.stems_csv:
    print(f"Limiting candidate stems to CSV: {args.stems_csv} (remaining {len(candidate_stems)})")
print(f"Skipping method when loading per-stem masks: {args.skip_method}")

exclude = {
    '6109-317_20150302_0647_ct','6142-308_20150610_0707_ct','6193-324_20150924_1431_ct',
    '6257-335_20160118_1150_ct','6418-193_20161228_1248_ct','6470-296_20170602_0607_ct',
    '6480-154_20170622_0937_ct'
}

exclude_prefixes = ("6046", "6084", "6096", "6246", "6315", "6342", "6499")
candidate_stems = [
    s for s in candidate_stems 
    if s not in exclude and not patient_id_from_stem(s).startswith(exclude_prefixes)
]

# If user supplied a stems CSV, restrict candidate_stems to that list (intersect)
if args.stems_csv:
    import csv
    if not os.path.exists(args.stems_csv):
        raise SystemExit(f"stems CSV not found: {args.stems_csv}")
    stems_from_csv = []
    with open(args.stems_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and 'stem' in reader.fieldnames:
            for row in reader:
                stems_from_csv.append(row['stem'])
        else:
            fh.seek(0)
            for r in csv.reader(fh):
                if len(r) > 0:
                    stems_from_csv.append(r[0])

    stems_set = set(stems_from_csv)
    candidate_stems = [s for s in candidate_stems if s in stems_set]

# If merge-only was requested, load existing aggregated CSV and skip per-scan processing
if args.merge_only:
    if not os.path.exists(OUT_CSV_RAW):
        raise SystemExit(f"merge-only requested but raw CSV not found at {OUT_CSV_RAW}")
    df = pd.read_csv(OUT_CSV_RAW)
    print(f"merge-only: loaded {len(df)} rows from {OUT_CSV_RAW}; running aggregation/plots")

else:

    rows = []
    for s in candidate_stems:
        imgs, masks = {}, {}
        aff, shape = None, None
        # respect skip-method CLI flag: only consider methods present for this stem and not the skip method
        methods_here = sorted([m for m in all_stems[s] if m != args.skip_method])

        # only keep methods that actually have a file for this stem
        available_methods = [m for m in methods_here if method_maps.get(m, {}).get(s)]
        if len(available_methods) < 2:
            continue

        # load and check geometry for available methods only
        ok = True
        for m in available_methods:
            p = method_maps[m][s]
            img, arr = load_mask_bool(p)
            if shape is None:
                shape, aff = img.shape, img.affine
            else:
                if img.shape != shape or not np.allclose(img.affine, aff, atol=1e-3):
                    ok = False; break
            imgs[m], masks[m] = img, arr
        if not ok:
            continue

        pid = patient_id_from_stem(s)
        # pairwise metrics (per scan) over available methods
        for A, B in itertools.combinations(available_methods, 2):
            a, b = masks[A], masks[B]
            tp, fp, fn, tn = confusion_2x2(a, b)

            # symmetric sensitivity/specificity: average A->B and B->A
            sens_A_B = sensitivity(tp, fn)
            sens_B_A = sensitivity(tp, fp) if False else None  # placeholder
            # compute both directions properly using masks
            sens_AB = sensitivity(np.count_nonzero(a & b), np.count_nonzero(a & ~b))
            sens_BA = sensitivity(np.count_nonzero(a & b), np.count_nonzero(b & ~a))
            spec_AB = specificity(np.count_nonzero(~a & ~b), np.count_nonzero(a & ~b))
            spec_BA = specificity(np.count_nonzero(~a & ~b), np.count_nonzero(b & ~a))
            sensitivity_sym = np.nanmean([sens_AB, sens_BA])
            specificity_sym = np.nanmean([spec_AB, spec_BA])

            # surface-based metrics (MSD, HD95) in mm
            imgA = imgs[A]
            imgB = imgs[B]
            # compute MSD and HD95 using MedPy (assume medpy is installed)
            spacing = tuple(map(float, imgA.header.get_zooms()[:3]))
            msd = float(medpy_binary.asd(a.astype(bool), b.astype(bool), voxelspacing=spacing))
            # compute HD95 using MedPy's hd95 helper
            try:
                hd95 = float(medpy_hd95(a.astype(bool), b.astype(bool), voxelspacing=spacing))
            except Exception:
                hd95 = np.nan

            # ICV (mL)
            vox_ml = voxel_volume_ml_from_img(imgA)
            icv_A = float(a.sum() * vox_ml)
            icv_B = float(b.sum() * vox_ml)
            delta_icv_ml = icv_A - icv_B
            delta_icv_pct = (delta_icv_ml / icv_B * 100.0) if icv_B else np.nan

            rows.append({
                "patient_id": pid,
                "stem": s,
                "method_A": A, "method_B": B,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "dice": dice(tp, fp, fn),
                "iou": iou(tp, fp, fn),
                "sensitivity_sym": sensitivity_sym,
                "specificity_sym": specificity_sym,
                "msd_mm": msd,
                "hd95_mm": hd95,
                "icv_A_ml": icv_A,
                "icv_B_ml": icv_B,
                "delta_icv_ml": delta_icv_ml,
                "delta_icv_pct": delta_icv_pct,
                "n_vox": int(tp+fp+fn+tn)
            })

    df = pd.DataFrame(rows)
    print("unique patients:", df["patient_id"].nunique())

    # If running in shard mode and an out-partial path is provided, write only this shard's partial
    if args.num_shards > 1 and args.out_partial:
        out_partial = args.out_partial
        os.makedirs(os.path.dirname(out_partial), exist_ok=True) if os.path.dirname(out_partial) else None
        df.to_csv(out_partial, index=False)
        print(f"[partial] scans: {len(set(df['stem']))}  pairs: {len(df)}  partial rows saved -> {out_partial}")
        # skip the heavy aggregation/plotting for shard workers
        raise SystemExit(0)

    # default behavior: write full raw CSV
    df.to_csv(OUT_CSV_RAW, index=False)
    print(f"[raw] scans: {len(set(df['stem']))}  pairs: {len(df)}  rows saved -> {OUT_CSV_RAW}")

# ===== run pipeline for multiple metrics and aggregations =====
METRICS = ['dice', 'iou', 'sensitivity_sym', 'specificity_sym', 'msd_mm', 'hd95_mm', 'delta_icv_ml', 'delta_icv_pct']
AGGRS = ['mean', 'median']

run_aggregation = args.merge_only or not (args.num_shards > 1 and args.out_partial)
if run_aggregation:
    for metric in METRICS:
        for aggr in AGGRS:
            outdir = os.path.join(DATA_DIR, metric, aggr)
            os.makedirs(outdir, exist_ok=True)

            # collapse scans -> patient using chosen aggregation, dropping NaNs
            def agg_func(x):
                arr = x.dropna().to_numpy()
                if arr.size == 0:
                    return np.nan
                return np.nanmedian(arr) if aggr == 'median' else np.nanmean(arr)

            scan2pat = df.groupby(["patient_id", "method_A", "method_B"], as_index=False)[metric].agg(agg_func)
            scan2pat.rename(columns={metric: f"{metric}_patient"}, inplace=True)

            # diagnostic: how many patient-level rows for this metric/aggr
            n_scan2pat = len(scan2pat)
            n_non_nan = scan2pat[f"{metric}_patient"].dropna().shape[0]
            print(f"Processing metric={metric} aggr={aggr}: scan2pat rows={n_scan2pat} non-NaN patients={n_non_nan}")

            # cohort-level point estimates (per pair)
            pair_stats = (scan2pat
                          .groupby(["method_A", "method_B"], as_index=False)[f"{metric}_patient"]
                          .agg(point=(lambda x: np.nanmedian(x) if aggr == "median" else np.nanmean(x)),
                               n_patients=("count")))

            pair_csv = os.path.join(outdir, f"pairwise_patient_agg_{metric}_{aggr}.csv")
            pair_stats.to_csv(pair_csv, index=False)

            # bootstrap CIs (patient-level)
            ci_rows = []
            # prefer median for HD95 and MSD, otherwise follow aggr
            use_median = True if metric in ("hd95_mm", "msd_mm") else (aggr == "median")
            for (A, B), g in scan2pat.groupby(["method_A", "method_B"]):
                v = g[f"{metric}_patient"].to_numpy()
                v = v[~np.isnan(v)]
                if v.size == 0:
                    lo = hi = point = np.nan
                elif v.size == 1:
                    point = float(np.mean(v))
                    lo = hi = np.nan
                else:
                    stat_fun = np.median if use_median else np.mean
                    res = bootstrap(
                        data=(v,),
                        statistic=stat_fun,
                        confidence_level=0.95,
                        n_resamples=2000,
                        method="percentile",
                        random_state=0,
                    )
                    lo = float(res.confidence_interval.low)
                    hi = float(res.confidence_interval.high)
                    point = float(stat_fun(v))

                ci_rows.append({
                    "method_A": A, "method_B": B, "metric": metric,
                    "point": point, "ci_lo": lo, "ci_hi": hi,
                    "n_patients": int(v.size)
                })

            ci_df = pd.DataFrame(ci_rows)
            ci_csv = os.path.join(outdir, f"pairwise_{metric}_bootstrap_ci_{aggr}.csv")
            ci_df.to_csv(ci_csv, index=False)

            # symmetric matrix (point estimates)
            methods = sorted(set(list(df["method_A"]) + list(df["method_B"])))
            mat = pd.DataFrame(np.full((len(methods), len(methods)), np.nan), index=methods, columns=methods, dtype=float)
            for _, r in pair_stats.iterrows():
                a, b, v = r["method_A"], r["method_B"], r["point"]
                mat.loc[a, b] = mat.loc[b, a] = v

            mat_csv = os.path.join(outdir, f"{metric}_matrix_point.csv")
            mat.to_csv(mat_csv)

            # heatmap (masked lower triangle)
            M = mat.values.astype(float)
            np.fill_diagonal(M, 1.0)
            M = (M + M.T) / 2.0
            mask = np.tri(M.shape[0], k=-1, dtype=bool)
            M_masked = np.ma.array(M, mask=mask)

            fig, ax = plt.subplots(figsize=(6.5, 6.0))
            im = ax.imshow(M_masked, interpolation="nearest")
            labels = list(mat.columns)
            ax.set_xticks(np.arange(M.shape[1]))
            ax.set_yticks(np.arange(M.shape[0]))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if i <= j and not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)

            ax.set_title(f"Pairwise {metric.upper()} (patient-aggregated, {aggr})", fontsize=12, pad=12)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(metric.upper(), fontsize=10)
            ax.set_aspect("equal")
            plt.tight_layout()

            png_path = os.path.join(outdir, f"{metric}_heatmap_{aggr}.png")
            pdf_path = os.path.join(outdir, f"{metric}_heatmap_{aggr}.pdf")
            plt.savefig(png_path, dpi=600, bbox_inches="tight")
            plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
            plt.close(fig)

            # macro-average per method
            macro_rows = []
            for m in methods:
                parts = []
                for (A, B), g in scan2pat.groupby(["method_A", "method_B"]):
                    if m in (A, B):
                        parts.append(g[["patient_id", f"{metric}_patient"]].rename(columns={f"{metric}_patient": "val"}))
                if not parts:
                    continue
                merged = pd.concat(parts, axis=0, ignore_index=True)
                # per patient: aggregate across pairs that include method m using aggr
                per_patient = merged.groupby("patient_id", as_index=False)["val"].agg(lambda x: np.nanmedian(x) if aggr == 'median' else np.nanmean(x))
                v = per_patient["val"].to_numpy()
                v = v[~np.isnan(v)]
                n = v.size
                if n == 0:
                    point = lo = hi = np.nan
                elif n == 1:
                    point = float(np.mean(v))
                    lo = hi = np.nan
                else:
                    point = float(np.median(v) if aggr == "median" else np.mean(v))
                    lo, hi = bootstrap_ci(v, B=2000, alpha=0.05, use_median=(aggr == "median"))

                macro_rows.append({
                    "method": m,
                    f"macro_{metric}_{aggr}": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_patients": int(n),
                })

            macro_df = pd.DataFrame(macro_rows)
            macro_csv = os.path.join(outdir, f"macro_avg_{metric}_{aggr}.csv")
            macro_df.to_csv(macro_csv, index=False)
            print(f"[done] metric={metric} aggr={aggr} -> files saved in {outdir}")

print('All metrics processed.')