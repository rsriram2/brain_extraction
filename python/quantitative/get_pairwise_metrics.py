import pandas as pd
import numpy as np
import os
import glob
import itertools
import nibabel as nib
from typing import Tuple, Dict, List
from scipy.stats import bootstrap
import matplotlib.pyplot as plt

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

rows = []
for s in candidate_stems:
    imgs, masks = {}, {}
    aff, shape = None, None
    methods_here = sorted(all_stems[s])
    # load and check geometry
    ok = True
    for m in methods_here:
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
    # pairwise metrics (per scan)
    for A, B in itertools.combinations(methods_here, 2):
        a, b = masks[A], masks[B]
        tp, fp, fn, tn = confusion_2x2(a, b)
        rows.append({
            "patient_id": pid,
            "stem": s,
            "method_A": A, "method_B": B,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "dice": dice(tp, fp, fn),
            "iou": iou(tp, fp, fn),
            "accuracy": accuracy(tp, fp, fn, tn),
            "sensitivity_A_vs_B": sensitivity(tp, fn),
            "specificity_A_vs_B": specificity(tn, fp),
            "kappa": kappa(tp, fp, fn, tn),
            "n_vox": int(tp+fp+fn+tn)
        })

df = pd.DataFrame(rows)
print("unique patients:", df["patient_id"].nunique())
df.to_csv(OUT_CSV_RAW, index=False)
print(f"[raw] scans: {len(set(df['stem']))}  pairs: {len(df)}  rows saved -> {OUT_CSV_RAW}")

# ===== run pipeline for multiple metrics and aggregations =====
METRICS = ['dice', 'iou']
AGGRS = ['mean', 'median']

for metric in METRICS:
    for aggr in AGGRS:
        outdir = os.path.join(DATA_DIR, metric, aggr)
        os.makedirs(outdir, exist_ok=True)

        # collapse scans -> patient using chosen aggregation
        scan2pat = df.groupby(["patient_id", "method_A", "method_B"], as_index=False)[metric].agg(aggr)
        scan2pat.rename(columns={metric: f"{metric}_patient"}, inplace=True)

        # cohort-level point estimates (per pair)
        pair_stats = (scan2pat
                      .groupby(["method_A", "method_B"], as_index=False)[f"{metric}_patient"]
                      .agg(point=(lambda x: np.median(x) if aggr == "median" else np.mean(x)),
                           n_patients=("count")))

        pair_csv = os.path.join(outdir, f"pairwise_patient_agg_{metric}_{aggr}.csv")
        pair_stats.to_csv(pair_csv, index=False)

        # bootstrap CIs (patient-level)
        ci_rows = []
        use_median = (aggr == "median")
        for (A, B), g in scan2pat.groupby(["method_A", "method_B"]):
            v = g[f"{metric}_patient"].to_numpy()
            if v.size == 0:
                lo = hi = np.nan
                point = np.nan
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
            per_patient = merged.groupby("patient_id", as_index=False)["val"].agg(aggr)
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