#!/usr/bin/env python3
import argparse, os, glob, json
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import binary_erosion

# Defaults match your repo layout
DEFAULT_PAIRWISE_RAW = "/users/rsriramb/brain_extraction/results/quantitative/pairwise_2x2_metrics_all_scans.csv"
CT_DIR = "/dcs05/ciprian/smart/mistie_3/data/nifti"
METHOD_DIRS = {
    "SynthStrip":     "/dcs05/ciprian/smart/mistie_3/data/brain_mask_synth",
    "Robust-CTBET":   "/dcs05/ciprian/smart/mistie_3/data/brain_mask",
    "Brainchop":      "/dcs05/ciprian/smart/mistie_3/data/brain_mask_brainchop",
    "HD-CTBET":       "/dcs05/ciprian/smart/mistie_3/data/brain_mask_hdctbet",
    "CTbet_Docker":   "/dcs05/ciprian/smart/mistie_3/data/brain_mask_dockerctbet",
    "CTBET":          "/dcs05/ciprian/smart/mistie_3/data/brain_mask_original",
    "CT_BET":         "/dcs05/ciprian/smart/mistie_3/data/brain_mask_ctbet",
}

def root(p): 
    return os.path.basename(p).replace(".nii.gz","").replace(".nii","")

def voxel_ml(img): 
    sx, sy, sz = img.header.get_zooms()[:3]
    return float((sx * sy * sz) / 1000.0)  # mm^3 -> mL

def load_maps():
    ct_files = glob.glob(os.path.join(CT_DIR, "*ct.nii.gz"))
    ct_map = {root(p): p for p in ct_files}
    mask_maps = {m: {root(p): p for p in glob.glob(os.path.join(d, "*ct.nii.gz"))}
                 for m,d in METHOD_DIRS.items()}
    return ct_map, mask_maps

def get_clean_stems(pairwise_raw_csv, dice_thresh):
    df = pd.read_csv(pairwise_raw_csv)
    # compute the minimum pairwise Dice per stem across all method pairs
    # this enforces that all method-method comparisons for that stem meet the threshold
    per_stem_min = df.groupby('stem', as_index=False)['dice'].min()
    clean = per_stem_min[per_stem_min['dice'] >= dice_thresh]['stem'].tolist()
    return set(clean)


def patient_id_from_stem(stem: str) -> str:
    return stem.split('_')[0]

def iter_rim_vals_for_stem_both(stem, ct_map, mask_maps, methods, r):
    """Return two arrays for a stem:
    - dup_vals: concatenated rim values from each method (may double-count overlapping voxels)
    - union_vals: rim values from the union of all method masks for this stem (no double-counting)
    If either result is empty, return None for that slot.
    """
    ct_path = ct_map.get(stem)
    if ct_path is None:
        print(f"[calibrate] MISSING raw CT for stem={stem}")
        return None, None
    try:
        ct_img = nib.load(ct_path)
        ct = ct_img.get_fdata(dtype=np.float32)
    except Exception as e:
        print(f"[calibrate] ERROR loading raw CT for stem={stem}: {ct_path} -- {e}")
        return None, None
    selem = np.ones((3,3,3), dtype=bool)

    dup_vals_list = []
    union_mask = None
    any_mask_loaded = False
    for m in methods:
        mp = mask_maps.get(m, {}).get(stem)
        if mp is None:
            print(f"[calibrate] MISSING mask '{m}' for stem={stem}")
            continue
        try:
            m_img = nib.load(mp)
            mask = m_img.get_fdata() > 0.5
            any_mask_loaded = True
        except Exception as e:
            print(f"[calibrate] ERROR loading mask '{m}' for stem={stem}: {mp} -- {e}")
            continue
        if mask.shape != ct.shape:
            print(f"[calibrate] SHAPE MISMATCH for stem={stem}, mask='{m}': mask.shape={mask.shape}, ct.shape={ct.shape}, mask_file={mp}, ct_file={ct_path}")
            continue
        # duplicate (per-method) rims
        rim = mask & ~binary_erosion(mask, structure=selem, iterations=r)
        if rim.any():
            dup_vals_list.append(ct[rim])
        # build union mask
        if union_mask is None:
            union_mask = mask.copy()
        else:
            union_mask = union_mask | mask

    # produce arrays
    dup_arr = None
    if dup_vals_list:
        dup_arr = np.concatenate(dup_vals_list, axis=0)

    union_arr = None
    if any_mask_loaded and union_mask is not None:
        union_rim = union_mask & ~binary_erosion(union_mask, structure=selem, iterations=r)
        if union_rim.any():
            union_arr = ct[union_rim]

    return dup_arr, union_arr


def iter_rim_method_summary(stem, ct_map, mask_maps, methods, r, summary='max'):
    """Return a 1D numpy array of length len(methods) with a per-method summary HU value from the rim.
    summary options: 'max', 'p99' (99th percentile), 'median'. Missing method -> np.nan
    """
    ct_path = ct_map.get(stem)
    if ct_path is None:
        print(f"[calibrate] MISSING raw CT for stem={stem}")
        return None
    try:
        ct_img = nib.load(ct_path)
        ct = ct_img.get_fdata(dtype=np.float32)
    except Exception as e:
        print(f"[calibrate] ERROR loading raw CT for stem={stem}: {ct_path} -- {e}")
        return None

    selem = np.ones((3,3,3), dtype=bool)
    vals = []
    any_loaded = False
    for m in methods:
        mp = mask_maps.get(m, {}).get(stem)
        if mp is None:
            print(f"[calibrate] MISSING mask '{m}' for stem={stem}")
            vals.append(np.nan)
            continue
        try:
            m_img = nib.load(mp)
            mask = m_img.get_fdata() > 0.5
            any_loaded = True
        except Exception as e:
            print(f"[calibrate] ERROR loading mask '{m}' for stem={stem}: {mp} -- {e}")
            vals.append(np.nan)
            continue
        if mask.shape != ct.shape:
            print(f"[calibrate] SHAPE MISMATCH for stem={stem}, mask='{m}': mask.shape={mask.shape}, ct.shape={ct.shape}")
            vals.append(np.nan)
            continue
        rim = mask & ~binary_erosion(mask, structure=selem, iterations=r)
        if not rim.any():
            vals.append(np.nan)
            continue
        rim_vals = ct[rim]
        if summary == 'max':
            vals.append(float(np.nanmax(rim_vals)))
        elif summary == 'p99':
            vals.append(float(np.nanpercentile(rim_vals, 99.0)))
        elif summary == 'median':
            vals.append(float(np.nanmedian(rim_vals)))
        else:
            vals.append(float(np.nanmax(rim_vals)))

    if not any_loaded:
        return None
    return np.array(vals, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise-raw", default=DEFAULT_PAIRWISE_RAW)
    ap.add_argument("--dice-thresh", type=float, default=0.98)
    ap.add_argument("--methods", nargs="+", default=["Robust-CTBET"],
                    help='Which masks to use for rim pooling. Use "ALL" to include all methods.')
    ap.add_argument("--rim-width", type=int, default=1)
    ap.add_argument("--q", type=float, default=0.997)
    ap.add_argument("--extra-qs", nargs="*", type=float, default=[0.995, 0.999])
    ap.add_argument("--summary", choices=["max", "p99", "median"], default="max",
                    help="Per-method rim summary to pool across methods (default: max)")
    # bin parameters removed (we use sample-based percentiles)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help='Number of bootstrap replicates over stems; set 0 to disable (default: 0)')
    ap.add_argument("--unique-rim", dest="unique_rim", action="store_true", default=True,
                    help="Use union of method masks per stem to avoid double-counting (default: true).")
    ap.add_argument("--no-unique-rim", dest="unique_rim", action="store_false",
                    help="Do not deduplicate rim voxels across methods; use concatenated per-method rims for primary result.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["json", "csv", "both"], default="both",
                    help='Output format: json, csv, or both (default: both)')
    ap.add_argument("--save-pooled", default=None,
                    help='Optional path to save the pooled rim HU values (CSV with one column "hu").')
    args = ap.parse_args()

    ct_map, mask_maps = load_maps()
    methods = list(METHOD_DIRS.keys()) if args.methods == ["ALL"] else args.methods

    clean_stems = get_clean_stems(args.pairwise_raw, args.dice_thresh)
    # diagnostic: show how many candidate stems were selected and a small sample
    try:
        sample_stems = sorted(clean_stems)[:20]
    except Exception:
        sample_stems = []
    print(f"[calibrate] candidate_stems={len(clean_stems)} sample(<=20)={sample_stems} methods={methods}")

    # Build per-stem raw rim-value lists for both analyses:
    # - dup_per_stem_vals: concatenated per-method rims (may double-count overlapping voxels)
    # - union_per_stem_vals: union of methods then rim (no double-counting)
    dup_per_stem_vals = []
    union_per_stem_vals = []
    n_vox_dup = 0
    n_vox_union = 0
    used_stems = []
    for stem in sorted(clean_stems):
        dup_arr, union_arr = iter_rim_vals_for_stem_both(stem, ct_map, mask_maps, methods, args.rim_width)
        if dup_arr is None and union_arr is None:
            continue
        # track stems where at least one method/load succeeded
        used_stems.append(stem)
        if dup_arr is not None and dup_arr.size > 0:
            dup_per_stem_vals.append(dup_arr)
            n_vox_dup += int(dup_arr.size)
        if union_arr is not None and union_arr.size > 0:
            union_per_stem_vals.append(union_arr)
            n_vox_union += int(union_arr.size)

    if not dup_per_stem_vals and not union_per_stem_vals:
        print(f"[calibrate] Collected 0 rim voxels from candidate stems. Check earlier [calibrate] messages in the log for missing files or shape mismatches.")
        print(f"[calibrate] candidate_stems={len(clean_stems)} methods={methods}")
        raise SystemExit("No rim voxels collected. Check methods, stems, or shapes.")

    def compute_quantiles_and_bootstrap(per_stem_vals, label):
        """Given a list of per-stem 1D arrays, compute main and extra quantiles and bootstrap CIs over stems."""
        if not per_stem_vals:
            return {
                'q_val': np.nan,
                'q_extras': {str(q): np.nan for q in args.extra_qs},
                'q_ci': [np.nan, np.nan],
                'extras_ci': {str(q): [np.nan, np.nan] for q in args.extra_qs},
                'n_vox': 0,
                'n_stems': 0,
            }

        all_vals = np.concatenate(per_stem_vals, axis=0)
        q_val = float(np.percentile(all_vals, args.q * 100.0))
        q_extras = {str(q): float(np.percentile(all_vals, q * 100.0)) for q in args.extra_qs}

        rng = np.random.default_rng(0)
        qs_boot = []
        extras_boot = {str(q): [] for q in args.extra_qs}
        k = len(per_stem_vals)
        for _ in range(args.bootstrap):
            idx = rng.integers(0, k, size=k)
            sampled = np.concatenate([per_stem_vals[i] for i in idx], axis=0)
            if sampled.size == 0:
                qs_boot.append(np.nan)
                for q in args.extra_qs:
                    extras_boot[str(q)].append(np.nan)
                continue
            qs_boot.append(float(np.percentile(sampled, args.q * 100.0)))
            for q in args.extra_qs:
                extras_boot[str(q)].append(float(np.percentile(sampled, q * 100.0)))

        def ci_local(a):
            a = np.asarray(a, dtype=float)
            a = a[~np.isnan(a)]
            if a.size == 0: return [np.nan, np.nan]
            lo, hi = np.percentile(a, [2.5, 97.5])
            return [float(lo), float(hi)]

        return {
            'q_val': q_val,
            'q_extras': q_extras,
            'q_ci': ci_local(qs_boot),
            'extras_ci': {k: ci_local(v) for k,v in extras_boot.items()},
            'n_vox': int(all_vals.size),
            'n_stems': k,
        }

    dup_stats = compute_quantiles_and_bootstrap(dup_per_stem_vals, 'dup')
    union_stats = compute_quantiles_and_bootstrap(union_per_stem_vals, 'union')

    # choose primary based on args.unique_rim
    primary_stats = union_stats if args.unique_rim else dup_stats

    # prepare pooled_vals for potential saving/inspection
    if args.unique_rim:
        pooled_vals = np.concatenate(union_per_stem_vals, axis=0) if union_per_stem_vals else np.array([])
    else:
        pooled_vals = np.concatenate(dup_per_stem_vals, axis=0) if dup_per_stem_vals else np.array([])

    # Optionally save pooled HU values to CSV for plotting
    if args.save_pooled:
        csv_path = args.save_pooled
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            df_pooled = pd.DataFrame({'hu': pooled_vals})
            df_pooled.to_csv(csv_path, index=False)
            print(f"Wrote pooled HU values ({pooled_vals.size} rows) to: {csv_path}")
        except Exception as e:
            print(f"Failed to write pooled HU CSV to {csv_path}: {e}")

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0: return [np.nan, np.nan]
        lo, hi = np.percentile(a, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        "n_clean_stems": len(used_stems),
    "n_rim_voxels_dup": int(primary_stats.get('n_vox', 0)),
    "n_rim_voxels_union": int(primary_stats.get('n_vox', 0)),
        "methods": methods,
        "dice_thresh": args.dice_thresh,
        "q_main": args.q,
        # primary
        "bone_hu_threshold": float(primary_stats['q_val']) if not np.isnan(primary_stats['q_val']) else None,
        "bone_hu_threshold_ci": primary_stats['q_ci'],
        # duplicate/union fields set to primary (we pooled per-method summaries)
        "bone_hu_threshold_dup": float(primary_stats['q_val']) if not np.isnan(primary_stats['q_val']) else None,
        "bone_hu_threshold_dup_ci": primary_stats['q_ci'],
        "bone_hu_threshold_union": float(primary_stats['q_val']) if not np.isnan(primary_stats['q_val']) else None,
        "bone_hu_threshold_union_ci": primary_stats['q_ci'],
        # alternate quantiles and CI
        "alt_quantiles": {k: float(v) for k,v in primary_stats['q_extras'].items()},
        "alt_quantiles_ci": {k: v for k,v in primary_stats['extras_ci'].items()},
    }
    # build a flat row (used for CSV output)
    row = {
        "n_clean_stems": out["n_clean_stems"],
        "n_rim_voxels_dup": out["n_rim_voxels_dup"],
        "n_rim_voxels_union": out["n_rim_voxels_union"],
        "methods": "|".join(out["methods"]),
        "dice_thresh": out["dice_thresh"],
        "q_main": out["q_main"],
        "bone_hu_threshold": out["bone_hu_threshold"],
        "bone_hu_threshold_ci_lo": None if out["bone_hu_threshold_ci"] is None else out["bone_hu_threshold_ci"][0],
        "bone_hu_threshold_ci_hi": None if out["bone_hu_threshold_ci"] is None else out["bone_hu_threshold_ci"][1],
        "bone_hu_threshold_dup": out["bone_hu_threshold_dup"],
        "bone_hu_threshold_dup_ci_lo": None if out.get("bone_hu_threshold_dup_ci") is None else out["bone_hu_threshold_dup_ci"][0],
        "bone_hu_threshold_dup_ci_hi": None if out.get("bone_hu_threshold_dup_ci") is None else out["bone_hu_threshold_dup_ci"][1],
        "bone_hu_threshold_union": out["bone_hu_threshold_union"],
        "bone_hu_threshold_union_ci_lo": None if out.get("bone_hu_threshold_union_ci") is None else out["bone_hu_threshold_union_ci"][0],
        "bone_hu_threshold_union_ci_hi": None if out.get("bone_hu_threshold_union_ci") is None else out["bone_hu_threshold_union_ci"][1],
    }

    # include alternate quantiles and their CIs
    for k, v in out.get("alt_quantiles", {}).items():
        col = f"alt_q_{k}"
        row[col] = v
        ci_key = out.get("alt_quantiles_ci", {}).get(k, [None, None])
        row[f"alt_q_{k}_ci_lo"] = ci_key[0]
        row[f"alt_q_{k}_ci_hi"] = ci_key[1]

    # include bins
    bins = out.get("bins", {})
    row["bins_min_hu"] = bins.get("min_hu")
    row["bins_max_hu"] = bins.get("max_hu")
    row["bins_bin_width"] = bins.get("bin_width")

    # Write JSON if requested
    if args.format in ("json", "both"):
        if args.out.lower().endswith('.json'):
            json_out = args.out
        else:
            json_out = args.out + '.json'
        os.makedirs(os.path.dirname(json_out), exist_ok=True) if os.path.dirname(json_out) else None
        with open(json_out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Wrote JSON: {json_out}")

    # Write CSV if requested
    if args.format in ("csv", "both"):
        if args.out.lower().endswith('.csv'):
            csv_out = args.out
        else:
            # if out ends with .json, replace suffix; otherwise append .csv
            if args.out.lower().endswith('.json'):
                csv_out = args.out[:-5] + '.csv'
            else:
                csv_out = args.out + '.csv'
        os.makedirs(os.path.dirname(csv_out), exist_ok=True) if os.path.dirname(csv_out) else None
        df = pd.DataFrame([row])
        df.to_csv(csv_out, index=False)
        print(f"Wrote CSV: {csv_out}")

if __name__ == "__main__":
    main()
