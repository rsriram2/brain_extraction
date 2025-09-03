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
    # median Dice per stem across all method pairs
    per_stem = (df.groupby("stem", as_index=False)["dice"]
                  .median(numeric_only=True))
    clean = per_stem[per_stem["dice"] >= dice_thresh]["stem"].tolist()
    return set(clean)

def iter_rim_vals_for_stem(stem, ct_map, mask_maps, methods, r):
    ct_path = ct_map.get(stem)
    if ct_path is None:
        print(f"[calibrate] MISSING raw CT for stem={stem}")
        return []
    try:
        ct_img = nib.load(ct_path)
        ct = ct_img.get_fdata(dtype=np.float32)
    except Exception as e:
        print(f"[calibrate] ERROR loading raw CT for stem={stem}: {ct_path} -- {e}")
        return []
    selem = np.ones((3,3,3), dtype=bool)
    vals_list = []
    for m in methods:
        mp = mask_maps.get(m, {}).get(stem)
        if mp is None:
            print(f"[calibrate] MISSING mask '{m}' for stem={stem}")
            continue
        try:
            m_img = nib.load(mp)
            mask = m_img.get_fdata() > 0.5
        except Exception as e:
            print(f"[calibrate] ERROR loading mask '{m}' for stem={stem}: {mp} -- {e}")
            continue
        if mask.shape != ct.shape:
            print(f"[calibrate] SHAPE MISMATCH for stem={stem}, mask='{m}': mask.shape={mask.shape}, ct.shape={ct.shape}, mask_file={mp}, ct_file={ct_path}")
            continue
        rim = mask & ~binary_erosion(mask, structure=selem, iterations=r)
        if rim.any():
            vals_list.append(ct[rim])
    if not vals_list:
        return []
    return [np.concatenate(vals_list, axis=None)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise-raw", default=DEFAULT_PAIRWISE_RAW)
    ap.add_argument("--dice-thresh", type=float, default=0.98)
    ap.add_argument("--methods", nargs="+", default=["Robust-CTBET"],
                    help='Which masks to use for rim pooling. Use "ALL" to include all methods.')
    ap.add_argument("--rim-width", type=int, default=1)
    ap.add_argument("--q", type=float, default=0.997)
    ap.add_argument("--extra-qs", nargs="*", type=float, default=[0.995, 0.999])
    # bin parameters removed (we use sample-based percentiles)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["json", "csv", "both"], default="both",
                    help='Output format: json, csv, or both (default: both)')
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

    # Build per-stem raw rim-value lists (sample-based percentile)
    per_stem_vals = []
    n_vox_total = 0
    used_stems = []
    for stem in sorted(clean_stems):
        vals_chunks = iter_rim_vals_for_stem(stem, ct_map, mask_maps, methods, args.rim_width)
        if not vals_chunks:
            continue
        # iter_rim_vals_for_stem returns a list containing a single concatenated array
        vals = vals_chunks[0]
        if vals.size == 0:
            continue
        per_stem_vals.append(vals)
        used_stems.append(stem)
        n_vox_total += int(vals.size)

    if not per_stem_vals:
        print(f"[calibrate] Collected 0 rim voxels from candidate stems. Check earlier [calibrate] messages in the log for missing files or shape mismatches.")
        print(f"[calibrate] candidate_stems={len(clean_stems)} methods={methods}")
        raise SystemExit("No rim voxels collected. Check methods, stems, or shapes.")

    # Concatenate all rim voxels and compute sample-based percentiles
    all_vals = np.concatenate(per_stem_vals, axis=0)
    if all_vals.size == 0:
        raise SystemExit("No rim voxels collected after filtering. Check methods, stems, or shapes.")
    q_main = float(np.percentile(all_vals, args.q * 100.0))
    q_extras = {str(q): float(np.percentile(all_vals, q * 100.0)) for q in args.extra_qs}

    # Bootstrap CIs over stems
    rng = np.random.default_rng(0)
    qs_boot = []
    extras_boot = {str(q): [] for q in args.extra_qs}
    k = len(per_stem_vals)
    # Bootstrap by resampling stems with replacement and computing sample-based percentiles
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

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0: return [np.nan, np.nan]
        lo, hi = np.percentile(a, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        "n_clean_stems": len(used_stems),
        "n_rim_voxels": int(n_vox_total),
        "methods": methods,
        "dice_thresh": args.dice_thresh,
        "q_main": args.q,
        "bone_hu_threshold": float(q_main),
        "bone_hu_threshold_ci": ci(qs_boot),
        "alt_quantiles": {k: float(v) for k,v in q_extras.items()},
        "alt_quantiles_ci": {k: ci(v) for k,v in extras_boot.items()},
    # bins metadata removed (sample-based percentile)
    }
    # build a flat row (used for CSV output)
    row = {
        "n_clean_stems": out["n_clean_stems"],
        "n_rim_voxels": out["n_rim_voxels"],
        "methods": "|".join(out["methods"]),
        "dice_thresh": out["dice_thresh"],
        "q_main": out["q_main"],
        "bone_hu_threshold": out["bone_hu_threshold"],
        "bone_hu_threshold_ci_lo": out["bone_hu_threshold_ci"][0],
        "bone_hu_threshold_ci_hi": out["bone_hu_threshold_ci"][1],
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
