import argparse
import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from typing import Dict, Tuple, List
from medpy.metric import binary as medpy_binary
from medpy.metric.binary import hd95 as medpy_hd95

def stem(p: str) -> str:
    base = os.path.basename(p)
    for suf in (".nii.gz", ".nii"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def patient_id_from_stem(s: str) -> str:
    return s.split("_")[0]


def load_mask_bool(path: str):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32) > 0
    return img, arr


def confusion_2x2(a, b):
    tp = np.count_nonzero(a & b)
    fp = np.count_nonzero(~a & b)
    fn = np.count_nonzero(a & ~b)
    tn = np.count_nonzero(~a & ~b)
    return int(tp), int(fp), int(fn), int(tn)


def dice_from_conf(tp, fp, fn):
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else np.nan


def iou_from_conf(tp, fp, fn):
    den = tp + fp + fn
    return (tp / den) if den else np.nan


def sensitivity_from_conf(tp, fn):
    den = tp + fn
    return (tp / den) if den else np.nan


def specificity_from_conf(tn, fp):
    den = tn + fp
    return (tn / den) if den else np.nan


def voxel_volume_ml_from_img(img: nib.Nifti1Image) -> float:
    sx, sy, sz = img.header.get_zooms()[:3]
    return float((sx * sy * sz) / 1000.0)

def build_file_index(dirs: List[str]) -> Dict[Tuple[str, str], str]:
    """Return mapping (patient_id, stem) -> filepath for all files under the provided dirs."""
    mapping = {}
    for d in dirs:
        if not d:
            continue
        for p in glob.glob(os.path.join(d, "*.nii*")):
            s = stem(p)
            pid = patient_id_from_stem(s)
            mapping[(pid, s)] = p
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Compare methods against a gold-standard set of masks")
    parser.add_argument('--out-dir', default='/users/rsriramb/brain_extraction', help='Output folder for CSVs (hardcoded default)')
    parser.add_argument('--threshold', type=float, default=0.97, help='Dice threshold for binary indicator')
    parser.add_argument('--save-csv', action='store_true', help='Save per-scan CSV and summaries')
    args = parser.parse_args()

    GOLD_DIR = '/dcs05/ciprian/smart/mistie_3/synthstrip_manual_segmentations/brain_extracted_double'
    GOLD_DIRS = [GOLD_DIR]

    BASE = '/dcs05/ciprian/smart/mistie_3/synthstrip_manual_segmentations'
    METHOD_DIRS: Dict[str, List[str]] = {
        'Brainchop': [os.path.join(BASE, 'brain_mask_brainchop')],
        'HD-CTBET': [os.path.join(BASE, 'brain_mask_hdctbet')],
        'SynthStrip': [os.path.join(BASE, 'brain_mask_synth')],
        'CTBET': [os.path.join(BASE, 'brain_mask_original')],
        'Robust-CTBET': [os.path.join(BASE, 'brain_mask')],
        'CT_BET': [os.path.join(BASE, 'brain_mask_ctbet')],
        'CTbet_Docker': [os.path.join(BASE, 'brain_mask_dockerctbet')]
    }

    if not os.path.isdir(GOLD_DIR):
        print(f'ERROR: gold directory does not exist: {GOLD_DIR}')
        return
    for m, dirs in list(METHOD_DIRS.items()):
        valid = [d for d in dirs if os.path.isdir(d)]
        if not valid:
            print(f'WARNING: method "{m}" has no valid directory (checked {dirs}) - it will be skipped')
            METHOD_DIRS.pop(m)
        else:
            METHOD_DIRS[m] = valid

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('Indexing gold masks...')
    gold_index = build_file_index(GOLD_DIRS)
    gold_keys = sorted(list(gold_index.keys()))
    print(f'Found {len(gold_keys)} gold masks (expected ~50).')

    # index method predictions
    method_indexes: Dict[str, Dict[Tuple[str, str], str]] = {}
    for m, ds in METHOD_DIRS.items():
        method_indexes[m] = build_file_index(ds)
        print(f'Method {m}: {len(method_indexes[m])} masks indexed')

    rows = []

    for pid, stem_id in gold_keys:
        gold_path = gold_index[(pid, stem_id)]
        try:
            gold_img, gold_arr = load_mask_bool(gold_path)
        except Exception as e:
            print(f'Failed to load gold {gold_path}: {e}')
            continue

        # compute reference voxel volume using gold header
        vox_ml = voxel_volume_ml_from_img(gold_img)
        icv_gold = float(gold_arr.sum() * vox_ml)

        for method_name, idx in method_indexes.items():
            if (pid, stem_id) not in idx:
                # method did not produce a mask for this gold scan
                continue
            pred_path = idx[(pid, stem_id)]
            try:
                pred_img, pred_arr = load_mask_bool(pred_path)
            except Exception as e:
                print(f'Failed to load pred {pred_path}: {e}')
                continue

            # enforce geometry consistency check (shape)
            if pred_arr.shape != gold_arr.shape:
                print(f'Geometry mismatch: gold {gold_path} vs pred {pred_path} -> skipping')
                continue

            tp, fp, fn, tn = confusion_2x2(gold_arr, pred_arr)

            dice_val = dice_from_conf(tp, fp, fn)
            iou_val = iou_from_conf(tp, fp, fn)
            sens = sensitivity_from_conf(tp, fn)
            spec = specificity_from_conf(tn, fp)

            # MSD/HD95: spacing from reference (gold_img)
            spacing = tuple(map(float, gold_img.header.get_zooms()[:3]))
            try:
                msd_val = float(medpy_binary.asd(gold_arr.astype(bool), pred_arr.astype(bool), voxelspacing=spacing))
            except Exception:
                msd_val = np.nan
            try:
                hd95_val = float(medpy_hd95(gold_arr.astype(bool), pred_arr.astype(bool), voxelspacing=spacing))
            except Exception:
                hd95_val = np.nan

            icv_pred = float(pred_arr.sum() * vox_ml)
            delta_icv_ml = icv_pred - icv_gold

            rows.append({
                'patient_id': pid,
                'stem': stem_id,
                'method_A': method_name,
                'method_B': 'GoldManual',
                'dice': dice_val,
                'iou': iou_val,
                'sensitivity': sens,
                'specificity': spec,
                'msd_mm': msd_val,
                'hd95_mm': hd95_val,
                'icv_ml_ref': icv_gold,
                'icv_ml_pred': icv_pred,
                'delta_icv_ml': delta_icv_ml,
                'gold_path': gold_path,
                'pred_path': pred_path,
            })

    df = pd.DataFrame(rows)
    print(f'Computed {len(df)} method-vs-gold rows')

    if df.empty:
        print('No rows computed. Exiting.')
        return

    # binary indicators
    df['above_thresh'] = (df['dice'] > args.threshold).astype(int)
    df['within_5ml'] = (df['delta_icv_ml'].abs() <= 5).astype(int)

    # ---------------- Aggregations ----------------
    coverage_rows = []
    gold_scan_count = len(gold_keys)
    for m in METHOD_DIRS.keys():
        preds = df[df['method_A'] == m]
        n_scans = preds[['patient_id', 'stem']].drop_duplicates().shape[0]
        n_subjects = preds['patient_id'].nunique()
        coverage_rows.append({'method': m, 'n_scans_with_pred': n_scans, 'n_subjects_with_pred': n_subjects, 'gold_scan_count': gold_scan_count})
    coverage_df = pd.DataFrame(coverage_rows)

    method_level = df.groupby('method_A').agg(n_scans_gt_97=('above_thresh', 'sum')).reset_index()
    method_level = method_level.rename(columns={'method_A': 'method'})
    method_level['n_scans_pred'] = gold_scan_count
    method_level['pct_of_gold_scans_gt_97'] = method_level['n_scans_gt_97'] / gold_scan_count * 100
    method_level = method_level.round(1)

    subj_per_patient = df.groupby(['patient_id', 'method_A'])['above_thresh'].mean().reset_index(name='pct_scans_above_0.97_per_patient')
    subject_level_mean = subj_per_patient.groupby('method_A')['pct_scans_above_0.97_per_patient'].mean().reset_index()
    subject_level_mean = subject_level_mean.rename(columns={'method_A': 'method', 'pct_scans_above_0.97_per_patient': 'subject_weighted_pct'}).round(1)

    subj_summary = subj_per_patient.copy()
    subj_summary['above_subj_thresh'] = (subj_summary['pct_scans_above_0.97_per_patient'] > 0.97).astype(int)
    subj_counts = subj_summary.groupby('method_A').agg(n_subjects=('patient_id', 'nunique'), n_subjects_gt_thresh=('above_subj_thresh', 'sum')).reset_index()
    subj_counts['pct_subjects_gt_thresh'] = (subj_counts['n_subjects_gt_thresh'] / subj_counts['n_subjects'] * 100).round(1)
    subj_counts = subj_counts.rename(columns={'method_A': 'method'})

    df_report = df.copy()
    df_report[['dice', 'iou', 'sensitivity', 'specificity', 'msd_mm', 'hd95_mm', 'icv_ml_ref', 'icv_ml_pred', 'delta_icv_ml']] = \
        df_report[['dice', 'iou', 'sensitivity', 'specificity', 'msd_mm', 'hd95_mm', 'icv_ml_ref', 'icv_ml_pred', 'delta_icv_ml']].round(3)

    method_level = method_level.round(1)

    print('\n=== Coverage per method ===')
    print(coverage_df.to_string(index=False))

    print('\n=== Method-level scan summary ===')
    print(method_level.to_string(index=False))

    print('\n=== Subject-level mean (subject-weighted) ===')
    print(subject_level_mean.to_string(index=False))

    print('\n=== Subjects summary (counts) ===')
    print(subj_counts.to_string(index=False))

    if args.save_csv:
        per_scan_csv = os.path.join(args.out_dir, 'method_vs_gold_per_scan_synth.csv')
        method_csv = os.path.join(args.out_dir, 'method_level_scan_summary_synth.csv')
        subject_mean_csv = os.path.join(args.out_dir, 'method_subject_weighted_mean_synth.csv')
        coverage_csv = os.path.join(args.out_dir, 'method_coverage_synth.csv')

        df_report.to_csv(per_scan_csv, index=False)
        method_level.to_csv(method_csv, index=False)
        subject_level_mean.to_csv(subject_mean_csv, index=False)
        coverage_df.to_csv(coverage_csv, index=False)
        print(f'CSVs saved to {args.out_dir}')

if __name__ == '__main__':
    main()
