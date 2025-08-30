import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation

# -------------------
# Helpers
# -------------------
def root_from_path(p: str) -> str:
    return os.path.basename(p).replace('.nii.gz','').replace('.nii','')

def voxel_volume_ml_from_img(img: nib.Nifti1Image) -> float:
    # Item 5: use header.get_zooms (mm). 1 mL = 1000 mm^3
    sx, sy, sz = img.header.get_zooms()[:3]
    return float((sx * sy * sz) / 1000.0)

def get_ct_metadata(ct_img: nib.Nifti1Image, stem: str) -> dict:
    data = ct_img.get_fdata(dtype=np.float32)  # applies slope/intercept
    vox_ml = voxel_volume_ml_from_img(ct_img)
    slope = getattr(ct_img.dataobj, "slope", None)
    inter = getattr(ct_img.dataobj, "inter", None)
    subject_id = stem.split('_')[0]
    return {
        "subject_id": subject_id,
        "stem": stem,
        "voxel_volume_mL": vox_ml,
        "HU_min": float(np.nanmin(data)),
        "HU_max": float(np.nanmax(data)),
        "hu_slope": None if slope is None else float(slope),
        "hu_intercept": None if inter is None else float(inter),
        "shape_x": int(ct_img.shape[0]),
        "shape_y": int(ct_img.shape[1]),
        "shape_z": int(ct_img.shape[2]),
    }

def rim_metrics_from_mask(ct_img: nib.Nifti1Image, mask_path: str, method: str, r: int = 1,
                          bone: float = 70.0, air: float = -200.0, brain_min: float = 20.0, brain_max: float = 60.0) -> dict:
    # Item 6: load CT once and reuse. Mask loaded here.
    ct = ct_img.get_fdata(dtype=np.float32)  # already scaled to HU
    vox_ml = voxel_volume_ml_from_img(ct_img)
    stem = root_from_path(ct_img.get_filename())
    subject_id = stem.split('_')[0]

    m_img = nib.load(mask_path)

    # 2. Check for affine equality
    if not np.allclose(ct_img.affine, m_img.affine):
        print(f"Affine mismatch for CT: {stem}, method: {method}. Skipping.")
        return {
            "subject_id": subject_id, "stem": stem, "method": method, "r": r,
            "rim_vol_ml": np.nan, "p95": np.nan, "p99": np.nan,
            "vol_bone_ml": np.nan, "vol_air_ml": np.nan,
            "frac_bone": np.nan, "frac_air": np.nan,
            "n_rim_vox": np.nan,
            "outer_vol_ml": np.nan, "frac_brain_out": np.nan,
            "hu_slope": getattr(ct_img.dataobj, "slope", None),
            "hu_intercept": getattr(ct_img.dataobj, "inter", None)
        }

    m = m_img.get_fdata()  # may be float
    mask_bin = m > 0.5      # Item 3: binarize via threshold

    if mask_bin.shape != ct_img.shape:
        raise ValueError(f"Mask shape {mask_bin.shape} != CT shape {ct_img.shape} for {mask_path}")

    # Item 4: isotropic 3x3x3 structuring element
    selem = np.ones((3, 3, 3), dtype=bool)
    
    # Inner rim (inclusion)
    rim = mask_bin & ~binary_erosion(mask_bin, structure=selem, iterations=r)
    
    # 1. Outer rim (omission)
    outer_rim = binary_dilation(mask_bin, structure=selem, iterations=r) & ~mask_bin

    vals = ct[rim]
    vals_outer = ct[outer_rim]

    if vals.size == 0:
        return {
            "subject_id": subject_id, "stem": stem, "method": method, "r": r,
            "rim_vol_ml": 0.0, "p95": np.nan, "p99": np.nan,
            "vol_bone_ml": 0.0, "vol_air_ml": 0.0,
            "frac_bone": np.nan, "frac_air": np.nan,
            "n_rim_vox": 0,
            "outer_vol_ml": float(outer_rim.sum() * vox_ml),
            "frac_brain_out": float(np.mean((vals_outer >= brain_min) & (vals_outer <= brain_max))) if vals_outer.size > 0 else np.nan,
            "hu_slope": getattr(ct_img.dataobj, "slope", None),
            "hu_intercept": getattr(ct_img.dataobj, "inter", None),
        }

    rim_vol_ml  = float(rim.sum() * vox_ml)
    vol_bone_ml = float((vals >= bone).sum() * vox_ml)
    vol_air_ml  = float((vals <= air).sum() * vox_ml)
    frac_bone   = float(vol_bone_ml / rim_vol_ml) if rim_vol_ml > 0 else np.nan
    frac_air    = float(vol_air_ml / rim_vol_ml) if rim_vol_ml > 0 else np.nan
    
    outer_vol_ml = float(outer_rim.sum() * vox_ml)
    frac_brain_out = float(np.mean((vals_outer >= brain_min) & (vals_outer <= brain_max))) if vals_outer.size > 0 else np.nan

    return {
        "subject_id": subject_id, "stem": stem, "method": method, "r": r,
        "rim_vol_ml": rim_vol_ml,
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "vol_bone_ml": vol_bone_ml,
        "vol_air_ml": vol_air_ml,
        "frac_bone": frac_bone,
        "frac_air": frac_air,
        "n_rim_vox": int(vals.size),
        "outer_vol_ml": outer_vol_ml,
        "frac_brain_out": frac_brain_out,
        "hu_slope": getattr(ct_img.dataobj, "slope", None),
        "hu_intercept": getattr(ct_img.dataobj, "inter", None)
    }

# -------------------
# I/O config
# -------------------

# Directories for each method
ct_dir = '/dcs05/ciprian/smart/mistie_3/data/nifti'
method_dirs = {
    'SynthStrip': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_synth',
    'Robust-CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask',
    'Brainchop': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_brainchop',
    'HD-CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_hdctbet',
    'CTbet_Docker': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_dockerctbet',
    'CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_original',
    'CT_BET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_ctbet',
}
out_dir = '/users/rsriramb/brain_extraction/results/quantitative'

os.makedirs(out_dir, exist_ok=True)

# -------------------
# Build stem→path maps (Item 1)
# -------------------

ct_files = glob.glob(os.path.join(ct_dir, '*ct.nii.gz'))
ct_map = {root_from_path(p): p for p in ct_files}
assert len(ct_map) == len(set(map(root_from_path, ct_files))), "Duplicate CT stems"

# Build mask maps for all methods
mask_maps = {}
for method, mdir in method_dirs.items():
    files = glob.glob(os.path.join(mdir, '*ct.nii.gz'))
    mask_maps[method] = {root_from_path(p): p for p in files}
    assert len(mask_maps[method]) == len(set(mask_maps[method].keys())), f"Duplicate stems for {method}"

# Find intersection of stems present in CT and all mask methods
common_stems = set(ct_map.keys())
for m in mask_maps.values():
    common_stems &= set(m.keys())
common_stems = sorted(common_stems)

# -------------------
# Run
# -------------------
meta_rows = []
metric_rows = []


for stem in common_stems:
    ct_path = ct_map[stem]
    ct_img = nib.load(ct_path)

    # Metadata per CT (only once per stem)
    meta_rows.append(get_ct_metadata(ct_img, stem))

    # Metrics for all methods
    for method, mask_map in mask_maps.items():
        metric_rows.append(rim_metrics_from_mask(ct_img, mask_map[stem], method=method, r=1))

# -------------------
# Save
# -------------------
all_metadata = pd.DataFrame(meta_rows)
all_metrics  = pd.DataFrame(metric_rows)

# 3. Subject-level aggregation
subj_metrics = all_metrics.groupby(['subject_id', 'method']).mean().reset_index()

all_metadata.to_csv(os.path.join(out_dir, 'all_metadata.csv'), index=False)
all_metrics.to_csv(os.path.join(out_dir, 'rim_metrics_r1_scan.csv'), index=False)
subj_metrics.to_csv(os.path.join(out_dir, 'rim_metrics_r1_subj.csv'), index=False)

print(f"Saved {len(all_metadata)} CT metadata rows, {len(all_metrics)} scan-level metric rows, and {len(subj_metrics)} subject-level metric rows.")
