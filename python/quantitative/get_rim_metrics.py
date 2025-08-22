import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion

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
    return {
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
                          bone: float = 70.0, air: float = -200.0) -> dict:
    # Item 6: load CT once and reuse. Mask loaded here.
    ct = ct_img.get_fdata(dtype=np.float32)  # already scaled to HU
    vox_ml = voxel_volume_ml_from_img(ct_img)

    m_img = nib.load(mask_path)
    m = m_img.get_fdata()  # may be float
    mask_bin = m > 0.5      # Item 3: binarize via threshold

    if mask_bin.shape != ct_img.shape:
        raise ValueError(f"Mask shape {mask_bin.shape} != CT shape {ct_img.shape} for {mask_path}")

    # Item 4: isotropic 3x3x3 structuring element
    selem = np.ones((3, 3, 3), dtype=bool)
    rim = mask_bin & ~binary_erosion(mask_bin, structure=selem, iterations=r)

    vals = ct[rim]
    stem = root_from_path(ct_img.get_filename())

    if vals.size == 0:
        return {
            "stem": stem, "method": method, "r": r,
            "rim_vol_ml": 0.0, "p95": np.nan, "p99": np.nan,
            "vol_bone_ml": 0.0, "vol_air_ml": 0.0,
            "frac_bone": np.nan, "frac_air": np.nan,
            "n_rim_vox": 0,
            "hu_slope": getattr(ct_img.dataobj, "slope", None),
            "hu_intercept": getattr(ct_img.dataobj, "inter", None),
        }

    rim_vol_ml  = float(rim.sum() * vox_ml)
    vol_bone_ml = float((vals >= bone).sum() * vox_ml)
    vol_air_ml  = float((vals <= air).sum() * vox_ml)
    frac_bone   = float(vol_bone_ml / rim_vol_ml) if rim_vol_ml > 0 else np.nan
    frac_air    = float(vol_air_ml / rim_vol_ml) if rim_vol_ml > 0 else np.nan

    return {
        "stem": stem, "method": method, "r": r,
        "rim_vol_ml": rim_vol_ml,
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "vol_bone_ml": vol_bone_ml,
        "vol_air_ml": vol_air_ml,
        "frac_bone": frac_bone,
        "frac_air": frac_air,
        "n_rim_vox": int(vals.size),
        "hu_slope": getattr(ct_img.dataobj, "slope", None),
        "hu_intercept": getattr(ct_img.dataobj, "inter", None)
    }

# -------------------
# I/O config
# -------------------
ct_dir     = '/dcs05/ciprian/smart/mistie_3/data/nifti'
synth_dir  = '/dcs05/ciprian/smart/mistie_3/data/brain_mask_synth'
robust_dir = '/dcs05/ciprian/smart/mistie_3/data/brain_mask'
out_dir    = '/users/rsriramb/ichseg_rep'

os.makedirs(out_dir, exist_ok=True)  # Item 7

# -------------------
# Build stem→path maps (Item 1)
# -------------------
ct_files     = glob.glob(os.path.join(ct_dir, '*ct.nii.gz'))
synth_files  = glob.glob(os.path.join(synth_dir, '*ct.nii.gz'))
robust_files = glob.glob(os.path.join(robust_dir, '*ct.nii.gz'))

ct_map     = {root_from_path(p): p for p in ct_files}
synth_map  = {root_from_path(p): p for p in synth_files}
robust_map = {root_from_path(p): p for p in robust_files}

assert len(ct_map) == len(set(map(root_from_path, ct_files))), "Duplicate CT stems"
assert len(synth_map) == len(set(map(root_from_path, synth_files))), "Duplicate synth stems"
assert len(robust_map) == len(set(map(root_from_path, robust_files))), "Duplicate robust stems"

# Intersection where both masks exist
common_stems = sorted(set(ct_map.keys()) & set(synth_map.keys()) & set(robust_map.keys()))

# -------------------
# Run
# -------------------
meta_rows = []
metric_rows = []

for stem in common_stems:
    ct_path = ct_map[stem]
    ct_img = nib.load(ct_path)  # Item 6: load once

    # Metadata per CT (Item 9: record slope/intercept)
    meta_rows.append(get_ct_metadata(ct_img, stem))

    # Metrics per method (Item 8: add 'method' and 'r')
    metric_rows.append(rim_metrics_from_mask(ct_img, synth_map[stem], method="synth", r=1))
    metric_rows.append(rim_metrics_from_mask(ct_img, robust_map[stem], method="robust", r=1))

# -------------------
# Save
# -------------------
all_metadata = pd.DataFrame(meta_rows)
all_metrics  = pd.DataFrame(metric_rows)

all_metadata.to_csv(os.path.join(out_dir, 'all_metadata.csv'), index=False)
all_metrics.to_csv(os.path.join(out_dir, 'rim_metrics_r1.csv'), index=False)

print(f"Saved {len(all_metadata)} CT metadata rows and {len(all_metrics)} metric rows.")
