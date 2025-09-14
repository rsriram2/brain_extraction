import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation
from collections import Counter

# -------------------
# Helpers
# -------------------
def root_from_path(p: str) -> str:
    return os.path.basename(p).replace('.nii.gz','').replace('.nii','')

def bone_percentage_from_mask(ct_img: nib.Nifti1Image, mask_path: str, method: str, r: int = 1,
                              bone_threshold: float = 1000.0) -> dict:
    """Calculate percentage of rim volume above bone threshold"""
    stem = root_from_path(ct_img.get_filename())
    subject_id = stem.split('_')[0]

    # Load mask
    m_img = nib.load(mask_path)
    m = m_img.get_fdata()
    mask_bin = m > 0.5

    # Check shape compatibility
    if mask_bin.shape != ct_img.shape:
        return {
            "subject_id": subject_id, 
            "stem": stem, 
            "method": method, 
            "r": r,
            "bone_percentage": np.nan
        }

    # Load CT data
    ct = ct_img.get_fdata(dtype=np.float32)
    
    # Create rim using erosion (same as original script)
    selem = np.ones((3, 3, 3), dtype=bool)
    rim = mask_bin & ~binary_erosion(mask_bin, structure=selem, iterations=r)
    
    # Get HU values in rim
    rim_vals = ct[rim]
    
    if rim_vals.size == 0:
        bone_percentage = np.nan
    else:
        # Calculate percentage above bone threshold
        n_above_threshold = (rim_vals >= bone_threshold).sum()
        bone_percentage = float((n_above_threshold / rim_vals.size) * 100)
    
    return {
        "subject_id": subject_id,
        "stem": stem,
        "method": method,
        "r": r,
        "bone_percentage": bone_percentage
    }

# -------------------
# I/O config (same as original script)
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
# Build stem→path maps (same as original script)
# -------------------
ct_files = glob.glob(os.path.join(ct_dir, '*ct.nii.gz'))
ct_map = {root_from_path(p): p for p in ct_files}
assert len(ct_map) == len(set(map(root_from_path, ct_files))), "Duplicate CT stems"

mask_maps = {}
for method, mdir in method_dirs.items():
    files = glob.glob(os.path.join(mdir, '*ct.nii.gz'))
    mask_maps[method] = {root_from_path(p): p for p in files}
    assert len(mask_maps[method]) == len(set(mask_maps[method].keys())), f"Duplicate stems for {method}"

# CT ∩ union(masks), not full-method intersection
mask_union = set().union(*[set(m.keys()) for m in mask_maps.values()]) if mask_maps else set()
common_stems = sorted(set(ct_map.keys()) & mask_union)

# -------------------
# Run analysis
# -------------------
bone_percentage_rows = []
counts_present = Counter()
counts_shape_mismatch = Counter()

for stem in common_stems:
    ct_path = ct_map[stem]
    ct_img = nib.load(ct_path)

    # Calculate bone percentage for each available method
    for method, mask_map in mask_maps.items():
        mp = mask_map.get(stem)
        if mp is None:
            continue
        
        row = bone_percentage_from_mask(ct_img, mp, method=method, r=1)
        bone_percentage_rows.append(row)
        
        if np.isnan(row["bone_percentage"]):
            counts_shape_mismatch[method] += 1
        else:
            counts_present[method] += 1

# -------------------
# Save results
# -------------------
bone_percentage_df = pd.DataFrame(bone_percentage_rows)
output_path = os.path.join(out_dir, 'rim_bone_percentage.csv')
bone_percentage_df.to_csv(output_path, index=False)

print(f"Saved {len(bone_percentage_df)} bone percentage measurements to {output_path}")
print("Counts per method:")
for method in method_dirs.keys():
    print(f"  {method}: {counts_present[method]} usable, {counts_shape_mismatch[method]} shape mismatches")