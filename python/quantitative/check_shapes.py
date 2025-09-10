import os
import glob
import pandas as pd
import nibabel as nib
import sys
from typing import Dict

# Mirror METHOD_DIRS from get_pairwise_metrics.py (keeps script standalone)
METHOD_DIRS = {
    'SynthStrip': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_synth',
    'Robust-CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask',
    'Brainchop': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_brainchop',
    'HD-CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_hdctbet',
    'CTbet_Docker': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_dockerctbet',
    'CTBET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_original',
    'CT_BET': '/dcs05/ciprian/smart/mistie_3/data/brain_mask_ctbet',
}


def stem(p: str) -> str:
    base = os.path.basename(p)
    for suf in ('.nii.gz', '.nii'):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def build_method_maps(method_dirs: Dict[str, str]):
    method_files = {m: glob.glob(os.path.join(d, '*.nii*')) for m, d in method_dirs.items()}
    method_maps = {m: {stem(p): p for p in files} for m, files in method_files.items()}
    return method_maps

out_csv = '/users/rsriramb/brain_extraction/results/quantitative/shape_checks.csv'

method_maps = build_method_maps(METHOD_DIRS)

all_stems = {}
for m, mp in method_maps.items():
    for s in mp.keys():
        all_stems.setdefault(s, set()).add(m)

candidate_stems = sorted(all_stems.keys())

meta = {m: {} for m in method_maps.keys()}
for m, mp in method_maps.items():
    for s, p in mp.items():
        if not os.path.exists(p):
            meta[m][s] = (None, 'missing')
            continue
        try:
            img = nib.load(p)
            meta[m][s] = (tuple(img.shape), '')
        except Exception as e:
            meta[m][s] = (None, str(e))

shapes_rows = []
good_stems = []
bad_stems = []
for s in candidate_stems:
    shapes = {}
    methods_present = sorted(all_stems.get(s, []))
    for m in methods_present:
        shape, err = meta.get(m, {}).get(s, (None, 'missing'))
        shapes.setdefault(shape, []).append(m)
        shapes_rows.append({'stem': s, 'method': m, 'shape': shape, 'read_error': err})
    non_none_shapes = [k for k in shapes.keys() if k is not None]
    # count how many methods actually provided a non-None shape (sum lengths of method lists)
    num_methods_with_shape = sum(len(v) for k, v in shapes.items() if k is not None and v)
    if len(set(non_none_shapes)) == 1 and num_methods_with_shape >= 2:
        good_stems.append(s)
    else:
        bad_stems.append({'stem': s, 'shapes': shapes})

# write shapes-only table and good stems
shapes_only_csv = os.path.splitext(out_csv)[0] + '_shapes_only.csv'
pd.DataFrame(shapes_rows).to_csv(shapes_only_csv, index=False)
good_csv = os.path.splitext(out_csv)[0] + '_good_stems.csv'
pd.Series(sorted(good_stems)).to_csv(good_csv, index=False, header=['stem'])

print(f"Shapes-only pass: shapes table -> {shapes_only_csv}; good stems -> {good_csv}")
print(f"Total stems: {len(candidate_stems)}  good by shape: {len(good_stems)}  bad: {len(bad_stems)}")
