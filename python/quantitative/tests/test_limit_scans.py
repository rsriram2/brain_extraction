import tempfile
import os
import numpy as np
import nibabel as nib
from pathlib import Path

from python.quantitative.get_pairwise_metrics import main


def make_nifti(path, arr, affine=None):
    affine = np.eye(4) if affine is None else affine
    img = nib.Nifti1Image(arr.astype(np.uint8), affine)
    nib.save(img, path)


def test_limit_30_scans(tmp_path):
    # create two method directories
    m1 = tmp_path / 'M1'
    m2 = tmp_path / 'M2'
    m1.mkdir()
    m2.mkdir()

    stems = [f"TEST-{i:04d}_20200101_0000_ct" for i in range(40)]

    method_files = {'M1': [], 'M2': []}
    # create 40 stems but we'll run with limit_scans=30
    for s in stems:
        # simple 5x5 mask with a single voxel set for variety
        arr = np.zeros((5,5,1), dtype=np.uint8)
        arr[0,0,0] = 1
        p1 = m1 / (s + '.nii.gz')
        p2 = m2 / (s + '.nii.gz')
        make_nifti(str(p1), arr)
        make_nifti(str(p2), arr)
        method_files['M1'].append(str(p1))
        method_files['M2'].append(str(p2))

    out_csv = str(tmp_path / 'out.csv')
    # run main with limit 30
    main(limit_scans=30, method_files=method_files, out_csv=out_csv)

    # read output and assert <= 30 unique stems
    import pandas as pd
    df = pd.read_csv(out_csv)
    unique_stems = df['stem'].unique()
    assert len(unique_stems) <= 30
