"""
    Ahmed Al Sunbati     Jul 13th, 2025
    Description: Cleaning the data folder, reorganizing and using the following
    format that follows BIDS
    BIDSdffr    -|
                 |-sub-01 -|
                           |- anat -|
                                    |- sub-01_T1w.nii.gz
                           |- func -|
                                    |- sub-01_task-StudyTest_bold.nii.gz
"""

import os
import shutil
import nibabel as nib
import numpy as np
from glob import glob
import time

SRC_ROOT = './ds001745'
DST_ROOT = './BIDSdffr'

def make_bids_dirs(subj_id):
    anat_dir = os.path.join(DST_ROOT, subj_id, 'anat')
    func_dir = os.path.join(DST_ROOT, subj_id, 'func')
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(func_dir, exist_ok=True)
    return anat_dir, func_dir

def copy_anat_files(subj_path, anat_dir, subj_id):
    anat_src = os.path.join(subj_path, 'anat')
    if not os.path.isdir(anat_src):
        print(f"No anat folder found for {subj_id}")
        return
    for file in os.listdir(anat_src):
        print("check")
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            src_file = os.path.join(anat_src, file)
            dst_file = os.path.join(anat_dir, f"{subj_id}_T1w.nii.gz")
            shutil.copy2(src_file, dst_file)
            print(f"Copied anat file for {subj_id}: {dst_file}")

def merge_func_files(subj_path, func_dir, subj_id):
    func_src = os.path.join(subj_path, 'func')
    if not os.path.isdir(func_src):
        print(f"No func folder found for {subj_id}")
        return
    func_files = sorted(glob(os.path.join(func_src, '*.nii*')))
    if not func_files:
        print(f"No func files found for {subj_id}")
        return

    arrays = []
    affine = None
    header = None

    for i, fpath in enumerate(func_files):
        print(f"Loading functional file {i+1}/{len(func_files)}: {fpath}")
        start = time.time()
        img = nib.load(fpath)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 3:
            data = data[..., np.newaxis]
        arrays.append(data)
        if affine is None:
            affine = img.affine
        if header is None:
            header = img.header
        print(f"Loaded in {time.time() - start:.2f} seconds")

    print("Concatenating functional runs...")
    start = time.time()
    combined = np.concatenate(arrays, axis=3)
    print(f"Concatenation took {time.time() - start:.2f} seconds")

    dst_file = os.path.join(func_dir, f"{subj_id}_task-StudyTest_bold.nii.gz")
    combined_img = nib.Nifti1Image(combined, affine, header)

    print("Saving combined file...")
    start = time.time()
    nib.save(combined_img, dst_file)
    print(f"Saved combined func file for {subj_id}: {dst_file} (took {time.time() - start:.2f} seconds)")
def main():
    subjects = [d for d in os.listdir(SRC_ROOT)
                if os.path.isdir(os.path.join(SRC_ROOT, d)) and d.startswith('sub-')]

    print(f"Found subjects: {subjects}")

    for subj_id in subjects:
        print(f"\nProcessing {subj_id} ...")
        subj_path = os.path.join(SRC_ROOT, subj_id)
        anat_dir, func_dir = make_bids_dirs(subj_id)
        copy_anat_files(subj_path, anat_dir, subj_id)
        merge_func_files(subj_path, func_dir, subj_id)

if __name__ == "__main__":
    main()
