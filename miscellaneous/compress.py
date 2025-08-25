import os
import nibabel as nib

def compress_nii_to_niigz(input_dir, output_dir):
    """
    Compress all .nii files from input_dir into .nii.gz format and save to output_dir,
    preserving directory structure. Original files are not deleted.

    Parameters:
        input_dir (str): Path to directory containing .nii files.
        output_dir (str): Path to directory where .nii.gz files will be saved.

    Returns:
        None
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii') and not file.endswith('.nii.gz'):
                nii_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                gz_path = os.path.join(output_subdir, file + '.gz')

                img = nib.load(nii_path)
                nib.save(img, gz_path)

                print(f"Compressed: {nii_path} -> {gz_path}")

if __name__ == "__main__":
    compress_nii_to_niigz("./data/DFFR", "./data/DFFR_gz")
