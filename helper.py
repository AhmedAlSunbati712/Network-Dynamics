import requests
import zipfile
import os
from subject import Subject
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
import pickle
import scipy as sp
from glob import glob as lsdir
from mat73 import loadmat
import nibabel as nib
from nilearn.input_data import NiftiMasker

dropbox_link = "https://www.dropbox.com/scl/fo/246zhmzuof9085ls4oy8n/AJFJOiD1lS-Jh5ZZp0mgyaQ?rlkey=4rb9rbjx8g9kumj6fyorf2dkg&dl=1"
download_path = "./data.zip"
targeted_dir = "./data/"

EXCLUDE = ['072413_DFFR_0', '112313_DFFR_0', '073113_DFFR_0']

def download_data(download_link, download_path):
    """
    Description:
    Downloads a file from the specified URL and saves it to the given directory.
    ====== Parameters ======
    @param download_link: The URL to download the file from.
    @param download_dir: The path to which to save the data.
    ====== Returns ======
    @return: None. Prints status messages.
    """
    print(f"Downloading from Dropbox...\nURL: {download_link}")

    try:
        response = requests.get(download_link, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Download complete. File saved to: {download_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")

def extract_zip_file(file_path, targeted_dir, exclude_files = ['__MACOSX/']):
    """
    Description: Extracts the contents of a ZIP file to the specified directory.
    ====== Parameters ======
    @param file_path: The path to the ZIP file.
    @param targeted_dir: The directory where the contents should be extracted.
    ====== returns =====
    @return: None if file doesn't exist or an error occurs. Otherwise, prints status messages.
    """
    if (not os.path.exists(file_path)):
        print("Error: File doesn't exist")
        return None
    try:
        exclude_files = exclude_files or []
        with zipfile.ZipFile(file_path, "r") as zip:
            for member in zip.infolist():
                if any(member.filename.startswith(prefix) for prefix in exclude_files):
                    continue
                zip.extract(member, targeted_dir)
    except zipfile.BadZipFile:
        print(f"Error: File {file_path} is not a valid zip file or is corrupted.")
    except zipfile.LargeZipFile:
        print(f"Error: ZIP file requires ZIP64 support but it's not enabled.")

def load_regressors(behavior_dir):
    """
    Dsecription:
        Loads regressors _regs_results.mat files for each subject from the 'regressors'
        subdirectory within the given behavior directory. Excludes files listed in the
        EXCLUDE list. Each file is listed as a dictionary using 'loadmat', and stored in
        a dictionary keyed by the filename.

    ========= Parameters =========
    @param behavior_dir: Path to the root subject data directory. Must contain a 
    'regressors/' subdirectory with *_regs_results.mat files.

    ========= Returns =========
    @returns: A dictionary where the keys are subject identifiers and values are the contents
    of the corresponding loaded .mat files.
    """
    return {os.path.split(r)[1][:-len('_regs_results.mat')]: 
               sp.io.loadmat(r)['all_regressors'][:-1, :] 
               for r in lsdir(os.path.join(behavior_dir, 'regressors', '*_regs_results.mat'))
               if os.path.split(r)[1][:-len('_regs_results.mat')] not in EXCLUDE}
    

def load_behavior(behavior_dir):
    """
    Description:
        Loads behavioral .mat files for each subject from the 'regressors' 
        subdirectory within the given behavior directory. Excludes files listed in the
        EXCLUDE list. Each file is loaded as a dictionary using `loadmat`, and stored in
        a dictionary keyed by the filename.

    ========== Parameters ==========
    @param behavior_dir: Path to the root subject data directory. Must contain a 
    'regressors/' subdirectory with *_0.mat and *_1.mat files.

    ========== Returns ==========
    @returns (dict): A dictionary where keys are subject-run identifiers (filenames 
    without extension) and values are the contents of the corresponding loaded .mat
    files (as dictionaries).
    """
    behavior = {os.path.split(r)[1][:-4]: loadmat(r)
            for r in lsdir(os.path.join(behavior_dir, 'regressors', '*_0.mat')) + lsdir(os.path.join(behavior_dir, 'regressors', '*_1.mat'))
            if os.path.split(r)[1][:-4] not in EXCLUDE}
    return behavior



def accuracy_by_cue_across_subjects(subjects_list):
    """
    Description:
        Computes and visualizes average recall accuracy across subjects for each list
        type (A or B) and cue condition (remember or forget). For each subject, it 
        builds an Egg object, extracts average accuracies by cue using `plot_accuracy_by_cue`,
        then averages across subjects. Displays bar plots if running in interactive mode, and 
        returns the average accuracies across all subjects.

    ========== Parameters ==========
    @param behavior_dir (str): Path to the directory containing behavioral data in .mat format
    under 'regressors/'.

    ========== Returns ==========
    @returns (tuple of floats):
        (list_A_remember_across_avg, list_A_forget_across_avg, list_B_remember_across_avg, 
        list_B_forget_across_avg) Average recall accuracies across all subjects for each 
        list (A or B) and cue type (remember or forget).
    """
    def get_average_accuracy(curr_list):
        """
        Description:
            Computes the average value of a list of floats. Returns 0 if the list is empty.
        """
        if (len(curr_list) == 0):
            return 0
        else:
            return sum(curr_list) / len(curr_list)
    
    def plot_cue_histogram(list_A, list_B, cue_type):
        """
        Description:
            Plots a bar chart showing average recall accuracy by list (A or B) for a given cue
            type (remember or forget), using seaborn for consistent styling.
        """
        df = pd.DataFrame({
            'List': ['A', 'B'],
            'Accuracy': [list_A, list_B],
            'Condition': [f'{cue_type}', f'{cue_type}']
        })

        plt.figure(figsize=(6, 5))
        sns.barplot(data=df, x='List', y='Accuracy', hue='List', palette='Reds_d', legend=False)

        plt.title(f'Average accuracy ({cue_type}) across subjects', fontsize=12)
        plt.ylim(0, 1)
        plt.xlabel('List')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.show()
    
    
    list_A_remember_subj_accuracies = []
    list_B_remember_subj_accuracies = []

    list_A_forget_subj_accuracies = []
    list_B_forget_subj_accuracies = []

    for subj in subjects_list:
        subj.build_egg_subject()
        
        (
            list_A_remember_avg_accuracy, 
            list_A_forget_avg_accuracy, 
            list_B_remember_avg_accuracy, 
            list_B_forget_avg_accuracy,
        ) = subj.accuracy_by_cue(plot='n')
        
        list_A_remember_subj_accuracies.append(list_A_remember_avg_accuracy)
        list_A_forget_subj_accuracies.append(list_A_forget_avg_accuracy)
        
        list_B_remember_subj_accuracies.append(list_B_remember_avg_accuracy)
        list_B_forget_subj_accuracies.append(list_B_forget_avg_accuracy)
    
    # Averaging out recall accuracy across subjects for remember cues across the two lists
    list_A_remember_across_avg = get_average_accuracy(list_A_remember_subj_accuracies)
    list_B_remember_across_avg = get_average_accuracy(list_B_remember_subj_accuracies)

    # Averaging out recall accuracy across subjects for forget cues across the two lists
    list_A_forget_across_avg = get_average_accuracy(list_A_forget_subj_accuracies)
    list_B_forget_across_avg = get_average_accuracy(list_B_forget_subj_accuracies)

    plot_cue_histogram(list_A_remember_across_avg, list_B_remember_across_avg, "Remember")
    plot_cue_histogram(list_A_forget_across_avg, list_B_forget_across_avg, "Forget")

    return list_A_remember_across_avg, list_A_forget_across_avg, list_B_remember_across_avg, list_B_forget_across_avg

def spc_by_list_across_subjects(subjects_list, cue_type, plot='y'):
    """
    Description:
        Extracts the serial position recall for each run in each subject according to the given cue
        type. Then, it averages the recall probability across subjects for each list for the given
        cue type. If @param plor is not 'y', it will plot the lineplots for both lists.
    
    =============== Parameters ================
    @param subjects_list: A list of Subject objects. Shape: (, 24)
    @param cue_type: The memory cue to target in the lists. Takes one of two values: 'remember'/'forget'
    @param plot: Takes 'y' (yes) or 'n' (no) to plot the spc for lists A and B

    ================ Returns ===============
    @returns A_array (np.ndarray), shape (, 16): The average recall probability at each serial position
                                                 for list A under the given memory cue.
    @returns B_array (np.ndarray), shape (, 16): The average recall probability at each serial position
                                                 for list B under the given memory cue.
    """
    def plot_spc(A_array, B_array):
        """
        Description:
            Plots the mean serial position curve for List A and List B with ±1 standard deviation shaded
            regions.
        """
        x = np.arange(1, 17) # Serial positions
        
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.figure(figsize=(6, 4))

        # Mean and std for List A
        A_mean = A_array.mean(axis=0)
        A_std = A_array.std(axis=0)

        # Mean and std for List B
        B_mean = B_array.mean(axis=0)
        B_std = B_array.std(axis=0)

        plt.plot(x, A_mean, color='#1f77b4', label='List A', linewidth=2)
        plt.fill_between(x, A_mean - A_std, A_mean + A_std, color='#1f77b4', alpha=0.2)

        plt.plot(x, B_mean, color='#d62728', label='List B', linewidth=2)
        plt.fill_between(x, B_mean - B_std, B_mean + B_std, color='#d62728', alpha=0.2)

        plt.xlabel('Serial Position', fontsize=14)
        plt.ylabel('Recall Probability', fontsize=14)
        plt.title(f'Serial Position Curve Across Subjects ({cue_type.capitalize()} Cue)', fontsize=12, weight='bold')
        plt.xticks(ticks=x)
        plt.legend(frameon=True, loc='upper right')
        plt.tight_layout()
        plt.show()
    
    # Initialize containers for SPC vectors
    lists_A = []
    lists_B = []
    
    # Collect SPC values from each subject
    for subject in subjects_list:
        subj_list_A, subj_list_B = subject.spc_by_list(cue_type, 'n')
        if (subj_list_A is not None): lists_A.append(subj_list_A)
        if (subj_list_B is not None): lists_B.append(subj_list_B)
    
    if lists_A: A_array = np.vstack(lists_A)
    if lists_B: B_array = np.vstack(lists_B)
    if (plot == 'y'): plot_spc(A_array, B_array)
    return A_array, B_array

def nii2cmu_all(subjects_list, mask_file):
    cmu_data = []
    for subject in subjects_list:
        nifti_file = subject.nii_file
        cmu_saved_path = f"data/cmuData/{subject.get_ID()}_cmu.pkl"
        if (os.path.exists(cmu_saved_path)):
            with open(cmu_saved_path, "rb") as f:
                Y, R = pickle.load(f)
        else:
            Y, R = nifti_to_cmu_format(nifti_file)
            with open(cmu_saved_path, "rb") as f:
                pickle.dump((Y, R), f)
        subject_cmu_data = {'R': R, 'Y': Y}
        cmu_data.append(subject_cmu_data)
        print(f'Successfully loaded subject {subject.get_ID()}')
    return cmu_data

def initialize_htfa_params(
        K=30,
        max_global_iter=15,
        max_local_iter=30,
        voxel_ratio=0.7,
        tr_ratio=0.7,
        max_voxel_scale=None,
        max_tr_scale=None,
        verbose=True
        ):
    params = {
        'K': K,
        'max_global_iter': max_global_iter,
        'max_local_iter': max_local_iter,
        'voxel_ratio': voxel_ratio,
        'tr_ratio': tr_ratio,
        'max_voxel_scale': max_voxel_scale,
        'max_tr_scale': max_tr_scale,
        'verbose': verbose,
    }
    return params

def params2str(params):
    param2abb = {
        'K': 'K',
        'max_global_iter': 'MGIter',
        'max_local_iter': 'MLIter',
        'voxel_ratio': 'vRatio',
        'tr_ratio': 'trRatio',
        'max_voxel_scale': 'MVScale',
        'max_tr_scale': 'MTRScale',
        'verbose': 'verbose'

    }
    res = ""
    for key in params.keys():
        abbreviation = param2abb[key]
        value = params[key]

        res += abbreviation + "-" + str(value) + '_'
    return res
        

def cmu2htfa(cmu_data):
    htfa_data = [{'R': s['R'], 'Z': s['Y'].T} for s in cmu_data]
    print(f'Converted cmu data into htfa format succesfully!')
    return htfa_data

def get_cmu_data(subjects_list):
    data = []
    for subject in subjects_list:
        data.append({'Y': subject.Y, 'R': subject.R})
    return data

def get_centers_and_widths(parcels_fname):
    Y, R = nifti_to_cmu_format(parcels_fname)
    Y = Y.astype(int)
    centers = np.zeros((Y.max() - Y.min() + 1, 3))
    widths = np.zeros(Y.max() - Y.min() + 1)

    for i in np.unique(Y):
        mask = (Y == i).ravel()
        centers[i - 1] = np.mean(R[mask, :], axis=0)
        widths[i - 1] = np.mean(cdist(R[mask, :], centers[i - 1].reshape(1, -1)))
    
    return centers, widths
def nifti_to_cmu_format(nifti_file, mask_file=None):
    """
    Description:
    Converts a NIfTI fMRI file into CMU format, applying a brain mask to remove background voxels.
    Outputs:
      - Y: (timepoints, brain_voxels) data matrix from the masked fMRI data.
      - R: (brain_voxels, 3) voxel coordinates in real space (mm), matching columns of Y.

    =========== Parameters ===========
    @param nifti_file: str
        Path to the NIfTI (.nii or .nii.gz) file.
    @param mask_file: str or None
        Optional path to a mask NIfTI file. If None, mask is computed from data.

    =========== Returns ===========
    @returns Y: numpy.ndarray
        Shape (num_timepoints, num_masked_voxels), masked fMRI data matrix.
    @returns R: numpy.ndarray
        Shape (num_masked_voxels, 3), real-world coordinates (mm) of masked voxels.
    """

    # Load NIfTI
    img = nib.load(nifti_file)
    S = img.get_sform()

    # Create mask (background strategy unless mask file provided)
    masker = NiftiMasker(mask_strategy='background')
    if mask_file is None:
        masker.fit(nifti_file)
    else:
        masker.fit(mask_file)

    # Apply mask to get Y (timepoints x masked_voxels)
    Y = np.float32(masker.transform(nifti_file)).copy()

    # Get the mask image data as boolean array
    mask_data = masker.mask_img_.get_fdata().astype(bool)

    # Get voxel indices inside the mask
    voxel_indices = np.array(np.nonzero(mask_data)).T  # shape (num_voxels, 3), coords in (x, y, z) order

    # Convert voxel coords to real space (mm)
    R = nib.affines.apply_affine(S, voxel_indices)

    return Y, R

def create_group_mask(nii_files, output_file="group_mask.nii.gz", method="intersection"):
    """
    Description: 
    Create a group brain mask from a list of NIfTI files.

    =========== Parameters ===========

    @param nii_files : list of str
        Paths to subject NIfTI files.
    @param output_file : str
        Filename to save the group mask NIfTI.
    @param method : str
        'intersection' (default) includes voxels present in all subjects.
        'union' includes voxels present in any subject.

    =========== Returns ===========
    @returns group_mask_img : nibabel.Nifti1Image
        The group mask NIfTI image.
    @returns output_file : str
        Path to the saved group mask file.
    """
    masks = []

    # Step 1: create individual masks
    for f in nii_files:
        img = nib.load(f)
        data = img.get_fdata()
        # mask = voxels with non-zero values across time (axis=3)
        mask = np.any(data != 0, axis=3)
        masks.append(mask)

    # Step 2: combine masks
    if method == "intersection":
        group_mask = np.logical_and.reduce(masks)
    elif method == "union":
        group_mask = np.logical_or.reduce(masks)
    else:
        raise ValueError("method must be 'intersection' or 'union'")

    # Step 3: save the group mask
    group_mask_img = nib.Nifti1Image(group_mask.astype(np.uint8), affine=img.affine)
    nib.save(group_mask_img, output_file)
    print(f"Group mask saved to {output_file} with shape {group_mask.shape}")

    return group_mask_img, output_file