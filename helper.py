import requests
import zipfile
import os
from subject import Subject
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
from glob import glob as lsdir
from mat73 import loadmat

dropbox_link = "https://www.dropbox.com/scl/fo/246zhmzuof9085ls4oy8n/AJFJOiD1lS-Jh5ZZp0mgyaQ?rlkey=4rb9rbjx8g9kumj6fyorf2dkg&dl=1"
download_path = "./data.zip"
targeted_dir = "./data/"

EXCLUDE = ['072413_DFFR_0', '112313_DFFR_0']

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

def load_behavior(behavior_dir):
    """
    Description:
        Loads behavioral .mat files for each subject-run pair from the 'regressors' 
        subdirectory within the given behavior directory. Excludes files listed in the
        EXCLUDE list. Each file is loaded as a dictionary using `loadmat`, and stored in
        a dictionary keyed by the filename (minus the `.mat` extension).

    ========== Parameters ==========
    @param behavior_dir (str): Path to the root subject data directory. Must contain a 
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