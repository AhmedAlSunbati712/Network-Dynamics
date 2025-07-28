import nibabel as nib
import nilearn as nl
import os
import pandas as pd
import numpy as np
from nltools.data import Brain_Data
from nltools.mask import expand_mask
from nltools.plotting import plot_glass_brain
from glob import glob
from matplotlib.colors import ListedColormap
import pickle
from tqdm import tqdm


def build_parcellation_key(key_path):
    """
    Description:
        Constructs a labeled parcellation key from a tab-delimited file, adds network metadata
        and codes, and builds a color map.

    ========== Parameters ==========
    @param key_path: Path to the parcellation key .txt file.

    ========== Returns ==========
    @returns Tuple of (dataframe with metadata, colormap object, dictionary of network color mappings)
    """
    df = read_parcellation_file(key_path)
    df = parse_parcellation_metadata(df)
    df = map_network_labels(df)
    df = finalize_key(df)
    cmap, color_dict = build_network_colormap()
    return df, cmap, color_dict


def load_parcellation_mask(parcels_path):
    """
    Description:
        Loads a parcellation mask from a NIfTI file and expands it into a list of binary ROI masks.

    ========== Parameters ==========
    @param parcels_path: Path to the parcellation NIfTI file.

    ========== Returns ==========
    @returns List of binary ROI masks.
    """
    return expand_mask(Brain_Data(parcels_path))

def create_labeled_parcellation(parcels_path, key):
    """
    Description:
    Replaces voxel values in the parcellation image with corresponding network codes from a provided key.

    ========== Parameters ==========
    @param parcels_path: Path to the parcellation NIfTI file.
    @param key: DataFrame containing mapping of parcel IDs to network codes.

    ========== Returns ==========
    @returns NIfTI image with network codes replacing parcel IDs.
    """
    nx = Brain_Data(parcels_path)
    nx.data = np.array([key.loc[i, 'code'] for i in nx.data]).astype(float)
    return nx.to_nifti()

def plot_parcellation(nifti_image, cmap, output_path=None, display=True):
    """
    Description:
        Displays a glass brain plot of the labeled parcellation and optionally saves it as a PDF.

    ========== Parameters ==========
    @param nifti_image: NIfTI image containing parcellation data.
    @param cmap: Colormap object to color the networks.
    @param output_path: Optional path to save the output PDF.
    @param display: Boolean to show the plot on screen.

    ========== Returns ==========
    @returns None
    """
    if output_path:
        plot_glass_brain(nifti_image, cmap=cmap, display_mode='lyrz', output_file=output_path)
    if display:
        plot_glass_brain(nifti_image, cmap=cmap, display_mode='lyrz')

def get_valid_nii_files(dffr_dir, exclude):
    """
    Description:
        Returns a list of valid NIfTI files from a directory, excluding files based on subject ID.

    ========== Parameters ==========
    @param dffr_dir: Directory containing .nii.gz files.
    @param exclude: List of subject IDs to exclude.

    ========== Returns ==========
    @returns List of filtered .nii.gz file paths.
    """
    all_files = glob(os.path.join(dffr_dir, '*.nii.gz'))
    return [f for f in all_files if nii_fname2subj(f) not in exclude]

def extract_and_cache_roi_data(nii_files, mask, scratch_dir):
    """
    Description:
        Extract ROI time series from each NIfTI file using the given mask,
        cache each result as a pickle, and return all results in a dictionary.

    Parameters:
        nii_files (list): List of .nii.gz file paths
        mask (list): List of ROI masks (e.g., from nltools.expand_mask)
        scratch_dir (str): Directory to save the .pkl files

    Returns:
        data (dict): Dictionary mapping subject ID to extracted ROI time series (numpy array)
    """
    data = {}

    for n in tqdm(nii_files, desc="Extracting ROI data"):
        subj_id = nii_fname2subj(n)
        pickle_path = os.path.join(scratch_dir, f'{subj_id}_{len(mask)}_parcels.pkl')

        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                x = pickle.load(f)
        else:
            img = Brain_Data(n)
            x = apply_mask_to_img(n, mask, img=img)

            with open(pickle_path, 'wb') as f:
                pickle.dump(x, f)

        data[subj_id] = x

    return data

# <---------------- Helper functions ---------------> #

def read_parcellation_file(path):
    """
    Description:
        Reads a tab-delimited parcellation file and removes the last column.

    ========== Parameters ==========
    @param path: Path to the .tsv file.

    ========== Returns ==========
    @returns DataFrame with columns: id, name, r, g, b.
    """
    return pd.read_csv(path, sep='\t', header=None, names=['id', 'name', 'r', 'g', 'b', 't']).drop(columns='t')


def parse_parcellation_metadata(df):
    """
    Description:
        Splits the region name into study, hemisphere, and raw network labels.

    ========== Parameters ==========
    @param df: DataFrame containing the parcellation info.

    ========== Returns ==========
    @returns Updated DataFrame with added columns: study, hemisphere, network_raw.
    """
    name_parts = df['name'].str.split('_', expand=True)
    df['study'] = name_parts[0]
    df['hemisphere'] = name_parts[1].str[0]
    df['network_raw'] = name_parts[2]
    return df


def map_network_labels(df):
    """
    Description:
        Maps raw network labels to human-readable names and assigns numeric codes.

    ========== Parameters ==========
    @param df: DataFrame with column network_raw.

    ========== Returns ==========
    @returns Updated DataFrame with columns network and code.
    """
    lookup_table = {
        'VisCent': 'Visual central',
        'VisPeri': 'Visual peripheral',
        'SomMotA': 'Somatomotor A',
        'SomMotB': 'Somatomotor B',
        'DorsAttnA': 'Dorsal attention A',
        'DorsAttnB': 'Dorsal attention B',
        'SalVentAttnA': 'Salience/ventral attention A',
        'SalVentAttnB': 'Salience/ventral attention B',
        'LimbicA': 'Limbic A',
        'LimbicB': 'Limbic B',
        'ContA': 'Control A',
        'ContB': 'Control B',
        'ContC': 'Control C',
        'DefaultA': 'Default mode A',
        'DefaultB': 'Default mode B',
        'DefaultC': 'Default mode C',
        'TempPar': 'Temporal parietal',
    }

    df['network'] = df['network_raw'].map(lookup_table)
    network_names = list(lookup_table.values())
    network_codes = {name: i + 1 for i, name in enumerate(network_names)}
    df['code'] = df['network'].map(network_codes)
    return df


def finalize_key(df):
    """
    Description:
        Cleans up the key DataFrame by removing unused columns, setting index, and assigning a background code.

    ========== Parameters ==========
    @param df: DataFrame containing network and code columns.

    ========== Returns ==========
    @returns Finalized key DataFrame indexed by parcel ID.
    """
    df.drop(columns=['name', 'network_raw'], inplace=True)
    df.set_index('id', inplace=True)
    df.loc[0, 'code'] = 0  # background code
    return df


def build_network_colormap():
    """
    Description:
        Creates a ListedColormap and dictionary of color codes for network labels.

    ========== Parameters ==========
    @param None:

    ========== Returns ==========
    @returns Tuple of (ListedColormap object, dictionary of network color mappings)
    """
    network_colors = {
        'Visual central': '#BE1E2D',
        'Visual peripheral': '#C85248',
        'Somatomotor A': '#F15A29',
        'Somatomotor B': '#F47C4B',
        'Dorsal attention A': '#F7941D',
        'Dorsal attention B': '#FAA74A',
        'Salience/ventral attention A': '#F9ED32',
        'Salience/ventral attention B': '#FBF392',
        'Limbic A': '#009444',
        'Limbic B': '#7CB380',
        'Control A': '#1C75BC',
        'Control B': '#7D9FD3',
        'Control C': '#556683',
        'Default mode A': '#92278F',
        'Default mode B': '#B37AB4',
        'Default mode C': '#844984',
        'Temporal parietal': '#DA1C5C',
    }

    colors = list(network_colors.values())
    return ListedColormap(colors, N=len(colors) * 2, name='networks'), network_colors


def nii_fname2subj(fname):
    """
    Description:
        Extracts subject ID from the filename of a .nii.gz file.

    ========== Parameters ==========
    @param fname: File path to a .nii.gz NIfTI image.

    ========== Returns ==========
    @returns Subject ID string extracted from the filename.
    """
    return os.path.basename(fname).replace('.nii.gz', '')

def apply_mask_to_img(fname, mask, img=None):
    """
    Description:
        Extracts time series data for each ROI in the mask from the given image.

    ========== Parameters ==========
    @param fname: Path to NIfTI file.
    @param mask: List of ROI masks.
    @param img: Optional pre-loaded Brain_Data object.

    ========== Returns ==========
    @returns NumPy array of shape (ROIs × timepoints) containing time series data.
    """
    img = Brain_Data(fname) if img is None else img
    x = np.zeros((len(img), len(mask)))
    for i, m in enumerate(mask):
        x[:, i] = img.extract_roi(m)
    return x