import os
import pandas as pd
import numpy as np
import seaborn as sns
#from scipy import spatial as sd
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from matplotlib.colors import ListedColormap
from nltools.data import Brain_Data
from nltools.mask import expand_mask
from nltools.plotting import plot_glass_brain
from nilearn import plotting as niplot
import nibabel as nib
import nilearn as nl
from glob import glob as lsdir
import pickle
from tqdm import tqdm
import timecorr as tc



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
        

def load_fmri_images(subjects_list, fmri_data_path):
    nii_files = []
    for fmri_file in os.listdir(fmri_data_path):
        nii_files.append(os.path.join(fmri_data_path, fmri_file))
        for subject in subjects_list:
            if subject.get_ID() == fmri_file[:-len(".nii.gz")]:
                subject.nii_file = os.path.join(fmri_data_path, fmri_file)
    return nii_files

def extract_and_cache_roi_data(subjects_list, mask, scratch_dir):
    """
    Description:
        Extract ROI time series from each NIfTI file using the given mask,
        cache each result as a pickle, and return all results in a dictionary.

    ============ Parameters ============
        @param subjects_list (list of Subject objects): List of subject objects
        @param mask (list): List of ROI masks (from nltools.expand_mask)
        @param scratch_dir (str): Directory to save the masked images .pkl files

    ============= Returns =============
        @returns data (dict): Dictionary mapping subject ID to extracted ROI time series (numpy array)
    """
    data = {}
    original_corrmat = []
    reduced_corrmat = []
    for subject in tqdm(subjects_list, desc="Extracting ROI data"):
        subj_id = subject.get_ID().split("_")[0]
        subj_day_index = subject.nii_file[-len("x.nii.gz")]
        pickle_path = os.path.join(scratch_dir, f'{subj_id}_{subj_day_index}_{len(mask)}_parcels.pkl')

        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                masked_img, original_corrmat, reduced_corrmat = pickle.load(f)
                subject.masked_data = masked_img
        else:
            unmasked_img = Brain_Data(subject.nii_file)
            masked_img = apply_mask_to_img(subject.nii_file, mask, unmasked_img)

            with open(pickle_path, 'wb') as f:
                pickle.dump([masked_img, original_corrmat, reduced_corrmat], f)

        data[subj_id + "_DFFR_" + subj_day_index] = masked_img
        subject.Y = masked_img

    return data

def event_triggered_average(data, event_times, before=25, after=25):
    """
    Description:
        Compute the event-triggered average (ETA) of time series data. For each event time,
        the function extracts a window of data centered around the event, spanning 
        `before` timepoints before and `after` timepoints after. The output is the average 
        across all such windows, while handling edge cases where the event window 
        would extend beyond the bounds of the data by inserting NaNs.

        If `data` or `event_times` is a dictionary, the function operates recursively, 
        computing ETAs for each key.

    ============ Parameters ============
        @param data (np.ndarray or dict): 
            Time series data of shape (T, D), where T is number of timepoints and D is number of voxels.
            Can also be a dictionary mapping subject IDs to data arrays.

        @param event_times (list or dict): 
            List of integer timepoints indicating event onsets. Can also be a dictionary
            mapping subject IDs to event time lists.

        @param before (int): 
            Number of timepoints to include before each event. Default is 25.

        @param after (int): 
            Number of timepoints to include after each event. Default is 25.

    ============= Returns =============
        @returns remember_etas (list): 
            List of averaged event-triggered responses for "remember" trials (per subject).

        @returns forget_etas (list): 
            List of averaged event-triggered responses for "forget" trials (per subject).
    """
    if type(data) is dict:
        return {k: event_triggered_average(v, event_times[k], before=before, after=after) for k, v in data.items()}

    if type(event_times) is dict:
        return {k: event_triggered_average(data, v, before=before, after=after) for k, v in event_times.items()}
    
    etas = np.multiply(np.nan, np.zeros([before + after + 1, data.shape[1], len(event_times)]))
    for i, t in enumerate(event_times):
        start_time = t - before - 1
        end_time = t + after
        
        if start_time >= 0:
            start_ind = 0
        else:
            start_ind = np.abs(start_time)
            start_time = 0
        
        if end_time <= data.shape[0]:
            end_ind = before + after + 1
        else:
            end_ind = end_time - data.shape[0]
            end_time = data.shape[0]
        
        start_ind, end_ind, start_time, end_time = int(start_ind), int(end_ind), int(start_time), int(end_time)
        etas[start_ind:end_ind, :, i] = data[start_time:end_time, :]

    return np.mean(etas, axis=2)


def compute_networks(mask, var, scratch_dir, remember_etas, forget_etas, weights_function, weights_param, combine, T):
    """
    Description:
        Compute inter-subject functional connectivity matrices for "remember" and "forget" conditions
        using event-triggered averages (ETAs) and a time correlation function. Results are cached to avoid
        recomputation, and the final output is returned as full square connectivity matrices.

    ============ Parameters ============
        @param mask (list): 
            List of parcel masks (e.g., voxel groupings or region indices), used for labeling/cache naming.

        @param var (dict): 
            ??? example: {"var": 5}

        @param scratch_dir (str): 
            Path to directory for saving and loading cached ISFC results.

        @param remember_etas (list of np.ndarray): 
            List of event-triggered average time series for "remember" trials, one array per subject.

        @param forget_etas (list of np.ndarray): 
            List of event-triggered average time series for "forget" trials, one array per subject.

        @param weights_function (callable): 
            Function to compute temporal weights for time correlation.

        @param weights_param (float or dict): 
            Parameter(s) passed to the weights function.

        @param combine (str): 
            Method for combining individual subject correlations.

    ============= Returns =============
        @returns remember_matrix (np.ndarray): 
            ISFC matrix (parcel-by-parcel) for "remember" condition.

        @returns forget_matrix (np.ndarray): 
            ISFC matrix (parcel-by-parcel) for "forget" condition.
    """
    file_name = os.path.join(scratch_dir, f'isfc_{len(mask)}_parcels_T{T}_var{var}.pkl')
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            remember_isfc, forget_isfc = pickle.load(f)
    else:
        remember_isfc = tc.timecorr(remember_etas, weights_function=weights_function, weights_param=weights_param, combine=combine)
        forget_isfc = tc.timecorr(forget_etas, weights_function=weights_function, weights_param=weights_param, combine=combine)

        with open(file_name, "wb") as f:
            pickle.dump([remember_isfc, forget_isfc], f)
    return tc.vec2mat(remember_isfc), tc.vec2mat(forget_isfc)

def plot_isfc_networks(remember_isfc, forget_isfc, T, fig_dir):
    """
    Description:
        Plot inter-subject functional connectivity (ISFC) matrices for both "remember" and "forget" conditions
        across multiple timepoints. The function generates side-by-side heatmaps of ISFC matrices at evenly spaced
        time lags, highlighting temporal dynamics in connectivity patterns. The output is saved as a single figure.

    ============ Parameters ============
        @param remember_isfc (np.ndarray): 
            3D array of ISFC matrices for the "remember" condition. Shape: [n_parcels, n_parcels, T].

        @param forget_isfc (np.ndarray): 
            3D array of ISFC matrices for the "forget" condition. Shape: [n_parcels, n_parcels, T].

        @param T (int): 
            Number of timepoints or temporal windows over which ISFC matrices are computed.

        @param fig_dir (str): 
            Path to directory where the output figure (`isfc.png`) will be saved.

    ============= Returns =============
        @returns None
            The function saves the plotted figure as 'isfc.png' in the specified directory.
    """
    n = 11
    fig, ax = plt.subplots(2, n, figsize=(2 * n + 1, 4), sharex=True, sharey=True)

    for i, t in enumerate(np.round(np.linspace(-(T - 1) / 4, (T - 1) / 4, n))):
        sns.heatmap(remember_isfc[:, :, i], ax=ax[0, i], vmin=-0.5, vmax=0.5, cmap='RdBu_r', cbar=False)
        sns.heatmap(forget_isfc[:, :, i], ax=ax[1, i], vmin=-0.5, vmax=0.5, cmap='RdBu_r', cbar=False)
        ax[0, i].set_title(f'TR = {int(t)}', fontsize=14)

        if i == 0:
            ax[0, i].set_ylabel('Remember', fontsize=14)
            ax[1, i].set_ylabel('Forget', fontsize=14)
        ax[1, i].set_xlabel('Time (samples)', fontsize=14)

        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

    plt.tight_layout()

    fig.savefig(os.path.join(fig_dir, 'isfc.png'), bbox_inches='tight')



def animate_connectome(hubs, connectomes, figsdir, animation_size_limit, force_refresh=False):
    """
    Description:
    Creates an animation of connectome visualizations over time, saving individual frames
    as PNG images and compiling them into a matplotlib animation.

    ============== Parameters =============
    @param hubs: 
        Coordinates of hub nodes in the connectome (n_nodes x 3).
    @param connectomes: 
        Time-series of connectome matrices (n_timepoints x n_connections).
    @param figsdir:
        Path to directory where frame images will be saved.
    @param force_refresh:
        If True, regenerates all frames even if they already exist.

    ============== Returns ==========
    @return ani:
        Animation object containing the connectome visualization sequence.
    """

    def get_frame(t, fname):
        """Generate and save a single connectome frame if it doesn't exist or if force_refresh=True."""
        if force_refresh or not os.path.exists(fname):
            niplot.plot_connectome(
                sd.squareform(connectomes[t, :]),
                hubs,
                node_color='k',
                edge_threshold='75%',
                output_file=fname
            )

    def update(frame):
        """Update function for FuncAnimation that loads and displays a frame image."""
        img = plt.imread(fnames[frame])
        im.set_array(img)
        return [im]

    # Set animation embedding limit (in MB)
    mpl.rcParams['animation.embed_limit'] = animation_size_limit

    # Ensure figure directory exists
    if not os.path.isdir(figsdir):
        os.makedirs(figsdir)

    # Generate all frames
    timepoints = np.arange(connectomes.shape[0])
    fnames = [os.path.join(figsdir, f"{t}.png") for t in timepoints]
    list(map(get_frame, timepoints, fnames))

    # Build animation
    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(plt.imread(fnames[0]))
    ani = animation.FuncAnimation(
        fig, update, frames=len(fnames), interval=50, blit=True
    )

    plt.close(fig)
    return ani

def dynamic_connectome(W, n):
    """
    Description:
    Computes a dynamic connectome over sliding windows using correlation distance.
    If n > 0, the connectome is computed for each sliding window of size n.
    If n <= 0, the connectome is computed across the entire time series.

    ============== Parameters =============
    @param W: (T x K)
        Input time series data where T = number of timepoints and K = number of nodes.
    @param n:
        Window size. If n > 0, sliding window length. If n <= 0, use the full time series.

    ============== Returns ==========
    @return connectome:
        Dynamic connectome array. Shape is:
          - (1, K*(K-1)/2) if n <= 0
          - (T - n + 1, K*(K-1)/2) if n > 0
        Each row is the upper-triangular vectorized connectome at that time/window.
    """
    T, K = W.shape

    if n <= 0:
        # Compute connectome using full time series
        connectome = np.array([1 - sd.pdist(W.T, 'correlation')])
    else:
        connectome = np.zeros((T - n + 1, int((K ** 2 - K) / 2)))
        for t in range(T - n + 1):
            window = W[t:(t+n), :].T  # (K x n) for this window
            connectome[t, :] = 1 - sd.pdist(window, 'correlation')

    return connectome

def dynamic_ISFC(data, windowsize=0):
    """
    Description:
    Compute the dynamic inter-subject functional connectivity (ISFC) matrix using a sliding window 
    across multiple time series datasets. The method is based on the approach described in 
    http://www.nature.com/articles/ncomms12141.

    =========== Parameters ========
    @param data: 
        A list of matrices, each with dimensions [number of observations × number of features].
    @param windowsize: 
        The number of observations to include in each sliding window. 
        Set to 0 (default) to use all timepoints.

    ========== Returns ========
    @return isfc_mat: 
        A matrix of shape [number of windows × number of feature-pairs] representing the 
        dynamic ISFC values across sliding windows.
    """

    def rows(x): return x.shape[0]
    def cols(x): return x.shape[1]
    def r2z(r): return 0.5 * (np.log(1 + r) - np.log(1 - r))
    def z2r(z): return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    def vectorize(m):
        np.fill_diagonal(m, 0)
        return sd.squareform(m)

    assert len(data) > 1

    ns = list(map(rows, data))
    vs = list(map(cols, data))

    n = np.min(ns)
    if windowsize == 0:
        windowsize = n

    assert len(np.unique(vs)) == 1
    v = vs[0]

    isfc_mat = np.zeros([n - windowsize + 1, int((v ** 2 - v) / 2)])
    for n in range(0, n - windowsize + 1):
        next_inds = range(n, n + windowsize)
        for i in range(len(data)):
            mean_other_data = np.zeros([len(next_inds), v])
            for j in range(len(data)):
                if i == j:
                    continue
                mean_other_data += data[j][next_inds, :]
            mean_other_data /= (len(data) - 1)
            next_corrs = np.array(
                r2z(1 - sd.cdist(data[i][next_inds, :].T, mean_other_data.T, 'correlation'))
            )
            isfc_mat[n, :] += vectorize(next_corrs + next_corrs.T)
        isfc_mat[n, :] = z2r(isfc_mat[n, :] / (2 * len(data)))

    isfc_mat[np.where(np.isnan(isfc_mat))] = 0
    return isfc_mat


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