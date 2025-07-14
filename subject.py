import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class Subject:
    def __init__(self, ID, regressors, fbrain_image):
        """
        Description:
            Defines a class that holds each subject data from the experiment along
            with an API that allows for data analysis and visualization.

        self.id (int): ID of the subject
        -------
        self.regressors (np.darray): Regressors matrix of shape (14, Experiment time = 1653)
        -------
        self.fbrain_image ???
        -------
        self.activation_windows_dic (dict): A dictionary with regressor keys as keys and a an array
        of arrays as values. The arrays correspond to the intervals during which each regressor key
        is active.
        -----
        @returns an object of the class Subject
        """
        self.id = ID
        self.regressors = regressors
        self.fbrain_image = fbrain_image
        self.activation_windows_dict = None

    # ========== Getters ========== #
    def get_ID(self):
        return self.id
    def get_regressors(self):
        return self.regressors
    def get_fbrain_image(self):
        return self.fbrain_image
    
    # ========== Setters ========== #
    def set_ID(self, ID):
        self.ID = ID
    def set_regressors(self, regressors):
        self.regressors = regressors
    def set_fbrain_image(self, fbrain_image):
        self.fbrain_image = fbrain_image
    
    # ========= Data Analysis Methods ====== #
    def display_regressor(self):
        """
        Description:
            Plots the regressor matrix as a heatmap, showing how each regressor changes over time.
            The plot includes labeled axes, a colorbar for intensity, and a title with the subject ID.
            Useful for getting a quick visual overview of the regressor patterns.
        """
        reg_matrix = self.regressors
        ID = self.ID
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        ax = sns.heatmap(
            reg_matrix,
            cmap='YlGn',        
            cbar_kws={'label': 'Intensity'},  
            linewidths=0.3,       
            linecolor='white',
            square=False          
        )

        plt.title(f'Regressors for {ID}', fontsize=22, pad=20)
        plt.xlabel('Time (TR)', fontsize=16, labelpad=10)
        plt.ylabel('Regressor', fontsize=16, labelpad=10)

       
        ax.tick_params(axis='both', which='major', labelsize=12)
        sns.despine(top=True, right=True)

        plt.tight_layout()
        plt.show()

    def regressor_activation_windows(self):
        """
        Description:
            Finds and stores activation windows for each regressor, where values are continuously 1.
            For each regressor (row), it identifies time ranges where activation occurs and returns
            a dictionary mapping each regressor to a list of (start, end) index tuples.
        ----------
        @returns activation_windows_dict (14 keys, each corresponding to a unique regressor key)
        """
        activation_windows_dict = {}
        reg_matrix = self.regressors
        for behavior in range(len(reg_matrix)):
            activation_windows_dict[behavior] = []
            window_start = 0
            window_end = 0
            while (window_end < len(reg_matrix[behavior])):
                if (reg_matrix[behavior][window_end] == 0):
                    window_end += 1
                    window_start += 1
                else:
                    while(window_end < len(reg_matrix[behavior]) and reg_matrix[behavior][window_end] == 1):
                        window_end += 1
                    activation_windows_dict[behavior].append((window_start, window_end))
                    window_start = window_end
        self.activation_windows_dict = activation_windows_dict
        return activation_windows_dict
    
    def regressor_windows_boundaries(self, regressor_key):
        """
        Description:
            Retrieves the start and end boundaries of activation windows for a specified regressor.
            If activation windows have not been computed yet, it calculates and stores them first.

        ====== Parameters =======
        @param regressor_key (int): The key identifying which regressor's windows to retrieve.

        ====== Returns =======
        @returns boundaries (list of np.array): A list containing two numpy arrays:
        - start: array of start indices of activation windows
        - end: array of end indices of activation windows
        """
        activation_windows_dict = None
        if (self.activation_windows_dict != None):
            activation_windows_dict = self.activation_windows_dict
        else:
            activation_windows_dict = self.regressor_activation_windows()
            self.activation_windows_dict = activation_windows_dict
        
        start = np.array([t1 for (t1, t2) in activation_windows_dict[str(regressor_key)]])
        end = np.array([t2 for (t1, t2) in activation_windows_dict[str(regressor_key)]])
        return [start, end]
    
    def get_cue_time(self, cue_offset):
        """
        Description:
            Calculates adjusted cue times based on activation windows for specific behavioral conditions.
            It finds the boundaries of activation windows for specific regressors (End time for all list 1 runs,
            start times for all list 2 runs the are preceded by a forget cue, start times for all list 2 runs
            that are preceded by a remember cue). It then computes the cue times for all remember
            and forget cues.
        ======= Parameters ======
        @param cue_offset (int): The offset value to add to the matched cue end times.
        ====== returns =======
        @returns cue_times (dict): A dictionary with two keys:
            - 'remember': numpy array of adjusted end times for "Remember" cues.
            - 'forget': numpy array of adjusted end times for "Forget" cues.
        """
        activation_windows_dict = self.activation_windows_dict
        cues = {
            "List 1": 0,
            "List 2, Remember": 1,
            "List 2, Forget": 2
        }
        _, end_L1 = self.regressor_windows_boundaries(activation_windows_dict, cues["List 1"])
        start_L2_R, _ = self.regressor_windows_boundaries(activation_windows_dict, cues["List 2, Remember"])
        start_L2_F, _ = self.regressor_windows_boundaries(activation_windows_dict, cues["List 2, Forget"])

        end_L1_R = end_L1[np.where(cdist(np.atleast_2d(end_L1).T, np.atleast_2d(start_L2_R).T) < 10)[0]] + cue_offset
        end_L1_F = end_L1[np.where(cdist(np.atleast_2d(end_L1).T, np.atleast_2d(start_L2_F).T) < 10)[0]] + cue_offset

        return {'remember': end_L1_R, 'forget': end_L1_F}
