import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import quail as quail



class Subject:
    def __init__(self, ID, regressors, behavior, nii_file=None):
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
        self.behavior = behavior
        
        self.nii_file = nii_file
        self.Y = None
        self.R = None
        self.activation_windows_dict = None
        self.egg = None

    # ========== Getters ========== #
    def get_ID(self):
        return self.id
    def get_regressors(self):
        return self.regressors
    def get_fbrain_image(self):
        return self.fbrain_image
    def get_behavior(self):
        return self.behavior
    
    # ========== Setters ========== #
    def set_ID(self, ID):
        self.ID = ID
    def set_regressors(self, regressors):
        self.regressors = regressors
    def set_fbrain_image(self, fbrain_image):
        self.fbrain_image = fbrain_image
    
    # ========= Data Analysis Methods ====== #
    def display_regressors(self):
        """
        Description:
            Plots the regressor matrix as a heatmap, showing how each regressor changes over time.
            The plot includes labeled axes, a colorbar for intensity, and a title with the subject ID.
            Useful for getting a quick visual overview of the regressor patterns.
        """
        reg_matrix = self.regressors
        ID = self.id
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        ax = sns.heatmap(
            reg_matrix,
            cmap='YlGn',
            cbar_kws={'label': 'Intensity'},
            linecolor='white',
            square=False,
            vmin=0,
            vmax=1  # assuming binary regressors
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
                    # If it's an active interval, keep expanding the window till we hit a 0
                    while(window_end < len(reg_matrix[behavior]) and reg_matrix[behavior][window_end] == 1):
                        window_end += 1
                    # Add this interval to our dictionary
                    activation_windows_dict[behavior].append((window_start, window_end))
                    
                    # Reset the window to scan for another interval
                    window_start = window_end
        # Saving the activation windows
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
        # If have already calculated the activation windows, load it to save time
        if (self.activation_windows_dict != None):
            activation_windows_dict = self.activation_windows_dict
        else:
        # Otherwise, calculate it and save it    
            activation_windows_dict = self.regressor_activation_windows()
            self.activation_windows_dict = activation_windows_dict
        
        # Array that stores the start indices of all active intervals for the given regressor key
        start = np.array([t1 for (t1, t2) in activation_windows_dict[regressor_key]])
        # Array that stores the end indices of all active intervals for the given regressor key
        end = np.array([t2 for (t1, t2) in activation_windows_dict[regressor_key]])
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
        cues = {
            "List 1": 0,
            "List 2, Remember": 1,
            "List 2, Forget": 2
        }
        _, end_L1 = self.regressor_windows_boundaries(cues["List 1"])
        start_L2_R, _ = self.regressor_windows_boundaries(cues["List 2, Remember"])
        start_L2_F, _ = self.regressor_windows_boundaries(cues["List 2, Forget"])

        end_L1_R = end_L1[np.where(cdist(np.atleast_2d(end_L1).T, np.atleast_2d(start_L2_R).T) < 10)[0]] + cue_offset
        end_L1_F = end_L1[np.where(cdist(np.atleast_2d(end_L1).T, np.atleast_2d(start_L2_F).T) < 10)[0]] + cue_offset
            
        return {'remember': end_L1_R, 'forget': end_L1_F}
    
    def build_egg_subject(self):
        """
        Description:
            Constructs a quail.Egg object using the subject’s behavioral data.
            For each of 8 runs, generates a list of presented and recalled words,
            each annotated with cue type ("remember" or "forget") and list type ("A" or "B").
            The final Egg object is stored in self.egg and returned.

        ========== Parameters ==========
        @param self (Subject): The instance of the Subject class containing behavioral data.

        ========== Returns ==========
        @returns (quail.Egg): A quail Egg object constructed from the subject's behavioral data.
        """

        def get_cue_type(cue_type_list, run):
            """Return cue type string from binary array: 1 = forget, 0 = remember."""
            return 'forget' if cue_type_list[run] == 1 else 'remember'

        def get_list_type(l1_recalls, l2_recalls, run):
            """Determine list type: 'A' if list B was not recalled, otherwise 'B'."""
            return 'A' if l2_recalls[run] == 0 else 'B'

        def build_run_list(run, cue_type, list_type, spc_filter=False):
            """
            Create list of word dictionaries for one run.
            If spc_filter is True, include only recalled words.
            """
            run_list = []
            for word_num in range(16):
                if spc_filter and spc[run][word_num] != 1:
                    continue
                run_list.append({
                    'item': f'word {word_num}',
                    'list': list_type,
                    'cuetype': cue_type
                })
            return run_list

        # Load behavioral data
        behavior = self.behavior
        l1_recalls = behavior['L1_recalls']
        l2_recalls = behavior['L2_recalls']
        spc = behavior['spc']
        cue_type_list = behavior['cuetype']

        pres_words = []
        rec_words = []

        for run in range(8):
            # Run 7 (last run) is hardcoded as localizer: 'forget' cue, list A
            cue_type = 'forget' if run == 7 else get_cue_type(cue_type_list, run)
            list_type = 'A' if run == 7 else get_list_type(l1_recalls, l2_recalls, run)

            # Build presented and recalled word lists
            pres_words.append(build_run_list(run, cue_type, list_type, spc_filter=False))
            rec_words.append(build_run_list(run, cue_type, list_type, spc_filter=True))

        # Create and store the Egg object
        self.egg = quail.Egg(pres=pres_words, rec=rec_words)
        return self.egg

    
    def plot_histogram_over_runs(self):
        """
        Description:
            Plots a bar chart showing the recall accuracy for each list (run) for the current 
            subject. It uses Quail's accuracy analysis to compute accuracy per list and visualizes it using Seaborn.

        ========== Parameters ==========
        @param self (Subject): The subject instance whose egg attribute contains recall and presentation data.

        ========== returns ==========
        @returns (None): Displays a histogram of recall accuracy by list/run. No value is returned.
        """
        # Get accuracy data as DataFrame
        df = quail.analyze(self.egg, analysis='accuracy').get_data()

        # Reset index so 'List' becomes a column
        df_reset = df.reset_index()

        # Only one subject, so we can just take subject 0
        df_subject = df_reset[df_reset['Subject'] == 0]

        # Plot
        plt.figure(figsize=(6, 3))
        sns.barplot(data=df_subject, x='List', y=0, color='skyblue')

        plt.title(f'Recall accuracy by list for {self.id}', fontsize=12)
        plt.xlabel('List Number', fontsize=12)
        plt.ylabel('Recall Accuracy', fontsize=12)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.show()
    
    def accuracy_by_cue(self, plot='y'):
        """
        Description:
            Calculates and visualizes average recall accuracy for each list type (A or B) under two cue 
            conditions (remember or forget). If plot is "y", the method generates bar plots
            showing average accuracy by list and cue type. It also returns the computed average accuracies
            for further use.

        ========== Parameters ==========
        @param self (Subject): The subject instance containing the behavioral data and egg.
        @param plot (char): 'y' displays plots. Anything else just returns the lists built in the method
                            without plotting.

        ========== Returns ==========
        @returns (tuple of floats):
        (list_A_remember_avg_accuracy, list_A_forget_avg_accuracy, list_B_remember_avg_accuracy,
        list_B_forget_avg_accuracy) Average recall accuracies by list and cue type.
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
            
        def plot_cue_histogram(df, cue):
            """
            Description:
                Plots a bar chart showing average recall accuracy by list (A or B) for a given cue
                type (remember or forget), using seaborn for consistent styling.
            """
            plt.figure(figsize=(6, 5))
            sns.barplot(data=df, x='List', y='Accuracy', hue='List', palette='Reds_d', legend=False)

            plt.title(f'Average accuracy ({cue}) for {self.id}', fontsize=12)
            plt.ylim(0, 1)
            plt.xlabel('List')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            plt.show()
        correct_recalls = self.get_behavior()["correct_recalls"]
        list_A_forget_accuracies = []
        list_B_forget_accuracies = []

        list_A_remember_accuracies = []
        list_B_remember_accuracies = []
        
        # A df that returns the recall accuracy for each list
        
        pres_features = self.egg.get_pres_features()
        
        # Iterate over all the lists and add the recall accuracy for each run to the corresponding list
        for run in range(8):
            run_meta = pres_features.iloc[run, 0]
            list_type = run_meta['list']
            cue_type = run_meta['cuetype']

            match list_type:
                case 'A':
                    match cue_type:
                        case 'forget':
                            # Asked to recall list A after a forget cue
                            list_A_forget_accuracies.append(correct_recalls[run]/16)
                        case 'remember':
                            # Asked to recall list A after a remember cue
                            list_A_remember_accuracies.append(correct_recalls[run]/16)
                            
                case 'B':
                    match cue_type:
                        case 'forget':
                            # Asked to recall list B after a forget cue
                            list_B_forget_accuracies.append(correct_recalls[run]/16)
                        case 'remember':
                            # Asked to recall list B after a remember cue
                            list_B_remember_accuracies.append(correct_recalls[run]/16)
        
        # Averaging out the accuracies for the remember cues
        list_A_remember_avg_accuracy = get_average_accuracy(list_A_remember_accuracies)
        list_B_remember_avg_accuracy = get_average_accuracy(list_B_remember_accuracies)

        # Averaging out the accuracies for the forget cues
        list_A_forget_avg_accuracy = get_average_accuracy(list_A_forget_accuracies)
        list_B_forget_avg_accuracy = get_average_accuracy(list_B_forget_accuracies)

        # If caller asked for a plot
        if (plot == 'y'):
            df_remember = pd.DataFrame({
                'List': ['A', 'B'],
                'Accuracy': [list_A_remember_avg_accuracy, list_B_remember_avg_accuracy],
                'Condition': ['Remember', 'Remember']
            })
            df_forget = pd.DataFrame({
                'List': ['A', 'B'],
                'Accuracy': [list_A_forget_avg_accuracy, list_B_forget_avg_accuracy],
                'Condition': ['Forget', 'Forget']
            })
            plot_cue_histogram(df_remember, "Remember")
            plot_cue_histogram(df_forget, "Forget")
        
        return list_A_remember_avg_accuracy, list_A_forget_avg_accuracy, list_B_remember_avg_accuracy, list_B_forget_avg_accuracy


    def spc_by_list(self, cue_type, plot='y'):
        """
        Description:
            Calculautes average recall probability for all serial positions for the given subject for
            lists A and B given the specified memory cue. If chosen, it also plots the lineplots for
            the spc for lists A and B.
        
        ============== Parameters ================
        @cue_type: The type of the cue to target: 'remember'/'forget'
        @plot: y/n

        ============== Returns =================
        @list_A: The average recall probability for list A at serial positions for the given cue.
        @list_B: The average recall probability at for list B serial positions for the given cue
        """
        def plot_spc(list_A, list_B):
            """
            Description:
                Plots the serial position curve for List A and List B.
            """
            x = np.arange(1, 17)
            sns.set_theme(style="whitegrid", font_scale=1.2)
            plt.figure(figsize=(6, 4))

            if list_A is not None:
                plt.plot(x, list_A, color='#1f77b4', label='List A', linewidth=1.5)
            if list_B is not None:
                plt.plot(x, list_B, color='#d62728', label='List B', linewidth=1.5)
            
            plt.xlabel('Serial Position', fontsize=14)
            plt.ylabel('Recall Probability', fontsize=14)
            plt.title(f'Serial Position Curve ({cue_type.capitalize()} Cue)', fontsize=12, weight='bold')
            plt.xticks(ticks=np.arange(1, 17))
            plt.legend(frameon=True, loc='upper right')
            plt.tight_layout()
            plt.show()
                


        count_A = 0
        count_B = 0

        list_A = np.zeros(16)
        list_B = np.zeros(16)

        spc = self.behavior['spc']
        subject_recall_features = self.egg.get_pres_features()

        for run in range(8):
            current_run_type = subject_recall_features.iloc[run, 0]['cuetype']
            current_run_list = subject_recall_features.iloc[run, 0]['list']
            if current_run_type == cue_type:
                match current_run_list:
                    case 'A':
                        list_A += np.array(spc[run])
                        count_A += 1
                    case 'B':
                        list_B += np.array(spc[run])
                        count_B += 1

        # Avoid division by zero
        if count_A > 0:
            list_A /= count_A
        else: list_A = None
        if count_B > 0:
            list_B /= count_B
        else: list_B = None
        if (plot == 'y'): plot_spc(list_A, list_B)
        return list_A, list_B



        