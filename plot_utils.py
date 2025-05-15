
import sys
import pickle 
import json


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker 
import seaborn as sns 

class PlotUtils:

    @staticmethod
    def plot_accuracy_comparison(
            paths_and_labels, 
            y_limits=None, 
            smoothing_window=None, 
            academic_style=True, 
            total_trials= None, 
            model_name = None,
            column_plot='acc'
            ):
        """
        Plots the 'acc' (accuracy) from multiple pickle files on a single graph for comparison.
        This version avoids try-except blocks.

        Args:
            paths_and_labels (list): A list of dictionaries, where each dictionary
                                    should have a 'path' key (string path to the
                                    pickle file) and a 'label' key (string label for
                                    the plot legend).
            y_limits (tuple, optional): A tuple (min_y, max_y) to set the y-axis limits.
                                        Defaults to None (matplotlib auto-scales).
                                        Example: (0.6, 0.8)
            smoothing_window (int, optional): The window size for a rolling mean.
                                            If provided, data will be smoothed.
                                            Defaults to None (no smoothing). Example: 5
            academic_style (bool, optional): If True, applies styling to resemble academic papers
                                            (serif font, no markers, no grid). Defaults to True.
        """
        
        if academic_style:
            # Apply styling closer to academic papers
            # Using a common serif font. For true LaTeX look, 'Computer Modern Roman' might be used
            # but requires the font to be installed. 'serif' is a good general fallback.
            plt.style.use('seaborn-v0_8-whitegrid') # Start with a clean base
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['axes.grid'] = False # Explicitly remove grid for axes
            plt.rcParams['savefig.pad_inches'] = 0.1 # Ensure padding is reasonable for saved figures
            # For a more classic look without seaborn styles:
            # plt.style.use('classic') 
            # plt.rcParams['axes.grid'] = False
        else:
            # Revert to default or a different preferred style if academic_style is False
            plt.style.use('default') # Or any other style like 'ggplot'
            plt.rcParams['font.family'] = 'sans-serif' # Common default
            plt.rcParams['axes.grid'] = True


        # Create a single figure for all plots
        plt.figure(figsize=(8, 7)) # Adjusted figure size slightly
        
        plot_colors = plt.cm.get_cmap('tab10').colors 
        color_index = 0
        plotted_anything = False 

        for item in paths_and_labels:
            path = item.get('path')
            label = item.get('label', path) 
            type = item.get('type')

            if not path:
                print(f"Warning: Configuration item missing 'path': {item}. Skipping.")
                continue
            
            print(f"\nProcessing: {label} (File: {path})")

            if type == 'pickle':
                with open(path, 'rb') as file:
                    loaded_object = pickle.load(file)

                df = None 
                if isinstance(loaded_object, list) and loaded_object and isinstance(loaded_object[0], pd.DataFrame):
                    df = loaded_object[0].copy() 
                    print(f"  DataFrame successfully extracted for {label}.")
                elif isinstance(loaded_object, pd.DataFrame):
                    df = loaded_object.copy()
                    print(f"  DataFrame successfully loaded directly for {label} (was not in a list).")
                else:
                    print(f"  FATAL Error: Loaded object from '{path}' for {label} is not structured as expected.")
                    print(f"    Loaded object type: {type(loaded_object)}")
                    if isinstance(loaded_object, list) and loaded_object:
                        print(f"    First element type: {type(loaded_object[0])}")
                    sys.exit(f"Exiting due to unexpected data structure in {path}.")

            else: 
                df = pd.read_csv(path)


            df = df.reset_index(drop=True)
            trial_numbers = df.index
            available_cols = df.columns


            if column_plot in available_cols:
                accuracy_data = df[column_plot].copy() 

                if total_trials: 
                    accuracy_data = accuracy_data[:total_trials]
                    trial_numbers = df.index[:total_trials]

                # --- Option 1: Remove or Cap Outliers (Manual Data Intervention) ---
                # Example: accuracy_data = accuracy_data.clip(lower=0.60) 

                # --- Option 2: Apply Smoothing (Rolling Mean) ---
                if smoothing_window and isinstance(smoothing_window, int) and smoothing_window > 0:
                    accuracy_data = accuracy_data.rolling(window=smoothing_window, min_periods=1).mean()
                    print(f"  Applied smoothing with window {smoothing_window} for {label}.")

                current_color = plot_colors[color_index % len(plot_colors)]
                
                # Plotting style adjustments for academic look
                plot_marker = '.' if not academic_style else '' # No markers for academic style
                
                plt.plot(trial_numbers, accuracy_data, marker=plot_marker, linestyle='-', label=label, color=current_color)
                color_index += 1
                plotted_anything = True
                print(f"  Successfully plotted {column_plot} for {label}.")
            else:
                print(f"  Column {column_plot} not found in DataFrame for {label}. Skipping accuracy plot for this file.")
            
        if plotted_anything:
            plt.title(f'Accuracy over Optimization Trials for Blood Transfusion {model_name}', fontsize=14) # Slightly smaller title
            plt.xlabel('Trial Number', fontsize=12)
            plt.ylabel('Accuracy (acc)', fontsize=12)
            
            if y_limits and isinstance(y_limits, tuple) and len(y_limits) == 2:
                plt.ylim(y_limits)
                print(f"\nSet Y-axis limits to: {y_limits}")

            # Grid is now controlled by rcParams based on academic_style
            # if not academic_style:
            #     plt.grid(True, linestyle='--', alpha=0.6) # Only show grid if not academic style

            plt.legend(loc='best', fontsize=10) 
            plt.tight_layout() 
            plt.show()
        else:
            print("\nNo accuracy data was found or plotted from any of the provided files.")


    @staticmethod
    def plot_best_accuracy_comparison(
            paths_and_labels, 
            y_limits=None, 
            smoothing_window=None, 
            academic_style=True, 
            total_trials=None, 
            plot_best_at_each_step=True,
            model_name = None,
            column_plot='acc'
            ): # Added new parameter
        """
        Plots the 'acc' (accuracy) from multiple pickle or CSV files on a single graph for comparison.
        Optionally plots the best performing score at each step.

        Args:
            paths_and_labels (list): A list of dictionaries, where each dictionary
                                    should have a 'path' key (string path to the
                                    file), 'label' key (string label for
                                    the plot legend), and optionally 'type' key
                                    ('pickle' or 'csv', defaults to 'pickle' if 'path'
                                    ends with '.pkl' or '.pickle', else 'csv').
            y_limits (tuple, optional): A tuple (min_y, max_y) to set the y-axis limits.
                                        Defaults to None (matplotlib auto-scales).
                                        Example: (0.6, 0.8)
            smoothing_window (int, optional): The window size for a rolling mean.
                                            If provided, data will be smoothed.
                                            Defaults to None (no smoothing). Example: 5
            academic_style (bool, optional): If True, applies styling to resemble academic papers
                                            (serif font, no markers, no grid). Defaults to True.
            total_trials (int, optional): Crop data to this number of trials. Defaults to None.
            plot_best_at_each_step (bool, optional): If True, plots the cumulative maximum
                                                    accuracy at each trial. Defaults to True.
        """

        if academic_style:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['axes.grid'] = False
            plt.rcParams['savefig.pad_inches'] = 0.1
        else:
            plt.style.use('default')
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.grid'] = True

        plt.figure(figsize=(8, 7))
        
        # Using a standard colormap like 'tab10' or 'viridis'
        # For more colors if many lines: plt.cm.get_cmap('nipy_spectral', len(paths_and_labels))
        cmap = plt.cm.get_cmap('tab10')
        plot_colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in range(cmap.N)]

        color_index = 0
        plotted_anything = False

        for item in paths_and_labels:
            path = item.get('path')
            label = item.get('label', path)
            # Infer type if not provided
            file_type = item.get('type')
            if not file_type and path:
                if path.lower().endswith(('.pkl', '.pickle')):
                    file_type = 'pickle'
                elif path.lower().endswith('.csv'):
                    file_type = 'csv'
                else:
                    print(f"Warning: Could not infer file type for {path}. Please specify 'type' ('pickle' or 'csv'). Skipping.")
                    continue
            elif not path:
                print(f"Warning: Configuration item missing 'path': {item}. Skipping.")
                continue


            print(f"\nProcessing: {label} (File: {path}, Type: {file_type})")

            df = None
            try:
                if file_type == 'pickle':
                    with open(path, 'rb') as file:
                        loaded_object = pickle.load(file)

                    if isinstance(loaded_object, list) and loaded_object and isinstance(loaded_object[0], pd.DataFrame):
                        df = loaded_object[0].copy()
                        print(f"  DataFrame successfully extracted for {label}.")
                    elif isinstance(loaded_object, pd.DataFrame):
                        df = loaded_object.copy()
                        print(f"  DataFrame successfully loaded directly for {label} (was not in a list).")
                    else:
                        print(f"  ERROR: Loaded object from '{path}' for {label} is not structured as expected.")
                        print(f"    Loaded object type: {type(loaded_object)}")
                        if isinstance(loaded_object, list) and loaded_object:
                            print(f"    First element type: {type(loaded_object[0])}")
                        continue # Skip this file
                elif file_type == 'csv':
                    df = pd.read_csv(path)
                    print(f"  DataFrame successfully loaded from CSV for {label}.")
                else:
                    print(f"  ERROR: Unknown file type '{file_type}' for {path}. Skipping.")
                    continue
            except FileNotFoundError:
                print(f"  ERROR: File not found at '{path}' for {label}. Skipping.")
                continue
            except Exception as e:
                print(f"  ERROR: Could not load or process file '{path}' for {label}: {e}. Skipping.")
                continue

            if df is None: # Should be caught by earlier checks, but as a safeguard
                continue

            df = df.reset_index(drop=True)
            trial_numbers = df.index # Base trial numbers from 0 to n-1

            available_cols = df.columns

            if column_plot in available_cols:
                accuracy_data = df[column_plot].copy().astype(float) # Ensure numeric type

                # Apply total_trials limit first
                if total_trials is not None and total_trials > 0 and total_trials < len(accuracy_data):
                    accuracy_data = accuracy_data.iloc[:total_trials]
                    trial_numbers = trial_numbers[:total_trials]
                    print(f"  Data truncated to first {total_trials} trials for {label}.")
                
                # --- MODIFICATION: Plot best performing score at each step ---
                if plot_best_at_each_step:
                    accuracy_data = accuracy_data.cummax()
                    print(f"  Applied cumulative maximum to accuracy for {label}.")
                # --- END MODIFICATION ---

                if smoothing_window and isinstance(smoothing_window, int) and smoothing_window > 0:
                    if len(accuracy_data) >= smoothing_window:
                        accuracy_data = accuracy_data.rolling(window=smoothing_window, min_periods=1).mean()
                        print(f"  Applied smoothing with window {smoothing_window} for {label}.")
                    elif len(accuracy_data) > 0 : # Apply if window is larger but data exists
                        accuracy_data = accuracy_data.rolling(window=len(accuracy_data), min_periods=1).mean()
                        print(f"  Applied smoothing with adjusted window {len(accuracy_data)} (data shorter than original window) for {label}.")
                    else:
                        print(f"  Skipping smoothing for {label} due to insufficient data points ({len(accuracy_data)}).")


                current_color = plot_colors[color_index % len(plot_colors)]
                
                plot_marker = '.' if not academic_style else ''
                
                # Ensure trial_numbers and accuracy_data have the same length for plotting
                # This should be naturally handled by the slicing above.
                plt.plot(trial_numbers, accuracy_data, marker=plot_marker, linestyle='-', label=label, color=current_color)
                color_index += 1
                plotted_anything = True
                print(f"  Successfully plotted 'acc' for {label}.")
            else:
                print(f"  Column 'acc' not found in DataFrame for {label}. Skipping accuracy plot for this file.")
            

        if plotted_anything:
            title = 'Best Accuracy at Each Optimization Trial' if plot_best_at_each_step else 'Accuracy over Optimization Trials'
            title += f' for Blood Transfusion {model_name}' # Append the specific context
            plt.title(title, fontsize=14)
            plt.xlabel('Trial Number', fontsize=12)
            ylabel_text = 'Best Accuracy (cumulative max of acc)' if plot_best_at_each_step else 'Accuracy (acc)'
            plt.ylabel(ylabel_text, fontsize=12)
            
            if y_limits and isinstance(y_limits, tuple) and len(y_limits) == 2:
                plt.ylim(y_limits)
                print(f"\nSet Y-axis limits to: {y_limits}")

            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo accuracy data was found or plotted from any of the provided files.")

    @staticmethod
    def plot_regret_comparison(paths_and_labels, column_plot='acc', y_limits=None, academic_style=True):
        """
        Calculates and plots approximate normalized regret from multiple pickle files
        on a single graph for comparison. Assumes higher is better for performance_col.

        Args:
            paths_and_labels (list): List of dicts, each with 'path' and 'label'.
            performance_col (str, optional): Name of the column to use for performance.
                                            Defaults to 'acc'.
            y_limits (tuple, optional): (min_y, max_y) for y-axis. Defaults to None.
            academic_style (bool, optional): If True, applies academic styling.
                                            Defaults to True.
        """
        if academic_style:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['axes.grid'] = False
            plt.rcParams['savefig.pad_inches'] = 0.1
        else:
            plt.style.use('default')
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.grid'] = True

        plt.figure(figsize=(7, 5))
        plot_colors = plt.cm.get_cmap('tab10').colors
        color_index = 0
        plotted_anything = False

        for item in paths_and_labels:
            path = item.get('path')
            label = item.get('label', path)
            type = item.get('type', path)

            if not path:
                print(f"Warning: Configuration item missing 'path': {item}. Skipping.")
                continue
            
            print(f"\nProcessing Regret for: {label} (File: {path})")

            if type == 'pickle':
                # --- Load DataFrame ---
                with open(path, 'rb') as file:
                    loaded_object = pickle.load(file)

                df = None
                if isinstance(loaded_object, list) and loaded_object and isinstance(loaded_object[0], pd.DataFrame):
                    df = loaded_object[0].copy()
                    print(f"  DataFrame successfully extracted for {label}.")
                elif isinstance(loaded_object, pd.DataFrame):
                    df = loaded_object.copy()
                    print(f"  DataFrame successfully loaded directly for {label}.")
                else:
                    print(f"  FATAL Error: Loaded object from '{path}' for {label} is not structured as expected.")
                    print(f"    Loaded object type: {type(loaded_object)}")
                    if isinstance(loaded_object, list) and loaded_object:
                        print(f"    First element type: {type(loaded_object[0])}")
                    sys.exit(f"Exiting due to unexpected data structure in {path}.")
            else: 
                df = pd.read_csv(path)

            # --- Regret Calculation Logic (per file) ---
            if column_plot not in df.columns:
                print(f"  Error: Column '{column_plot}' not found for {label}. Skipping regret plot for this file.")
                continue
            
            # Ensure the performance column is numeric
            # No try-except here as per user request; will error if conversion fails.
            df[column_plot] = pd.to_numeric(df[column_plot])

            df = df.reset_index(drop=True)
            trial_numbers = df.index

            s_min_observed = df[column_plot].min()
            s_max_observed = df[column_plot].max()
            cumulative_best_max_score = df[column_plot].cummax()
            observed_range = s_max_observed - s_min_observed

            normalized_regret_approx = pd.Series(np.zeros(len(df)), index=df.index) # Default to 0
            if observed_range > 1e-9: # Avoid division by zero or near-zero
                gap_to_best_observed = s_max_observed - cumulative_best_max_score
                normalized_regret_approx = gap_to_best_observed / observed_range
            else:
                print(f"  Warning: Observed range for '{column_plot}' in {label} is zero or near-zero. Regret set to 0.")
            
            print(f"  Regret for {label} - Observed Min: {s_min_observed:.4f}, Max: {s_max_observed:.4f}, Range: {observed_range:.4f}")

            # --- Plotting this file's regret ---
            current_color = plot_colors[color_index % len(plot_colors)]
            plot_marker = '.' if not academic_style else ''
            
            plt.plot(trial_numbers, normalized_regret_approx, marker=plot_marker, linestyle='-', label=label, color=current_color)
            color_index += 1
            plotted_anything = True
            print(f"  Successfully plotted regret for {label}.")

        # --- Finalize Plot ---
        if plotted_anything:
            title = f'Normalized Regret vs. Trials'
            plt.title(title, fontsize=14)
            plt.xlabel('Trial Number', fontsize=12)
            plt.ylabel('Normalized Regret', fontsize=12)
            
            if y_limits and isinstance(y_limits, tuple) and len(y_limits) == 2:
                plt.ylim(y_limits)
            else:
                plt.ylim(bottom=-0.05, top=1.05) # Default y-limits for normalized regret

            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo regret data was successfully calculated or plotted from any of the provided files.")

    @staticmethod
    def plot_cummulative_explored_points(paths_and_labels, split_col='acc', y_limits=None, academic_style=True, total_trials = None):

        if academic_style:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['axes.grid'] = False
            plt.rcParams['savefig.pad_inches'] = 0.1
        else:
            plt.style.use('default')
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.grid'] = True

        plt.figure(figsize=(8, 7))
        plot_colors = plt.cm.get_cmap('tab10').colors
        color_index = 0
        max_cumulative_count = 0  # To help set y-limits if not provided
        plotted_anything = False

        for item in paths_and_labels:
            path = item.get('path')
            label = item.get('label', path)
            type = item.get('type', path)

            if not path:
                print(f"Warning: Configuration item missing 'path': {item}. Skipping.")
                continue
            
            print(f"\nProcessing Regret for: {label} (File: {path})")

            if type == 'pickle':
                # --- Load DataFrame ---
                with open(path, 'rb') as file:
                    loaded_object = pickle.load(file)

                df = None
                if isinstance(loaded_object, list) and loaded_object and isinstance(loaded_object[0], pd.DataFrame):
                    df = loaded_object[0].copy()
                    print(f"  DataFrame successfully extracted for {label}.")
                elif isinstance(loaded_object, pd.DataFrame):
                    df = loaded_object.copy()
                    print(f"  DataFrame successfully loaded directly for {label}.")
                else:
                    print(f"  FATAL Error: Loaded object from '{path}' for {label} is not structured as expected.")
                    print(f"    Loaded object type: {type(loaded_object)}")
                    if isinstance(loaded_object, list) and loaded_object:
                        print(f"    First element type: {type(loaded_object[0])}")
                    sys.exit(f"Exiting due to unexpected data structure in {path}.")
            else: 
                df = pd.read_csv(path)

        
        #split by 'acc' column


            if df is None:
                # This case should ideally be caught by specific errors above,
                # but as a fallback:
                print(f"  ERROR: DataFrame for {label} could not be loaded. Skipping.")
                continue

            if split_col not in df.columns:
                print(f"  Warning: Column '{split_col}' not found in DataFrame for {label} (File: {path}). Skipping.")
                print(f"    Available columns: {df.columns.tolist()}")
                continue
            
            if total_trials: 
                df = df[:total_trials]

            # --- Calculate cumulative unique entries ---
            cumulative_unique_counts = []
            seen_values = set()
            
            # Ensure the column is treated consistently, e.g., as strings, if appropriate
            # This depends on the nature of your 'acc' column. If it contains complex objects
            # that are not hashable, you might need to convert them to a string representation
            # or handle them specifically. For typical numerical or string data, this should be fine.

            # print(df)

            # print("Hello:")
            # print(df.loc[:, 'time_final':]) 

            if type == 'pickle':
                df = df.loc[:, 'time_final':]
                df = df.iloc[:, 1:]
            else: 
                df = df.loc[:, :'acc']
                
            try:
                for value in df.values:
                    # print(value.tolist())
                    # print(type(value.tolist()))
                    seen_values.add(tuple(value))
                    cumulative_unique_counts.append(len(seen_values))
            except TypeError as te:
                print(f"  ERROR: TypeError while processing column '{split_col}' for {label}. This might be due to unhashable types. Error: {te}. Skipping.")
                continue
            
            if not cumulative_unique_counts:
                print(f"  Warning: No data to plot for {label} after processing '{split_col}'.")
                continue

            # --- Plotting the results ---
            trials = range(1, len(cumulative_unique_counts) + 1) # X-axis: 1 to N
            current_color = plot_colors[color_index % len(plot_colors)]
            plt.plot(trials, cumulative_unique_counts, label=f'{label}', color=current_color, linewidth=2)
            
            print(f"  Plotted {len(cumulative_unique_counts)} data points for {label}.")
            print(f"  Max unique configurations for {label}: {cumulative_unique_counts[-1]}")

            if cumulative_unique_counts and cumulative_unique_counts[-1] > max_cumulative_count:
                max_cumulative_count = cumulative_unique_counts[-1]

            color_index += 1
            plotted_anything = True



        # --- Finalize Plot ---
        if plotted_anything:
            title = f'Cummulative number of explored configurations over trials'
            plt.title(title, fontsize=14)
            plt.xlabel('Trial Number', fontsize=12)
            plt.ylabel('Nr. observed configurations', fontsize=12)
            
            # if y_limits and isinstance(y_limits, tuple) and len(y_limits) == 2:
            #     plt.ylim(y_limits)
            # else:
            #     plt.ylim(bottom=-0.05, top=1.05) # Default y-limits for normalized regret

            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo regret data was successfully calculated or plotted from any of the provided files.")


    @staticmethod
    def plot_attention_heatmap_focused(
        attention_matrix,
        num_tokens_to_show=500, # Max number of tokens for rows/cols to display
        title="Attention Heatmap",
        token_labels=None, # Optional: FULL list of original token labels for the whole sequence
        cmap='viridis',
        global_scale_vmin=0.0, # Default: Ensure scale starts at 0
        global_scale_vmax=1.0 # Default: Ensure scale ends at 1 (good for probabilities)
    ):
        """
        Plots a heatmap for a potentially large attention matrix, focusing on the
        interactions among the last `num_tokens_to_show` tokens, while keeping
        the color scale relative to the specified global_scale_vmin/vmax range.

        Args:
            attention_matrix (np.array): The FULL 2D attention matrix to plot.
            num_tokens_to_show (int): The size of the square window (last N x last N tokens)
                                    to display. If None or >= seq_len, plots the whole matrix.
            title (str): The title for the plot.
            token_labels (list, optional): FULL list of token strings for original axis labels.
            cmap (str): Colormap to use for the heatmap.
            global_scale_vmin (float): Min value for the color scale (often 0 for attention).
            global_scale_vmax (float): Max value for the color scale (often 1 for attention).
        """
        if attention_matrix is None:
            print(f"Warning: Cannot plot None attention matrix for title: {title}")
            return
        if attention_matrix.ndim != 2 or attention_matrix.shape[0] != attention_matrix.shape[1]:
            print(f"Warning: attention_matrix must be square 2D. Got shape {attention_matrix.shape}. Skipping plot.")
            return


        original_seq_len = attention_matrix.shape[0]

        # Determine if we need to plot the full matrix or a focused view
        plot_full_matrix = (num_tokens_to_show is None or
                            num_tokens_to_show <= 0 or
                            num_tokens_to_show >= original_seq_len)

        if plot_full_matrix:
            matrix_to_plot = attention_matrix
            start_index = 0
            plot_title = title + f" (Full {original_seq_len}x{original_seq_len})"
            effective_plot_dim = original_seq_len
        else:
            # Slice the matrix to get the last num_tokens_to_show interactions
            start_index = max(0, original_seq_len - num_tokens_to_show) # Ensure start_index is not negative
            actual_num_shown = original_seq_len - start_index # Handle cases where matrix is smaller than num_tokens_to_show
            matrix_to_plot = attention_matrix[start_index:, start_index:]
            plot_title = title 
            effective_plot_dim = actual_num_shown

        # --- Plotting ---
        # Adjust figure size dynamically based on the plotted dimension, with limits
        fig_base_size = 6
        fig_scale_factor = max(0.01, min(0.03, 10 / effective_plot_dim)) # Adjust scaling factor
        fig_size_inch = max(fig_base_size, effective_plot_dim * fig_scale_factor)
        fig_size_inch = min(fig_size_inch, 25) # Cap max size to prevent enormous plots
        figsize = (fig_size_inch + 2, fig_size_inch) # Add width for colorbar

        plt.figure(figsize=figsize)

        ax = sns.heatmap(
            matrix_to_plot,
            cmap=cmap,
            cbar_kws={'label': 'Attention Weight'},
            square=True, # Keep aspect ratio square for the heatmap cells
            vmin=global_scale_vmin, # <<< Use the global scale min
            vmax=global_scale_vmax, # <<< Use the global scale max
            annot=False # Annotations are usually too dense for large (e.g., 500x500) matrices
        )

        plt.title(plot_title)
        plt.xlabel("Token Index (Attended To / Key)")
        plt.ylabel("Token Index (Attending From / Query)")

        # --- Adjust ticks and labels to show ORIGINAL indices ---
        # Determine tick positions within the plotted submatrix (0 to effective_plot_dim-1)
        num_ticks = min(10, effective_plot_dim) # Aim for ~10 ticks max for clarity
        tick_step = max(1, effective_plot_dim // num_ticks)
        sub_matrix_tick_indices = np.arange(0, effective_plot_dim, step=tick_step)

        # Calculate corresponding original indices
        original_indices_labels = np.arange(start_index, original_seq_len, step=tick_step)

        # Set ticks and labels
        ax.set_xticks(sub_matrix_tick_indices + 0.5) # Center ticks on cells
        ax.set_yticks(sub_matrix_tick_indices + 0.5)
        ax.set_xticklabels(original_indices_labels)
        ax.set_yticklabels(original_indices_labels)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Optional: Add grid lines for very large plots if needed
        # ax.grid(True, which='major', axis='both', linestyle='-', color='k', linewidth=0.1)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_entropy(json_filepath):

        PLOT_X_LOG_SCALE = False # Set True to use log2 scale for x-axis like paper's Fig 3
        SOURCE_TAG_TO_PLOT = 'ACQ' # Ensure we only plot data logged from ACQ steps

        with open(json_filepath, 'r') as f:
            search_info = json.load(f)

        attention_data = search_info.get('acq_attention_results') # Check primary key first

        # --- 2. Prepare DataFrame ---
        df = pd.DataFrame(attention_data)
        print("\n--- Raw Data Sample ---")
        print(df.head())
        print("----------------------")


        # --- 3. Data Cleaning/Filtering ---
        # Ensure required columns exist
        required_cols = {'n_history', 'entropy'}
        if not required_cols.issubset(df.columns):
            print(f"ERROR: DataFrame missing required columns. Needed: {required_cols}. Found: {df.columns}")
            exit()

        # Filter for the specific source if 'source' column exists
        if 'source' in df.columns and SOURCE_TAG_TO_PLOT:
            print(f"Filtering results for source = '{SOURCE_TAG_TO_PLOT}'...")
            df_filtered = df[df['source'] == SOURCE_TAG_TO_PLOT].copy()
            print(f"  {len(df_filtered)} entries remaining after source filtering.")
        else:
            df_filtered = df.copy() # Use all data if no source column or no filter needed

        # Drop rows with NaN entropy or invalid n_history
        initial_rows = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=['entropy'])
        df_filtered = df_filtered[df_filtered['n_history'] >= 0] # Filter out placeholder -1 if used

        if initial_rows > len(df_filtered):
            print(f"Removed {initial_rows - len(df_filtered)} rows with NaN entropy or invalid n_history.")

        if df_filtered.empty:
            print("ERROR: No valid attention data remaining after cleaning.")
            exit()

        # Ensure types are correct
        df_filtered['n_history'] = df_filtered['n_history'].astype(int)
        df_filtered['entropy'] = df_filtered['entropy'].astype(float)


        # --- 4. Aggregate Data ---
        print("\nAggregating data by 'n_history'...")
        # Group by history length and calculate mean, std, and count of entropy values
        agg_df = df_filtered.groupby('n_history')['entropy'].agg(['mean', 'std', 'count']).reset_index()
        agg_df = agg_df.sort_values(by='n_history') # Ensure sorting for plotting

        # Handle cases where std is NaN (only one data point for that n_history) -> replace with 0
        agg_df['std'].fillna(0, inplace=True)

        print("\n--- Aggregated Data ---")
        print(agg_df)
        print("----------------------")


        # --- 5. Plotting ---


        print("Generating plot...")
        plt.style.use('seaborn-v0_8-whitegrid') # Use a visually appealing style
        fig, ax = plt.subplots(figsize=(6, 5)) # Create figure and axes

        # Extract data for plotting
        x = agg_df['n_history']
        y_mean = agg_df['mean']
        y_std = agg_df['std']

        # Plot the mean entropy line
        ax.plot(x, y_mean, marker='o', linestyle='-', label=f'Mean Entropy (Source: {SOURCE_TAG_TO_PLOT})', markersize=5, zorder=3)

        # Add the shaded region for +/- 1 standard deviation
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, label='Mean ± 1 Std Dev', zorder=2)

        # --- Customize Plot ---
        ax.set_xlabel("Number of History Configurations in Prompt (n_history)")
        ax.set_ylabel("Shannon Entropy (bits)")
        ax.set_title("Attention Entropy vs. BO History Length")

        # Optional: Set x-axis to log scale (base 2) like the paper's Figure 3
        if PLOT_X_LOG_SCALE:
            ax.set_xscale('log', base=2)
            # Customize log ticks for better readability
            # Use major ticks for powers of 2
            ax.xaxis.set_major_locator(mticker.LogLocator(base=2.0, numticks=15)) # Adjust numticks as needed
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) # Use normal numbers
            # Add minor ticks between powers of 2 if desired (can make it busy)
            # ax.xaxis.set_minor_locator(mticker.LogLocator(base=2.0, subs='all'))
            # ax.xaxis.set_minor_formatter(mticker.NullFormatter()) # Hide minor labels
            ax.tick_params(axis='x', which='minor', bottom=False) # Hide minor ticks if too cluttered


        # Set reasonable y-axis limits based on data
        min_plot_y = np.floor(max(0, (y_mean - y_std).min())) if not agg_df.empty else 0
        max_plot_y = np.ceil((y_mean + y_std).max()) if not agg_df.empty else 1
        ax.set_ylim(bottom=min_plot_y, top=max_plot_y)
        # Ensure x-axis starts appropriately (e.g., slightly before first data point)
        min_plot_x = x.min()
        if PLOT_X_LOG_SCALE:
            # For log scale, start near the first power of 2 below min_plot_x
            ax.set_xlim(left=max(1, 2**(np.floor(np.log2(min_plot_x))))) # Start at 1 or nearest power of 2
        else:
            ax.set_xlim(left=max(0, min_plot_x - 1)) # Start slightly before first point


        ax.legend()
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='grey')
        if PLOT_X_LOG_SCALE: # Only show minor grid if using log scale
            ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgrey')

        plt.tight_layout() # Adjust layout to prevent labels overlapping


    @staticmethod
    def plot_top_attended_tokens_as_heatmap_strip(
        last_token_dist, # Full 1D attention distribution from the last token
        num_top_tokens_to_show=20,
        title_prefix="Top Attended Tokens by Last Token (Heatmap Strip)",
        original_seq_len=None,
        token_id_to_string_fn=None, # Optional: maps original token index to token string
        cmap='viridis' # e.g., 'Blues', 'PuBu', 'viridis'
    ):
        """
        Identifies the top N tokens attended to by the last token and plots their
        attention weights as a horizontal heatmap strip, ordered by attention strength.

        Args:
            last_token_dist (np.array): The FULL 1D attention distribution from the last token.
            num_top_tokens_to_show (int): How many of the top attended tokens to display.
            title_prefix (str): Prefix for the plot title.
            original_seq_len (int, optional): Original sequence length for context.
            token_id_to_string_fn (function, optional): Function to get token strings.
            cmap (str): Colormap for the heatmap.
        """
        if last_token_dist is None:
            print(f"Warning: Cannot plot None last_token_dist for title: {title_prefix}")
            return

        if original_seq_len is None:
            original_seq_len = len(last_token_dist)

        if original_seq_len == 0:
            print(f"Warning: Original sequence length is 0. Cannot plot for: {title_prefix}")
            return

        # Get indices that would sort the attention distribution in descending order of weight
        sorted_indices_desc = np.argsort(last_token_dist)[::-1]

        # Select the top N tokens: their original indices and their attention weights
        num_to_plot = min(num_top_tokens_to_show, original_seq_len)
        top_original_indices = sorted_indices_desc[:num_to_plot]
        # The corresponding attention weights for these top tokens (already sorted high to low)
        top_attention_weights_sorted = last_token_dist[top_original_indices]

        # Reshape the (already sorted) top attention weights for heatmap plotting
        attention_reshaped = top_attention_weights_sorted.reshape(1, -1)  # Shape (1, num_to_plot)

        # Make squares bigger: Aim for cells to be visually distinct.
        cell_width_inch = 0.8 # Increased for "bigger squares" feel
        fig_width = num_to_plot * cell_width_inch + 2.0 # Base width + chrome
        fig_height = 3.0 # Height for one row of large cells, title, colorbar

        plt.figure(figsize=(fig_width, fig_height))

        ax = sns.heatmap(
            attention_reshaped,
            cmap=cmap,
            annot=True,       # Show attention values in cells
            fmt=".3f",        # Format for the annotations
            linewidths=0.5,    # Add lines between cells
            linecolor='gray',
            cbar_kws={'label': 'Attention Weight', 'orientation': 'horizontal', 'pad': 0.3, 'shrink': 0.7},
            yticklabels=[f"Top {num_to_plot} Attended"], # Label for the single row
            square=True,       # Make cells square
            vmin=0,  # <-- Set minimum value for color scale to 0
            vmax=1   # <-- Set maximum value for color scale to 1
        )

        title = f"{title_prefix}\n(Last Token to Top {num_to_plot} Attended Tokens, Sorted Left-to-Right by Weight)"
        ax.set_title(title, pad=15)
        # ax.set_xlabel("Original Identity of Attended Token (Sorted by Received Attention)")
        ax.set_ylabel("")

        # X-axis ticks and labels: These are the original indices (or string representations)
        # of the tokens that received the highest attention, ordered left-to-right by that attention.
        x_tick_labels = []
        if token_id_to_string_fn:
            # If you have a way to get the actual token strings based on original_input_ids
            # This example assumes token_id_to_string_fn takes the original index
            x_tick_labels = [f"'{token_id_to_string_fn(idx)}'\n(idx {idx})" for idx in top_original_indices]
        else:
            x_tick_labels = [f"{idx}" for idx in top_original_indices]
        
        tick_positions = np.arange(num_to_plot) + 0.5 # Center ticks for heatmap cells
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
                
        plt.tight_layout(rect=[0, 0.05, 1, 0.92]) # Adjust layout
        plt.show()
