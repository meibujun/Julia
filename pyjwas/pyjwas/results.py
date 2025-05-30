"""
Provides utility functions for saving MCMC results, including posterior summaries
and raw MCMC samples, to files.
"""
import pandas as pd
import numpy as np
import os

def save_summary_df(summary_df: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame, typically containing summary statistics, to a CSV file.

    Args:
        summary_df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The complete path (including filename) where the CSV
                         file will be saved. The directory will be created if
                         it doesn't exist.
    
    Raises:
        TypeError: If `summary_df` is not a pandas DataFrame.
        IOError: If an error occurs during file writing.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("Input summary_df must be a pandas DataFrame.")
    
    try:
        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name: # If path includes a directory
            os.makedirs(dir_name, exist_ok=True)
        summary_df.to_csv(file_path, index=False)
        print(f"Summary DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving summary DataFrame to {file_path}: {e}")
        raise

def save_mcmc_samples(samples_dict: dict, base_file_path: str):
    """
    Saves MCMC samples from a dictionary to separate CSV files for each parameter.

    For each parameter in `samples_dict`:
    - If the parameter is 'variance_components' (list of dicts), it's converted to a DataFrame.
    - Otherwise (e.g., 'beta', 'g', which are lists of NumPy arrays/scalars),
      it's converted to a NumPy array and then to a DataFrame.
    The resulting DataFrame is saved to a CSV file named `"{base_file_path}_{param_name}.csv"`.
    Directories in `base_file_path` will be created if they don't exist.

    Args:
        samples_dict (dict): A dictionary where keys are parameter names (str)
                             and values are lists of samples (e.g., list of dicts for
                             variance components, list of np.ndarray for beta/g).
        base_file_path (str): The base path and filename prefix for the output CSV files.
                              Example: "output/run1/mcmc_samples" will produce files like
                              "output/run1/mcmc_samples_beta.csv".

    Raises:
        TypeError: If `samples_dict` is not a dictionary.
        IOError: If an error occurs during file writing for any parameter.
    """
    if not isinstance(samples_dict, dict):
        raise TypeError("Input samples_dict must be a dictionary.")

    # Ensure directory exists for base_file_path
    dir_name = os.path.dirname(base_file_path)
    if dir_name: # If path includes a directory
        os.makedirs(dir_name, exist_ok=True)

    for param_name, sample_list in samples_dict.items():
        if not sample_list:
            print(f"No samples found for {param_name}, skipping file save.")
            continue

        file_path = f"{base_file_path}_{param_name}.csv"
        
        try:
            if param_name == 'variance_components':
                # sample_list is a list of dicts
                df = pd.DataFrame(sample_list)
            else: # For 'beta', 'g', etc., sample_list is a list of np.arrays
                np_array = np.array(sample_list)
                # If g is a list of 1D arrays, np_array will be 2D.
                # If beta is a list of 1D arrays, np_array will be 2D.
                # If beta is a list of scalars (e.g. single fixed effect), np_array will be 1D.
                if np_array.ndim == 1:
                    np_array = np_array.reshape(-1, 1) # Ensure 2D for DataFrame
                df = pd.DataFrame(np_array)
            
            df.to_csv(file_path, index=False, header=True) # Add header for clarity
            print(f"Samples for {param_name} successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving samples for {param_name} to {file_path}: {e}")
            # Optionally re-raise, or continue to save other parameters
            # raise
    print("Finished saving MCMC samples.")
