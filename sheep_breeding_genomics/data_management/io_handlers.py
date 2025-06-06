# sheep_breeding_genomics/data_management/io_handlers.py

import pandas as pd
import numpy as np # For np.nan in example usage
from .data_structures import GenomicData # Import the new class

def read_phenotypic_data(file_path: str) -> pd.DataFrame:
    """
    Reads phenotypic data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the phenotypic data.
                      Returns an empty DataFrame if reading fails.
    """
    try:
        phen_data = pd.read_csv(file_path)
        print(f"Successfully read phenotypic data from {file_path}")
        return phen_data
    except FileNotFoundError:
        print(f"Error: Phenotypic data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading phenotypic data from {file_path}: {e}")
        return pd.DataFrame()

def read_pedigree_data(file_path: str) -> pd.DataFrame:
    """
    Reads pedigree data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the pedigree data.
                      Returns an empty DataFrame if reading fails.
    """
    try:
        ped_data = pd.read_csv(file_path)
        print(f"Successfully read pedigree data from {file_path}")
        return ped_data
    except FileNotFoundError:
        print(f"Error: Pedigree data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading pedigree data from {file_path}: {e}")
        return pd.DataFrame()

def save_data_to_csv(dataframe: pd.DataFrame, file_path: str) -> bool:
    """
    Saves a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the CSV file will be saved.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if dataframe.empty:
        print(f"Warning: Attempting to save an empty DataFrame to {file_path}. File will not be created.")
        return False
    try:
        dataframe.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"An error occurred while saving data to {file_path}: {e}")
        return False

def read_genomic_data(file_path: str, animal_id_col: str, **kwargs) -> GenomicData | None:
    """
    Reads genomic data from a CSV file into a GenomicData object.
    Assumes rows are animals and columns are SNPs, with one column for Animal IDs.

    Args:
        file_path (str): The path to the CSV file.
        animal_id_col (str): The name of the column containing animal identifiers.
        **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
        GenomicData | None: A GenomicData object containing the SNP data,
                            or None if reading or validation fails.
    """
    try:
        geno_df = pd.read_csv(file_path, **kwargs)
        if animal_id_col not in geno_df.columns:
            print(f"Error: Animal ID column '{animal_id_col}' not found in {file_path}.")
            return None

        genomic_data_obj = GenomicData(data=geno_df, animal_id_col=animal_id_col)
        print(f"Successfully read genomic data from {file_path} into GenomicData object.")
        return genomic_data_obj
    except FileNotFoundError:
        print(f"Error: Genomic data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading genomic data from {file_path}: {e}")
        return None

def save_genomic_data(genomic_data_df: pd.DataFrame, file_path: str, animal_id_col: str, allow_empty: bool = False) -> bool:
    """
    Saves genomic data from a DataFrame to a CSV file.
    The DataFrame should have animals as rows and SNPs as columns, plus an animal ID column.

    Args:
        genomic_data_df (pd.DataFrame): The DataFrame to save. Must include the animal_id_col.
        file_path (str): The path where the CSV file will be saved.
        animal_id_col (str): The name of the column containing animal identifiers.
        allow_empty (bool): If False (default), trying to save an empty DataFrame (no rows, or only ID column with no SNPs)
                            will result in a warning and return False. If True, allows saving empty or ID-only DataFrames.
                            The `animal_id_col` must still be present if `allow_empty` is True and DataFrame is not entirely empty.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not isinstance(genomic_data_df, pd.DataFrame):
        print("Error: Input data for saving must be a Pandas DataFrame.")
        return False

    if animal_id_col not in genomic_data_df.columns:
        # Allow saving a completely empty DataFrame if allow_empty is True, even if animal_id_col is technically missing because there are no columns.
        if genomic_data_df.empty and allow_empty:
            pass
        else:
            print(f"Error: Animal ID column '{animal_id_col}' not found in DataFrame. Cannot save.")
            return False

    # Check for emptiness based on `allow_empty`
    if not allow_empty:
        if genomic_data_df.shape[0] == 0: # No animals
            print(f"Warning: Attempting to save an empty DataFrame (no animals) to {file_path}. File will not be created as allow_empty=False.")
            return False
        # Has animals, but check if it has SNP columns besides the animal ID column
        snp_columns = [col for col in genomic_data_df.columns if col != animal_id_col]
        if not snp_columns: # No SNP data
            print(f"Warning: Attempting to save DataFrame with no SNP data (only Animal IDs) to {file_path}. File will not be created as allow_empty=False.")
            return False

    try:
        genomic_data_df.to_csv(file_path, index=False)
        print(f"Genomic data successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"An error occurred while saving genomic data to {file_path}: {e}")
        return False

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)

    # Create dummy phenotypic data
    phen_example_data = {'AnimalID': [1, 2, 3], 'Trait1': [10.5, 12.3, 11.0]}
    phen_df = pd.DataFrame(phen_example_data)
    save_data_to_csv(phen_df, 'dummy_phenotypic_data.csv')

    # Read dummy phenotypic data
    phen_data_read = read_phenotypic_data('dummy_phenotypic_data.csv')
    if not phen_data_read.empty:
        print("\\nPhenotypic Data:")
        print(phen_data_read)

    # Create dummy pedigree data
    ped_example_data = {'AnimalID': [1, 2, 3], 'SireID': [0, 1, 1], 'DamID': [0, 0, 2]} # 0 for unknown parent
    ped_df = pd.DataFrame(ped_example_data)
    save_data_to_csv(ped_df, 'dummy_pedigree_data.csv')

    # Read dummy pedigree data
    ped_data_read = read_pedigree_data('dummy_pedigree_data.csv')
    if not ped_data_read.empty:
        print("\\nPedigree Data:")
        print(ped_data_read)

    # Test reading non-existent files
    read_phenotypic_data('non_existent_phen_data.csv')
    read_pedigree_data('non_existent_ped_data.csv')

    # --- Example Usage for Genomic Data I/O ---
    print("\\n--- Genomic Data I/O Examples ---")
    # Create dummy genomic data DataFrame
    geno_example_data_dict = {
        'AnimalID': ['Animal1', 'Animal2', 'Animal3'],
        'SNP_A': [0, 1, 2],
        'SNP_B': [1, np.nan, 0], # With a missing value
        'SNP_C': [2, 2, 1]
    }
    geno_df_to_save = pd.DataFrame(geno_example_data_dict)

    # Save genomic data
    genomic_output_file = 'dummy_genomic_data.csv'
    if save_genomic_data(geno_df_to_save, genomic_output_file, animal_id_col='AnimalID'):
        # Read genomic data back
        loaded_genomic_data_obj = read_genomic_data(genomic_output_file, animal_id_col='AnimalID')
        if loaded_genomic_data_obj and not loaded_genomic_data_obj.data.empty:
            print("\\nLoaded Genomic Data Object:")
            print(loaded_genomic_data_obj) # This will use GenomicData __str__
            print("\\nGenotypes from loaded object:")
            print(loaded_genomic_data_obj.get_genotypes().head())
        else:
            print("Failed to load or parse genomic data correctly after saving.")
    else:
        print("Failed to save genomic data initially.")

    # Test reading non-existent genomic file
    print("\\n--- Test reading non-existent genomic file ---")
    read_genomic_data('non_existent_geno_data.csv', animal_id_col='AnimalID')

    # Test saving/reading empty genomic data (allowed)
    print("\\n--- Test saving empty genomic data (allowed) ---")
    empty_geno_df_allowed = pd.DataFrame(columns=['AnimalID', 'SNP1']) # Has AnimalID column but no rows
    save_genomic_data(empty_geno_df_allowed, 'empty_genomic_output_allowed.csv', 'AnimalID', allow_empty=True)
    loaded_empty_allowed = read_genomic_data('empty_genomic_output_allowed.csv', 'AnimalID')
    if loaded_empty_allowed: # Should be a GenomicData object, possibly with empty .data
       print(f"Loaded empty genomic data (allowed): {loaded_empty_allowed}")

    # Test saving empty genomic data (not allowed - default)
    print("\\n--- Test saving empty genomic data (not allowed by default) ---")
    empty_geno_df_not_allowed = pd.DataFrame({'AnimalID': []}) # No rows
    save_genomic_data(empty_geno_df_not_allowed, 'empty_genomic_output_not_allowed.csv', 'AnimalID') # allow_empty=False by default

    # Test saving genomic data with only ID column (not allowed - default)
    print("\\n--- Test saving genomic data with only ID column (not allowed by default) ---")
    id_only_df = pd.DataFrame({'AnimalID': ['ID1', 'ID2']})
    save_genomic_data(id_only_df, 'id_only_genomic_output.csv', 'AnimalID')
