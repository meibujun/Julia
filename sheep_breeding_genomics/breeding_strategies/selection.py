# sheep_breeding_genomics/breeding_strategies/selection.py

import pandas as pd
import numpy as np

def rank_animals(ebv_df: pd.DataFrame, ebv_col: str = 'EBV', animal_id_col: str = None, ascending: bool = False) -> pd.DataFrame:
    """
    Ranks animals based on their Estimated Breeding Values (EBVs) or other scores.

    Args:
        ebv_df (pd.DataFrame): DataFrame containing animal IDs and their EBVs.
                               If animal_id_col is provided, it's used. Otherwise, assumes index is animal ID.
        ebv_col (str): Name of the column containing the EBV or score to rank by.
        animal_id_col (str, optional): Name of the column containing animal IDs.
                                       If None, the DataFrame index is assumed to be animal IDs.
        ascending (bool): Sort order. False for descending (higher EBV is better), True for ascending.

    Returns:
        pd.DataFrame: DataFrame sorted by EBV, with a new column 'Rank'.
                      Returns an empty DataFrame if input is invalid or ebv_col is not found.
    """
    if not isinstance(ebv_df, pd.DataFrame) or ebv_df.empty:
        print("Error: Input ebv_df must be a non-empty Pandas DataFrame.")
        return pd.DataFrame()

    if ebv_col not in ebv_df.columns:
        print(f"Error: EBV column '{ebv_col}' not found in the DataFrame.")
        return pd.DataFrame()

    if animal_id_col and animal_id_col not in ebv_df.columns:
        print(f"Error: Animal ID column '{animal_id_col}' not found in DataFrame when specified.")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original DataFrame
    ranked_df = ebv_df.copy()

    # Ensure animal ID is available for reference, using index if col not specified
    if animal_id_col is None:
        # If index has no name, give it one for clarity if we were to reset_index
        # However, for ranking, we can just sort and add rank.
        # If we need to refer to ID later, it's good to have it named or as a column.
        if ranked_df.index.name is None:
            ranked_df.index.name = 'AnimalID' # Default name if using index
        # animal_ids_present = True # Handled by ranked_df structure

    ranked_df.sort_values(by=ebv_col, ascending=ascending, inplace=True)
    ranked_df['Rank'] = range(1, len(ranked_df) + 1)

    return ranked_df


def select_top_n(ranked_ebv_df: pd.DataFrame, n_select: int) -> pd.DataFrame:
    """
    Selects the top N animals from a ranked DataFrame.

    Args:
        ranked_ebv_df (pd.DataFrame): DataFrame of animals, already ranked (e.g., by rank_animals function).
                                      Assumes higher ranks (e.g., 1, 2, ...) are better if standard ranking.
                                      Or, more simply, it takes the top N rows after sorting.
        n_select (int): The number of top animals to select.

    Returns:
        pd.DataFrame: DataFrame containing the selected top N animals.
                      Returns an empty DataFrame if input is invalid or n_select is out of bounds.
    """
    if not isinstance(ranked_ebv_df, pd.DataFrame): # Allow empty df if n_select is 0
        print("Error: Input ranked_ebv_df must be a Pandas DataFrame.")
        return pd.DataFrame()

    if n_select < 0:
        print("Error: Number of animals to select (n_select) cannot be negative.")
        return pd.DataFrame()

    if n_select == 0:
        print("Info: n_select is 0, returning an empty DataFrame.")
        # Return empty DataFrame with same columns
        return pd.DataFrame(columns=ranked_ebv_df.columns)

    if ranked_ebv_df.empty and n_select > 0:
        print("Warning: ranked_ebv_df is empty, cannot select animals.")
        return pd.DataFrame()

    # Assumes ranked_ebv_df is already sorted with best animals at the top.
    selected_df = ranked_ebv_df.head(n_select)

    if len(selected_df) < n_select and len(selected_df) < len(ranked_ebv_df) :
        # This case should ideally not be hit if head(n_select) is used correctly,
        # unless n_select > len(ranked_ebv_df)
        print(f"Warning: Requested top {n_select} animals, but only {len(selected_df)} were available or selected.")

    return selected_df


def select_top_percent(ranked_ebv_df: pd.DataFrame, percent_select: float) -> pd.DataFrame:
    """
    Selects the top X% of animals from a ranked DataFrame.

    Args:
        ranked_ebv_df (pd.DataFrame): DataFrame of animals, already ranked.
        percent_select (float): The percentage of top animals to select (e.g., 10.0 for top 10%).

    Returns:
        pd.DataFrame: DataFrame containing the selected top X% animals.
                      Returns an empty DataFrame if input is invalid or percent_select is out of bounds.
    """
    if not isinstance(ranked_ebv_df, pd.DataFrame): # Allow empty if percent_select leads to 0
        print("Error: Input ranked_ebv_df must be a Pandas DataFrame.")
        return pd.DataFrame()

    if not (0 <= percent_select <= 100):
        print("Error: Percentage to select must be between 0 and 100.")
        return pd.DataFrame()

    if ranked_ebv_df.empty:
        if percent_select == 0:
             return pd.DataFrame(columns=ranked_ebv_df.columns)
        print("Warning: ranked_ebv_df is empty, cannot select animals by percentage.")
        return pd.DataFrame()

    n_total = len(ranked_ebv_df)
    # Calculate number to select, using np.ceil to round up (e.g., top 10% of 15 animals = 1.5 -> 2 animals)
    # Or use floor/round based on strategy, ceil is common for "at least top X%"
    n_select = int(np.ceil(n_total * (percent_select / 100.0)))

    return select_top_n(ranked_ebv_df, n_select)


if __name__ == '__main__':
    print("--- Selection Module Examples ---")

    # Sample EBV data
    data = {
        'AnimalID': ['S101', 'S102', 'D201', 'D202', 'S103', 'D203', 'S104', 'D204'],
        'GrowthRateEBV': [1.5, 1.2, 0.9, 1.8, 0.5, 1.1, 1.9, 0.8],
        'WoolQualityEBV': [0.5, 0.7, 0.6, 0.4, 0.8, 0.3, 0.6, 0.7]
    }
    ebv_data_df = pd.DataFrame(data)
    print("\nOriginal EBV Data:")
    print(ebv_data_df)

    # --- Test rank_animals ---
    print("\n--- Testing rank_animals ---")
    ranked_by_growth = rank_animals(ebv_data_df, ebv_col='GrowthRateEBV', animal_id_col='AnimalID')
    print("\nRanked by GrowthRateEBV (descending):")
    print(ranked_by_growth)

    ranked_by_wool_asc = rank_animals(ebv_data_df, ebv_col='WoolQualityEBV', animal_id_col='AnimalID', ascending=True)
    print("\nRanked by WoolQualityEBV (ascending):")
    print(ranked_by_wool_asc)

    # Test ranking with index as AnimalID
    ebv_data_indexed_df = ebv_data_df.set_index('AnimalID')
    ranked_indexed = rank_animals(ebv_data_indexed_df, ebv_col='GrowthRateEBV') # animal_id_col is None
    print("\nRanked by GrowthRateEBV (using index as AnimalID):")
    print(ranked_indexed)


    # --- Test select_top_n ---
    print("\n--- Testing select_top_n ---")
    # Use ranked_by_growth (best are at the top)
    top_3_growth = select_top_n(ranked_by_growth, 3)
    print("\nTop 3 animals by GrowthRateEBV:")
    print(top_3_growth)

    top_0_growth = select_top_n(ranked_by_growth, 0)
    print("\nTop 0 animals by GrowthRateEBV (should be empty with columns):")
    print(top_0_growth)

    top_10_growth = select_top_n(ranked_by_growth, 10) # Request more than available
    print("\nTop 10 animals by GrowthRateEBV (more than available):")
    print(top_10_growth)


    # --- Test select_top_percent ---
    print("\n--- Testing select_top_percent ---")
    # ranked_by_growth has 8 animals
    # Top 25% of 8 animals = 2 animals
    top_25_percent_growth = select_top_percent(ranked_by_growth, 25.0)
    print("\nTop 25% animals by GrowthRateEBV (should be 2 animals):")
    print(top_25_percent_growth)

    # Top 30% of 8 animals = ceil(2.4) = 3 animals
    top_30_percent_growth = select_top_percent(ranked_by_growth, 30.0)
    print("\nTop 30% animals by GrowthRateEBV (should be 3 animals):")
    print(top_30_percent_growth)

    top_0_percent_growth = select_top_percent(ranked_by_growth, 0.0)
    print("\nTop 0% animals by GrowthRateEBV (should be empty with columns):")
    print(top_0_percent_growth)

    # Test with empty DataFrame
    empty_df_for_selection = pd.DataFrame(columns=ebv_data_df.columns)
    ranked_empty = rank_animals(empty_df_for_selection, 'GrowthRateEBV', 'AnimalID')
    print(f"\nRanking empty DataFrame: \n{ranked_empty}")
    select_top_n(ranked_empty, 2)
    select_top_percent(ranked_empty, 10.0)

    # Test errors
    print("\n--- Testing error cases ---")
    rank_animals(ebv_data_df, ebv_col='NonExistentCol', animal_id_col='AnimalID')
    select_top_n(ranked_by_growth, -1)
    select_top_percent(ranked_by_growth, 110.0)
    select_top_percent(ranked_by_growth, -10.0)
    rank_animals(pd.DataFrame(), 'EBV') # Empty dataframe
    rank_animals(ebv_data_df, ebv_col='GrowthRateEBV', animal_id_col='WrongIDCol')

    print("\nSelection module examples complete.")
