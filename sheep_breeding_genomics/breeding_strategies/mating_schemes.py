# sheep_breeding_genomics/breeding_strategies/mating_schemes.py

import pandas as pd
import numpy as np
import random # For random mating selections

# For example usage, we might need to calculate NRM.
# This assumes the project structure allows this relative import.
# from ..genetic_evaluation.relationship_matrix import calculate_nrm
# For now, the example will create a placeholder relationship matrix directly.


def random_mating(selected_sires_df: pd.DataFrame,
                  selected_dams_df: pd.DataFrame,
                  sire_id_col: str = 'AnimalID',
                  dam_id_col: str = 'AnimalID',
                  n_matings_per_sire: int = None,
                  n_progeny_per_mating: int = 1,
                  avoid_self_mating: bool = True,
                  max_total_matings: int = None) -> list:
    """
    Generates random mating pairs from selected sires and dams.

    Args:
        selected_sires_df (pd.DataFrame): DataFrame of selected sires. Must contain sire_id_col.
        selected_dams_df (pd.DataFrame): DataFrame of selected dams. Must contain dam_id_col.
        sire_id_col (str): Column name for sire identifiers.
        dam_id_col (str): Column name for dam identifiers.
        n_matings_per_sire (int, optional): Number of unique dams each sire is mated to.
                                            If None, sires are used potentially multiple times based on dam availability
                                            and overall mating limits.
        n_progeny_per_mating (int): Number of progeny to list for each successful sire-dam pair.
        avoid_self_mating (bool): If True, prevents mating an animal to itself if its ID appears in both lists.
                                  (More relevant if sires and dams lists could overlap significantly).
        max_total_matings (int, optional): Maximum total number of unique sire-dam pairs to generate.
                                           If None, limited by n_matings_per_sire or total available unique pairs.

    Returns:
        list: A list of tuples, where each tuple is (sire_id, dam_id, progeny_number).
              Returns an empty list if no valid matings can be made.
    """
    if not isinstance(selected_sires_df, pd.DataFrame) or selected_sires_df.empty:
        print("Error: selected_sires_df must be a non-empty DataFrame.")
        return []
    if not isinstance(selected_dams_df, pd.DataFrame) or selected_dams_df.empty:
        print("Error: selected_dams_df must be a non-empty DataFrame.")
        return []
    if sire_id_col not in selected_sires_df.columns:
        print(f"Error: Sire ID column '{sire_id_col}' not found in selected_sires_df.")
        return []
    if dam_id_col not in selected_dams_df.columns:
        print(f"Error: Dam ID column '{dam_id_col}' not found in selected_dams_df.")
        return []

    sire_ids = selected_sires_df[sire_id_col].unique().tolist()
    dam_ids = selected_dams_df[dam_id_col].unique().tolist()

    if not sire_ids:
        print("Warning: No unique sire IDs available for mating.")
        return []
    if not dam_ids:
        print("Warning: No unique dam IDs available for mating.")
        return []

    mating_list = []
    used_dams_for_sire = {sire: set() for sire in sire_ids} # Tracks dams already mated to a specific sire to ensure unique pairs per sire

    available_dams_pool = list(dam_ids) # Dams that can still be chosen

    # Determine iteration strategy based on constraints
    if n_matings_per_sire is not None:
        # Iterate through sires, trying to assign n_matings_per_sire to each
        random.shuffle(sire_ids) # Shuffle sires to vary order of selection
        for sire in sire_ids:
            matings_for_this_sire = 0
            # Shuffle available_dams_pool for each sire to ensure randomness if pool is reused
            random.shuffle(available_dams_pool)

            # Potential dams for this sire (dams not yet mated to this sire)
            potential_dams = [d for d in available_dams_pool if d not in used_dams_for_sire[sire]]
            if avoid_self_mating and sire in potential_dams:
                 potential_dams.remove(sire)

            for dam in potential_dams:
                if matings_for_this_sire >= n_matings_per_sire:
                    break # This sire has reached its quota of unique dam matings

                if max_total_matings is not None and len(mating_list) / n_progeny_per_mating >= max_total_matings:
                    break # Reached overall mating limit

                # Record mating
                for prog_num in range(1, n_progeny_per_mating + 1):
                    mating_list.append((sire, dam, prog_num))

                used_dams_for_sire[sire].add(dam)
                matings_for_this_sire += 1

                # Optional: remove dam from global pool if each dam can only be mated once overall
                # if dam in available_dams_pool: available_dams_pool.remove(dam)
                # Current logic: a dam can be mated to multiple sires if not limited by n_matings_per_sire or max_total_matings

            if max_total_matings is not None and len(mating_list) / n_progeny_per_mating >= max_total_matings:
                break # Reached overall mating limit
    else:
        # No per-sire limit, try to generate up to max_total_matings (if set) or exhaust possibilities
        # This becomes more complex: ensure all sires are used? or completely random?
        # For now, let's make it simple: iterate until max_total_matings or reasonable exhaustion.
        # This simple version might not use all sires or dams optimally.

        # Create all possible unique pairs first
        possible_pairs = []
        for sire in sire_ids:
            for dam in dam_ids:
                if avoid_self_mating and sire == dam:
                    continue
                possible_pairs.append((sire, dam))

        random.shuffle(possible_pairs)

        num_pairs_to_form = len(possible_pairs)
        if max_total_matings is not None:
            num_pairs_to_form = min(len(possible_pairs), max_total_matings)

        for sire, dam in possible_pairs[:num_pairs_to_form]:
            for prog_num in range(1, n_progeny_per_mating + 1):
                mating_list.append((sire, dam, prog_num))

    if not mating_list:
        print("No matings generated based on the criteria.")

    return mating_list


def calculate_progeny_inbreeding(sire_id, dam_id, relationship_matrix_df: pd.DataFrame) -> float:
    """
    Calculates the expected inbreeding coefficient of progeny from a given sire and dam.
    F_progeny = 0.5 * A(sire, dam), where A is the relationship matrix (NRM).

    Args:
        sire_id: Identifier of the sire.
        dam_id: Identifier of the dam.
        relationship_matrix_df (pd.DataFrame): The relationship matrix (e.g., NRM) containing
                                               coefficients between parents. Index and columns should be animal IDs.

    Returns:
        float: The expected inbreeding coefficient of the progeny. Returns np.nan if relationship is not found.
    """
    if not isinstance(relationship_matrix_df, pd.DataFrame):
        print("Error: relationship_matrix_df must be a Pandas DataFrame.")
        return np.nan

    if sire_id not in relationship_matrix_df.index or dam_id not in relationship_matrix_df.columns:
        # print(f"Warning: Sire ID {sire_id} or Dam ID {dam_id} not found in relationship matrix. Cannot calculate inbreeding.")
        return np.nan # Or handle as error
    if dam_id not in relationship_matrix_df.index or sire_id not in relationship_matrix_df.columns: # Check other way for safety
        return np.nan

    # Relationship between sire and dam
    try:
        relationship_s_d = relationship_matrix_df.loc[sire_id, dam_id]
    except KeyError:
        # This should be caught by the checks above, but as a safeguard:
        # print(f"Warning: Could not retrieve relationship for Sire {sire_id} and Dam {dam_id}.")
        return np.nan

    # Inbreeding of progeny F_prog = 0.5 * relationship(sire, dam)
    # Note: Relationship A(i,i) = 1 + F_i. So, if using diagonal for parent's own inbreeding, adjust.
    # Here, we need off-diagonal A(sire,dam).
    progeny_inbreeding = 0.5 * relationship_s_d
    return progeny_inbreeding


def generate_mating_list_with_inbreeding(mating_pairs_progeny: list,
                                         relationship_matrix_df: pd.DataFrame,
                                         sire_id_idx: int = 0,
                                         dam_id_idx: int = 1) -> pd.DataFrame:
    """
    Generates a list of mating pairs with their expected progeny inbreeding coefficients.

    Args:
        mating_pairs_progeny (list): A list of tuples, where each tuple contains (sire_id, dam_id, ...other_info...).
        relationship_matrix_df (pd.DataFrame): The relationship matrix (e.g., NRM).
        sire_id_idx (int): Index of sire ID in the mating_pairs_progeny tuples.
        dam_id_idx (int): Index of dam ID in the mating_pairs_progeny tuples.

    Returns:
        pd.DataFrame: DataFrame with columns ['SireID', 'DamID', 'ProgenyInbreeding', ...other_info columns...].
    """
    if not mating_pairs_progeny:
        print("Warning: Empty mating_pairs_progeny list provided.")
        return pd.DataFrame(columns=['SireID', 'DamID', 'ProgenyInbreeding'])

    results = []
    for pair_info_tuple in mating_pairs_progeny:
        sire = pair_info_tuple[sire_id_idx]
        dam = pair_info_tuple[dam_id_idx]

        other_info = [info for i, info in enumerate(pair_info_tuple) if i not in (sire_id_idx, dam_id_idx)]

        inbreeding_coeff = calculate_progeny_inbreeding(sire, dam, relationship_matrix_df)

        current_row = [sire, dam, inbreeding_coeff] + other_info
        results.append(current_row)

    # Determine column names
    # Assume first two are SireID, DamID, then ProgenyInbreeding, then others
    # This part is a bit heuristic if tuple structure varies greatly.
    # A more robust way might be to expect dicts or namedtuples.
    # For now, assume standard (sire, dam, progeny_num_if_any)

    base_cols = ['SireID', 'DamID', 'ProgenyInbreeding']
    num_other_cols = len(mating_pairs_progeny[0]) - 2 # -2 for sire and dam
    other_cols_names = [f'Info{i+1}' for i in range(num_other_cols)]

    final_cols = base_cols + other_cols_names

    output_df = pd.DataFrame(results, columns=final_cols)
    return output_df


if __name__ == '__main__':
    print("--- Mating Schemes Module Examples ---")

    # Sample selected sires and dams (AnimalID column is 'ID')
    sires_data = {'ID': ['S1', 'S2', 'S3'], 'SireEBV': [2.0, 1.8, 2.2]}
    dams_data = {'ID': ['D1', 'D2', 'D3', 'D4', 'D5'], 'DamEBV': [1.5, 1.7, 1.6, 1.9, 1.4]}
    sires_df = pd.DataFrame(sires_data)
    dams_df = pd.DataFrame(dams_data)

    print("\nSelected Sires:")
    print(sires_df)
    print("\nSelected Dams:")
    print(dams_df)

    # --- Test random_mating ---
    print("\n--- Testing random_mating ---")
    # Example 1: Each sire mates to 2 dams, 1 progeny per mating
    matings1 = random_mating(sires_df, dams_df, sire_id_col='ID', dam_id_col='ID',
                             n_matings_per_sire=2, n_progeny_per_mating=1)
    print("\nRandom matings (2 per sire, 1 progeny/mating):")
    for m in matings1[:10]: print(m) # Print first few

    # Example 2: Max 3 total matings, 2 progeny per mating
    matings2 = random_mating(sires_df, dams_df, sire_id_col='ID', dam_id_col='ID',
                             n_progeny_per_mating=2, max_total_matings=3)
    print("\nRandom matings (max 3 unique pairs, 2 progeny/mating):")
    for m in matings2: print(m)

    # Example 3: No specific limits (relies on default behavior)
    matings3 = random_mating(sires_df, dams_df, sire_id_col='ID', dam_id_col='ID', n_progeny_per_mating=1)
    print(f"\nRandom matings (default limits, 1 progeny/mating, total {len(matings3)} pairs generated):")
    # for m in matings3[:10]: print(m)


    # --- Test calculate_progeny_inbreeding and generate_mating_list_with_inbreeding ---
    print("\n--- Testing Inbreeding Calculations ---")
    # Create a sample relationship matrix (NRM)
    # Animals: S1, S2, S3, D1, D2, D3, D4, D5
    animal_ids_for_nrm = ['S1', 'S2', 'S3', 'D1', 'D2', 'D3', 'D4', 'D5']
    nrm_size = len(animal_ids_for_nrm)
    # Create a dummy NRM: diagonal 1.0, S1 related to D1 (0.25), S2 to D2 (0.125)
    # This is a very simplified NRM for example purposes.
    nrm_values = np.identity(nrm_size)
    # S1-D1 relationship
    s1_idx, d1_idx = animal_ids_for_nrm.index('S1'), animal_ids_for_nrm.index('D1')
    nrm_values[s1_idx, d1_idx] = nrm_values[d1_idx, s1_idx] = 0.25
    # S2-D2 relationship
    s2_idx, d2_idx = animal_ids_for_nrm.index('S2'), animal_ids_for_nrm.index('D2')
    nrm_values[s2_idx, d2_idx] = nrm_values[d2_idx, s2_idx] = 0.125

    nrm_example_df = pd.DataFrame(nrm_values, index=animal_ids_for_nrm, columns=animal_ids_for_nrm)
    print("\nSample NRM for Inbreeding Calculation:")
    print(nrm_example_df)

    # Test calculate_progeny_inbreeding
    inbreeding_s1d1 = calculate_progeny_inbreeding('S1', 'D1', nrm_example_df)
    print(f"\nExpected inbreeding for progeny of S1 x D1: {inbreeding_s1d1:.4f} (0.5 * 0.25 = 0.125)")
    inbreeding_s1d2 = calculate_progeny_inbreeding('S1', 'D2', nrm_example_df) # Should be 0 if A(S1,D2)=0
    print(f"Expected inbreeding for progeny of S1 x D2: {inbreeding_s1d2:.4f} (0.5 * 0.0 = 0.0)")
    inbreeding_s_missing = calculate_progeny_inbreeding('SX', 'D1', nrm_example_df) # Sire SX not in NRM
    print(f"Expected inbreeding for progeny of SX x D1: {inbreeding_s_missing} (should be nan)")


    # Use some generated matings (e.g., matings1) for generate_mating_list_with_inbreeding
    # Ensure mating_pairs_progeny for this function are (sire, dam, progeny_num)
    # If matings1 is empty due to random chance and small numbers, make a fixed list for test
    if not matings1: # If random mating produced no results due to small numbers/chance
        matings1 = [('S1','D1',1), ('S1','D3',1), ('S2','D2',1), ('S2','D4',1), ('S3','D5',1)]
        print("\nUsing fixed mating list for inbreeding example as random list was empty.")

    mating_list_with_inbreeding_df = generate_mating_list_with_inbreeding(matings1, nrm_example_df)
    print("\nMating List with Expected Progeny Inbreeding:")
    print(mating_list_with_inbreeding_df)

    # Test with empty mating list
    empty_inbreeding_list = generate_mating_list_with_inbreeding([], nrm_example_df)
    print("\nInbreeding list for empty matings (should be empty DataFrame):")
    print(empty_inbreeding_list)

    print("\nMating schemes module examples complete.")
