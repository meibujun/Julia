"""
Mating Plan Assistance functions for the Sheep Breeding Management System.

This module provides functions to help in making informed mating decisions
by calculating expected inbreeding coefficients, expected progeny EBVs,
and suggesting potential mates based on selection criteria.

These functions conceptually use outputs from `genetic_evaluator.py` (A matrix, EBVs)
and `data_manager.py` (animal details), but do not directly call them.
Necessary data structures like the relationship matrix and EBV dictionaries
are expected as inputs.
"""
import numpy as np

def calculate_expected_progeny_ebv(sire_ebvs: dict, dam_ebvs: dict, traits_of_interest: list):
    """
    Calculates the expected Estimated Breeding Value (EBV) for progeny.
    The expected progeny EBV is the average of the parent EBVs for each trait.

    Args:
        sire_ebvs: A dictionary mapping trait_id to EBV for the sire.
                   Example: {trait1_id: ebv1_sire, trait2_id: ebv2_sire}
        dam_ebvs: A dictionary mapping trait_id to EBV for the dam.
                 Example: {trait1_id: ebv1_dam, trait2_id: ebv2_dam}
        traits_of_interest: A list of trait_ids for which to calculate
                            expected progeny EBVs.

    Returns:
        A dictionary mapping trait_id to the expected progeny EBV.
        Returns an empty dictionary if inputs are problematic or no common traits with EBVs.
    """
    if not sire_ebvs or not dam_ebvs or not traits_of_interest:
        return {}

    progeny_ebvs = {}
    for trait_id in traits_of_interest:
        sire_trait_ebv = sire_ebvs.get(trait_id)
        dam_trait_ebv = dam_ebvs.get(trait_id)

        if sire_trait_ebv is not None and dam_trait_ebv is not None:
            progeny_ebvs[trait_id] = 0.5 * (sire_trait_ebv + dam_trait_ebv)
        else:
            # Optionally, log a warning or handle missing EBVs for a trait differently
            print(f"Warning: EBV for trait {trait_id} not available for both parents.")
            progeny_ebvs[trait_id] = None # Or skip this trait

    return {k: v for k, v in progeny_ebvs.items() if v is not None}


def calculate_expected_inbreeding(sire_id, dam_id,
                                  relationship_matrix: np.ndarray,
                                  animal_to_matrix_idx_map: dict):
    """
    Calculates the expected inbreeding coefficient of the progeny from a given sire and dam.
    F_progeny = 0.5 * A_sire_dam (relationship coefficient between sire and dam).

    Args:
        sire_id: The ID of the sire.
        dam_id: The ID of the dam.
        relationship_matrix: The additive relationship matrix (A).
        animal_to_matrix_idx_map: A dictionary mapping animal_id to its
                                     index in the relationship_matrix.

    Returns:
        The expected inbreeding coefficient of the progeny (float).
        Returns None if sire or dam is not found in the relationship matrix map,
        or if the relationship matrix is not valid.
    """
    if relationship_matrix is None or not animal_to_matrix_idx_map:
        return None
    if sire_id not in animal_to_matrix_idx_map or dam_id not in animal_to_matrix_idx_map:
        print(f"Warning: Sire ID {sire_id} or Dam ID {dam_id} not found in relationship matrix map.")
        return None

    sire_idx = animal_to_matrix_idx_map[sire_id]
    dam_idx = animal_to_matrix_idx_map[dam_id]

    if not (0 <= sire_idx < relationship_matrix.shape[0] and \
            0 <= dam_idx < relationship_matrix.shape[1]):
        print(f"Warning: Sire index {sire_idx} or Dam index {dam_idx} is out of bounds for the relationship matrix.")
        return None

    # A_sire_dam is the relationship between sire and dam
    a_sire_dam = relationship_matrix[sire_idx, dam_idx]

    # Expected inbreeding of progeny
    f_progeny = 0.5 * a_sire_dam
    return f_progeny


def suggest_mates(target_animals_ids: list,
                  potential_mates_ids: list,
                  trait_ebvs: dict,
                  relationship_matrix: np.ndarray,
                  animal_to_matrix_idx_map: dict,
                  selection_criteria: dict,
                  max_inbreeding_threshold: float):
    """
    Suggests potential mates for a list of target animals based on selection
    criteria and an inbreeding threshold.

    Args:
        target_animals_ids: List of IDs for animals for whom mates are sought (e.g., ewes).
        potential_mates_ids: List of IDs for potential mates (e.g., rams).
        trait_ebvs: Dictionary mapping animal_id to a sub-dictionary of
                      {trait_id: ebv_value}.
                      Example: {
                          ewe1_id: {trait1: 0.5, trait2: -0.2},
                          ram1_id: {trait1: 0.7, trait2: 0.1}, ...
                      }
        relationship_matrix: The full additive relationship matrix (A).
        animal_to_matrix_idx_map: Maps animal_id to matrix index.
        selection_criteria: Dictionary defining trait weights for ranking.
                            Example: {trait1_id: 0.6, trait2_id: 0.4}
                            Traits not in criteria are ignored for ranking.
        max_inbreeding_threshold: Maximum allowable inbreeding coefficient for progeny.

    Returns:
        A dictionary where keys are target_animal_id and values are lists of
        suggested mate objects, ranked by selection index. Each suggestion is a dict:
        {
            'mate_id': potential_mate_id,
            'expected_inbreeding': float,
            'expected_progeny_ebvs': {trait_id: ebv_value, ...},
            'selection_index': float
        }
    """
    suggestions = {}
    traits_for_ranking = list(selection_criteria.keys())

    for target_id in target_animals_ids:
        if target_id not in trait_ebvs:
            print(f"Warning: EBVs not available for target animal {target_id}. Skipping.")
            continue

        target_ebvs = trait_ebvs[target_id]
        potential_ranked_mates = []

        for mate_id in potential_mates_ids:
            if mate_id == target_id: # Cannot mate with self
                continue
            if mate_id not in trait_ebvs:
                print(f"Warning: EBVs not available for potential mate {mate_id}. Skipping this mate for target {target_id}.")
                continue

            mate_ebvs = trait_ebvs[mate_id]

            # 1. Calculate expected inbreeding
            expected_inbreeding = calculate_expected_inbreeding(
                target_id, mate_id, relationship_matrix, animal_to_matrix_idx_map
            )

            if expected_inbreeding is None:
                print(f"Warning: Could not calculate inbreeding for pair ({target_id}, {mate_id}). Skipping.")
                continue

            if expected_inbreeding > max_inbreeding_threshold:
                continue # Skip this pairing due to high inbreeding

            # 2. Calculate expected progeny EBVs for traits relevant to selection criteria
            expected_progeny_ebvs = calculate_expected_progeny_ebv(
                target_ebvs, mate_ebvs, traits_for_ranking
            )

            if not expected_progeny_ebvs:
                 print(f"Warning: Could not calculate progeny EBVs for pair ({target_id}, {mate_id}) for ranking traits. Skipping.")
                 continue


            # 3. Calculate selection index (weighted sum of expected progeny EBVs)
            selection_index = 0.0
            valid_criteria_count = 0
            for trait_id, weight in selection_criteria.items():
                progeny_trait_ebv = expected_progeny_ebvs.get(trait_id)
                if progeny_trait_ebv is not None:
                    selection_index += weight * progeny_trait_ebv
                    valid_criteria_count +=1

            if valid_criteria_count == 0 and selection_criteria:
                print(f"Warning: For pair ({target_id}, {mate_id}), no progeny EBVs available for any traits in selection criteria. Index is 0.")
                # continue # Or assign a very low index / handle as per specific rules


            potential_ranked_mates.append({
                'mate_id': mate_id,
                'expected_inbreeding': expected_inbreeding,
                'expected_progeny_ebvs': expected_progeny_ebvs, # Store EBVs used for ranking
                'selection_index': selection_index
            })

        # Sort potential mates by selection index (descending)
        potential_ranked_mates.sort(key=lambda x: x['selection_index'], reverse=True)
        suggestions[target_id] = potential_ranked_mates

    return suggestions


if __name__ == '__main__':
    print("mating_planner.py loaded.")
    print("This module contains functions for mating plan assistance.")

    # --- Example Data (Conceptual - would come from other modules) ---
    # Animal IDs
    ewe1, ewe2 = 1, 2
    ram1, ram2, ram3 = 11, 12, 13

    # Trait IDs
    weaning_weight_trait = 101
    fleece_weight_trait = 102

    # EBVs (AnimalID -> {TraitID: EBV})
    mock_ebvs = {
        ewe1: {weaning_weight_trait: 0.5, fleece_weight_trait: 0.2},
        ewe2: {weaning_weight_trait: 0.3, fleece_weight_trait: 0.1},
        ram1: {weaning_weight_trait: 1.0, fleece_weight_trait: 0.4}, # Good ram
        ram2: {weaning_weight_trait: 0.2, fleece_weight_trait: -0.1},# Average/Poor ram
        ram3: {weaning_weight_trait: 0.8, fleece_weight_trait: 0.5}, # Another good ram
    }

    # Relationship Matrix (A) and map
    # (Simplified: ewe1, ewe2, ram1, ram2, ram3 are indices 0,1,2,3,4)
    # This A matrix is purely illustrative.
    # A real one would be calculated by genetic_evaluator.calculate_additive_relationship_matrix
    mock_animal_ids = [ewe1, ewe2, ram1, ram2, ram3]
    mock_animal_to_idx = {aid: i for i, aid in enumerate(mock_animal_ids)}

    # Example A matrix (diagonal = 1 + F, off-diagonal = relationship)
    # Assume ewe1 and ram1 are related (0.25), ewe1 and ram2 unrelated (0.0)
    # ewe2 and ram1 unrelated (0.0), ewe2 and ram2 related (0.125)
    # ram1 and ram3 are half-sibs (0.25)
    # For simplicity, let's make a generic A matrix.
    # For F_progeny = 0.5 * A_sire_dam, A_sire_dam is the relationship.
    # So, A_sire_dam = 0.25 implies F_progeny = 0.125
    # A_sire_dam = 0.5 implies F_progeny = 0.25 (e.g. full sibs)

    _A = np.array([
    # ewe1, ewe2, ram1, ram2, ram3 (Indices 0, 1, 2, 3, 4)
    [1.00, 0.00, 0.25, 0.00, 0.10], # ewe1
    [0.00, 1.00, 0.00, 0.125,0.05], # ewe2
    [0.25, 0.00, 1.00, 0.00, 0.25], # ram1
    [0.00, 0.125,0.00, 1.00, 0.00], # ram2
    [0.10, 0.05, 0.25, 0.00, 1.00]  # ram3
    ])
    # Symmetrize, although it should be symmetric from calculation
    mock_A_matrix = (_A + _A.T) / 2
    np.fill_diagonal(mock_A_matrix, 1.0 + np.array([0.0, 0.0, 0.0, 0.0, 0.0])) # Example inbreeding for parents


    # --- Test calculate_expected_progeny_ebv ---
    print("\n--- Test Expected Progeny EBV ---")
    progeny_ebvs_ewe1_ram1 = calculate_expected_progeny_ebv(
        mock_ebvs[ewe1], mock_ebvs[ram1], [weaning_weight_trait, fleece_weight_trait]
    )
    print(f"Expected EBVs for progeny of Ewe {ewe1} and Ram {ram1}: {progeny_ebvs_ewe1_ram1}")
    # Expected: WW = 0.5 * (0.5 + 1.0) = 0.75; FW = 0.5 * (0.2 + 0.4) = 0.3

    # --- Test calculate_expected_inbreeding ---
    print("\n--- Test Expected Inbreeding ---")
    # Test Ewe1 (idx 0) with Ram1 (idx 2), A[0,2] = 0.25
    inbreeding_ewe1_ram1 = calculate_expected_inbreeding(
        ewe1, ram1, mock_A_matrix, mock_animal_to_idx
    )
    print(f"Expected inbreeding for progeny of Ewe {ewe1} and Ram {ram1}: {inbreeding_ewe1_ram1}")
    # Expected: 0.5 * 0.25 = 0.125

    # Test Ewe1 (idx 0) with Ram2 (idx 3), A[0,3] = 0.0
    inbreeding_ewe1_ram2 = calculate_expected_inbreeding(
        ewe1, ram2, mock_A_matrix, mock_animal_to_idx
    )
    print(f"Expected inbreeding for progeny of Ewe {ewe1} and Ram {ram2}: {inbreeding_ewe1_ram2}")
    # Expected: 0.5 * 0.0 = 0.0

    # Test with an animal not in map
    inbreeding_unknown = calculate_expected_inbreeding(
        ewe1, 999, mock_A_matrix, mock_animal_to_idx # 999 is not in map
    )
    print(f"Expected inbreeding with unknown animal: {inbreeding_unknown}") # Expected: None

    # --- Test suggest_mates ---
    print("\n--- Test Suggest Mates ---")
    target_ewes = [ewe1, ewe2]
    potential_rams = [ram1, ram2, ram3]

    # Selection criteria: Emphasize weaning weight more
    criteria = {
        weaning_weight_trait: 0.7,
        fleece_weight_trait: 0.3
    }
    max_f = 0.15 # Maximum inbreeding allowed (e.g. Ewe1 x Ram1 would be 0.125, so allowed)

    suggested_pairings = suggest_mates(
        target_ewes, potential_rams, mock_ebvs,
        mock_A_matrix, mock_animal_to_idx,
        criteria, max_f
    )

    for target_ewe_id, ranked_mates in suggested_pairings.items():
        print(f"\nSuggestions for Target Ewe {target_ewe_id}:")
        if not ranked_mates:
            print("  No suitable mates found.")
        for mate_info in ranked_mates:
            print(f"  - Mate Ram {mate_info['mate_id']}:")
            print(f"    - Index Score: {mate_info['selection_index']:.4f}")
            print(f"    - Expected Inbreeding: {mate_info['expected_inbreeding']:.4f}")
            print(f"    - Expected Progeny EBVs: {mate_info['expected_progeny_ebvs']}")

    # Example: Ewe1 (0.5 WW, 0.2 FW)
    # Ram1 (1.0 WW, 0.4 FW): ProgEBV (0.75 WW, 0.3 FW), Index = 0.7*0.75 + 0.3*0.3 = 0.525 + 0.09 = 0.615. Inbreeding = 0.125 (OK)
    # Ram2 (0.2 WW, -0.1 FW): ProgEBV (0.35 WW, 0.05 FW), Index = 0.7*0.35 + 0.3*0.05 = 0.245 + 0.015 = 0.26. Inbreeding = 0.0 (OK)
    # Ram3 (0.8 WW, 0.5 FW): ProgEBV (0.65 WW, 0.35 FW), Index = 0.7*0.65 + 0.3*0.35 = 0.455 + 0.105 = 0.56. Inbreeding = 0.5 * A[mock_animal_to_idx[ewe1], mock_animal_to_idx[ram3]] = 0.5 * 0.10 = 0.05 (OK)
    # Expected order for Ewe1: Ram1, Ram3, Ram2
```
