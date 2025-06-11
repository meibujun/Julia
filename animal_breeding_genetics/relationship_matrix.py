import numpy as np

# Optional: For cleaner matrix printing during debugging, not for production
# np.set_printoptions(precision=4, suppress=True)

def _get_ordered_animal_list_and_map(
    pedigree: list[tuple[int, int | None, int | None]]
) -> tuple[list[int], dict[int, int], dict[int, tuple[int | None, int | None]]]:
    """
    Internal helper to validate pedigree, create an ordered list of animals
    (parents before offspring), and return necessary mappings.

    Args:
        pedigree: List of (animal_id, sire_id, dam_id) tuples.

    Returns:
        A tuple containing:
            - animal_list_final_order: List of unique animal IDs, sorted for processing.
            - animal_to_idx_map: Dictionary mapping animal ID to its matrix index.
            - ped_map: Dictionary mapping animal ID to (sire_id, dam_id).

    Raises:
        TypeError: If pedigree format or ID types are incorrect.
        ValueError: If IDs are non-positive, animal is its own parent, or cycles detected.
    """
    if not isinstance(pedigree, list):
        raise TypeError("Pedigree must be a list of tuples.")
    if not pedigree: # Handle empty pedigree
        return [], {}, {}
    if not all(isinstance(entry, tuple) and len(entry) == 3 for entry in pedigree):
        raise TypeError("Each pedigree entry must be a tuple of (animal, sire, dam).")

    all_animals_in_pedigree = set()
    # Preliminary validation and collection of all IDs
    for animal, sire, dam in pedigree:
        if not isinstance(animal, int) or animal <= 0:
            raise ValueError(f"Animal ID {animal} must be a positive integer.")
        if sire is not None and (not isinstance(sire, int) or sire <= 0):
            raise ValueError(f"Sire ID {sire} for animal {animal} must be a positive integer or None.")
        if dam is not None and (not isinstance(dam, int) or dam <= 0):
            raise ValueError(f"Dam ID {dam} for animal {animal} must be a positive integer or None.")
        if animal == sire or animal == dam:
            raise ValueError(f"Animal {animal} cannot be its own parent.")

        all_animals_in_pedigree.add(animal)
        if sire: all_animals_in_pedigree.add(sire)
        if dam: all_animals_in_pedigree.add(dam)

    animal_list_final_order = []
    animal_to_idx_map = {}

    # ped_map stores parentage for all unique animals found.
    # Animals appearing only as parents will have (None, None) initially if not in individual column.
    ped_map_internal = {entry[0]: (entry[1], entry[2]) for entry in pedigree}
    for animal_id_val in all_animals_in_pedigree:
        if animal_id_val not in ped_map_internal:
            ped_map_internal[animal_id_val] = (None, None)

    num_total_animals = len(all_animals_in_pedigree)
    processed_ids = set()

    while len(animal_list_final_order) < num_total_animals:
        added_in_this_iteration = False
        # Iterate over sorted list of remaining IDs for deterministic behavior
        remaining_ids = sorted(list(all_animals_in_pedigree - processed_ids))

        if not remaining_ids and len(animal_list_final_order) < num_total_animals :
             # This state should ideally not be reached if ValueError for cycle is effective
            raise ValueError("Logic error in ordering: no remaining IDs but not all animals processed.")

        for animal_id in remaining_ids:
            sire_id, dam_id = ped_map_internal.get(animal_id, (None, None))

            sire_ready = (sire_id is None) or (sire_id in animal_to_idx_map)
            dam_ready = (dam_id is None) or (dam_id in animal_to_idx_map)

            if sire_ready and dam_ready:
                new_idx = len(animal_list_final_order)
                animal_to_idx_map[animal_id] = new_idx
                animal_list_final_order.append(animal_id)
                processed_ids.add(animal_id)
                added_in_this_iteration = True

        if not added_in_this_iteration and len(animal_list_final_order) < num_total_animals:
            unprocessed_details = {r: ped_map_internal.get(r) for r in remaining_ids[:5]}
            raise ValueError(
                f"Could not resolve parentage order for all animals. "
                f"Possible cycle or missing parents for some of the remaining {len(remaining_ids)} animals. "
                f"Example unprocessed (animal: (sire, dam)): {unprocessed_details}"
            )

    return animal_list_final_order, animal_to_idx_map, ped_map_internal


def construct_a_matrix(
    pedigree: list[tuple[int, int | None, int | None]]
) -> tuple[dict[int, int], np.ndarray, list[int], dict[int, tuple[int | None, int | None]]]:
    """
    Constructs the Numerator Relationship Matrix (A) using Henderson's tabular method.

    Args:
        pedigree (list[tuple[int, int | None, int | None]]): Pedigree data where
            each tuple is (animal_id, sire_id, dam_id). Unknown parents are None.
            Animal IDs must be positive integers. Sire/Dam IDs can be positive integers or None.

    Returns:
        tuple[dict[int, int], np.ndarray, list[int], dict[int, tuple[int | None, int | None]]]:
            - animal_to_idx_map: Dictionary mapping animal IDs to matrix index.
            - a_matrix: The Numerator Relationship Matrix (A).
            - animal_list_final_order: Ordered list of animal IDs for matrix rows/columns.
            - ped_map: Parentage map (animal_id -> (sire_id, dam_id))).
    """
    if not isinstance(pedigree, list):
        raise TypeError("Pedigree must be a list of tuples.")
    # Allow empty pedigree to return empty structures
    if not pedigree and not any(pedigree): # check if list contains only empty lists/tuples
        return {}, np.array([]), [], {}


    animal_list_final_order, animal_to_idx_map, ped_map = _get_ordered_animal_list_and_map(pedigree)

    n = len(animal_list_final_order)
    if n == 0: # If pedigree was non-empty but resulted in no processable animals (e.g. only Nones)
        return {}, np.array([]), [], {}

    a_matrix = np.zeros((n, n))

    for i in range(n): # Current animal index
        animal_i_id = animal_list_final_order[i]
        sire_i_id, dam_i_id = ped_map.get(animal_i_id, (None, None))

        s_idx = animal_to_idx_map.get(sire_i_id) if sire_i_id else None
        d_idx = animal_to_idx_map.get(dam_i_id) if dam_i_id else None

        # Calculate F_i (inbreeding coefficient for animal_i_id)
        # F_i = 0.5 * A(sire_i, dam_i) if both parents known
        inbreeding_fi = 0.0
        if s_idx is not None and d_idx is not None:
            # Parents s_idx and d_idx must have smaller indices than i.
            # Access A(s,d) using min/max for upper triangle, as only lower triangle + diag is filled
            # at the point s_idx, d_idx were processed.
            inbreeding_fi = 0.5 * a_matrix[min(s_idx, d_idx), max(s_idx, d_idx)]

        a_matrix[i, i] = 1.0 + inbreeding_fi

        # Off-diagonal elements A(i,j) where j < i
        for j in range(i): # Previous animal index
            # A(i,j) = 0.5 * (A(j, sire_i) + A(j, dam_i))
            # A(j, parent_of_i) means a_matrix[j, parent_idx_of_i]

            val_j_sire_i = a_matrix[j, s_idx] if s_idx is not None else 0.0
            val_j_dam_i = a_matrix[j, d_idx] if d_idx is not None else 0.0

            current_val = 0.5 * (val_j_sire_i + val_j_dam_i)
            a_matrix[i, j] = current_val
            a_matrix[j, i] = current_val # Matrix is symmetric

    return animal_to_idx_map, a_matrix, animal_list_final_order, ped_map


def construct_a_inverse_matrix_from_pedigree(
    pedigree: list[tuple[int, int | None, int | None]]
) -> tuple[dict[int, int], np.ndarray]:
    """
    Placeholder for constructing A-inverse directly from pedigree without pre-forming A.
    This is complex if inbreeding is fully accounted for without iterative F calculations.
    The recommended method for A-inverse with inbreeding uses F values derived from A.
    """
    raise NotImplementedError(
        "Direct A-inverse from pedigree accounting for inbreeding without A matrix "
        "is complex. Use 'construct_a_inverse_matrix' which utilizes a pre-computed A matrix."
    )

def construct_a_inverse_matrix(
    a_matrix: np.ndarray,
    animal_list_final_order: list[int],
    animal_to_idx_map: dict[int, int],
    ped_map: dict[int, tuple[int | None, int | None]]
) -> np.ndarray:
    """
    Constructs the inverse of the Numerator Relationship Matrix (A-inverse)
    using Henderson's rules, accounting for inbreeding. This version assumes the A matrix
    and its derived animal order, index map, and parentage map are available.

    Args:
        a_matrix (np.ndarray): The pre-computed Numerator Relationship Matrix.
        animal_list_final_order (list[int]): Ordered list of animal IDs (parents before offspring).
        animal_to_idx_map (dict[int, int]): Map from animal ID to matrix index.
        ped_map (dict[int, tuple[int | None, int | None]]): Map of animal ID to (sire_id, dam_id).

    Returns:
        np.ndarray: The A-inverse matrix.
    """
    n = len(animal_list_final_order)
    if n == 0:
        return np.array([])

    a_inv = np.zeros((n, n))

    # Extract F_values (inbreeding coefficients) from the diagonal of A
    # A_ii = 1 + F_i  => F_i = A_ii - 1
    f_values = {
        animal_id: a_matrix[animal_to_idx_map[animal_id], animal_to_idx_map[animal_id]] - 1.0
        for animal_id in animal_list_final_order
    }
    # Ensure F values are within reasonable bounds (e.g. not much less than 0 or more than 1)
    for animal_id, f_val in f_values.items():
        if not (-1e-6 <= f_val <= 1.0 + 1e-6): # Allow small tolerance around 0 and 1
             # This could indicate issues in A matrix construction or extreme pedigree
             # For F > 1, it's biologically impossible. F slightly < 0 can be float error for non-inbred.
            pass # Consider logging a warning: f"Warning: Inbreeding for animal {animal_id} is {f_val:.4f}"


    for k_idx, animal_k_id in enumerate(animal_list_final_order):
        sire_k_id, dam_k_id = ped_map.get(animal_k_id, (None, None))

        s_idx = animal_to_idx_map.get(sire_k_id) if sire_k_id else None
        d_idx = animal_to_idx_map.get(dam_k_id) if dam_k_id else None

        f_s = f_values.get(sire_k_id, 0.0) # F_s = 0 if sire is unknown (base animal)
        f_d = f_values.get(dam_k_id, 0.0) # F_d = 0 if dam is unknown

        # Calculate d_i_star = 1 / (variance of Mendelian sampling for animal k)
        d_mendel_var = 0.0
        if s_idx is None and d_idx is None: # Both parents unknown
            d_mendel_var = 1.0
        elif s_idx is not None and d_idx is None: # Sire known, Dam unknown
            # Mendelian variance = 1 * sigma_a^2 - 0.25 * (1+F_s) * sigma_a^2
            # Coefficient relative to sigma_a^2 is (1 - 0.25*(1+F_s)) = (4 - 1 - F_s)/4 = (3-F_s)/4
            d_mendel_var = 0.75 - 0.25 * f_s
        elif s_idx is None and d_idx is not None: # Dam known, Sire unknown
            d_mendel_var = 0.75 - 0.25 * f_d
        else: # Both parents known
            # Mendelian variance = 0.5 * sigma_a^2 * (1 - 0.5*(F_s + F_d))
            d_mendel_var = 0.5 * (1.0 - 0.5 * (f_s + f_d))

        d_i_star = 0.0
        if abs(d_mendel_var) < 1e-9: # Avoid division by zero
            # This case implies issues: F_s=3 or F_d=3 for one-parent (impossible as F<=1),
            # or F_s=1 and F_d=1 for two-parent case.
            # This indicates an animal whose Mendelian sampling variance is zero,
            # e.g., offspring of two identical, fully inbred parents.
            # A-inverse would be undefined or matrix singular.
            # For practical software, this might involve setting d_i_star to a very large number
            # or specific handling for such extreme cases.
            # Raising an error is safer for now.
            raise ValueError(
                f"Mendelian sampling variance is zero or near-zero for animal {animal_k_id} (idx {k_idx}). "
                f"Fs={f_s:.4f}, Fd={f_d:.4f}, d_mendel_var={d_mendel_var:.4e}. "
                "This may indicate extreme inbreeding (e.g. parents are identical and fully inbred) "
                "or errors in the A matrix/pedigree."
            )
        else:
            d_i_star = 1.0 / d_mendel_var

        # Add contributions based on d_i_star
        a_inv[k_idx, k_idx] += d_i_star

        if s_idx is not None and d_idx is None: # Sire known, Dam unknown
            a_inv[s_idx, k_idx] -= 0.5 * d_i_star # Symmetric part below
            a_inv[s_idx, s_idx] += 0.25 * d_i_star
        elif s_idx is None and d_idx is not None: # Dam known, Sire unknown
            a_inv[d_idx, k_idx] -= 0.5 * d_i_star # Symmetric part below
            a_inv[d_idx, d_idx] += 0.25 * d_i_star
        elif s_idx is not None and d_idx is not None: # Both parents known
            a_inv[s_idx, k_idx] -= 0.5 * d_i_star
            a_inv[d_idx, k_idx] -= 0.5 * d_i_star

            a_inv[s_idx, s_idx] += 0.25 * d_i_star
            if s_idx == d_idx: # Parents are the same individual
                # The previous line already added 0.25 for s_idx.
                # The d_idx is same, so it would add to same element.
                # The cross term A_inv[s,d] also becomes A_inv[s,s].
                # Total for s_idx (being also d_idx):
                # a_inv[s,s] gets +0.25 (as s) +0.25 (as d) +0.25 (as s,d) +0.25 (as d,s) = +1.0 * d_i_star
                # The current structure:
                # a_inv[s_idx,s_idx] += 0.25 * d_i_star (from sire part)
                # a_inv[d_idx,d_idx] would be the same element, gets another +0.25*d_i_star
                # Then the cross-parent term:
                a_inv[d_idx, d_idx] += 0.25 * d_i_star # This is correct if s_idx != d_idx
                                                      # If s_idx == d_idx, this is the second 0.25 to same element

                # Parent-parent term
                a_inv[s_idx, d_idx] += 0.25 * d_i_star # if s_idx==d_idx, this is 3rd 0.25 to s_idx,s_idx
                                                       # if s_idx!=d_idx, this is cross term
                if s_idx != d_idx: # Ensure symmetric part for cross term if parents different
                    a_inv[d_idx, s_idx] += 0.25 * d_i_star
            else: # Parents s and d are different individuals
                a_inv[d_idx, d_idx] += 0.25 * d_i_star
                # Parent-parent cross terms
                a_inv[s_idx, d_idx] += 0.25 * d_i_star
                a_inv[d_idx, s_idx] += 0.25 * d_i_star

    # Symmetrize off-diagonal elements that were only added one way for s_idx/d_idx and k_idx
    for k_idx in range(n):
        sire_k_id, dam_k_id = ped_map.get(animal_list_final_order[k_idx], (None, None))
        s_idx = animal_to_idx_map.get(sire_k_id) if sire_k_id else None
        d_idx = animal_to_idx_map.get(dam_k_id) if dam_k_id else None

        if s_idx is not None and d_idx is None:
            if k_idx != s_idx : a_inv[k_idx, s_idx] = a_inv[s_idx, k_idx]
        elif s_idx is None and d_idx is not None:
            if k_idx != d_idx : a_inv[k_idx, d_idx] = a_inv[d_idx, k_idx]
        elif s_idx is not None and d_idx is not None:
            if k_idx != s_idx : a_inv[k_idx, s_idx] = a_inv[s_idx, k_idx]
            if k_idx != d_idx : a_inv[k_idx, d_idx] = a_inv[d_idx, k_idx]
            # s_idx, d_idx already symmetric if s_idx != d_idx

    return a_inv
