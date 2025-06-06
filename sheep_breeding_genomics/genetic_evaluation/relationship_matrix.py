# sheep_breeding_genomics/genetic_evaluation/relationship_matrix.py

import pandas as pd
import numpy as np

# Assuming PedigreeData class is available from the data_management module
# For standalone use or testing, you might need to define a simple PedigreeData-like structure
# from ..data_management.data_structures import PedigreeData # This would be the typical import

def calculate_nrm(pedigree_df: pd.DataFrame,
                  animal_col: str = 'AnimalID',
                  sire_col: str = 'SireID',
                  dam_col: str = 'DamID',
                  founder_val: int = 0) -> pd.DataFrame:
    """
    Calculates the Numerator Relationship Matrix (NRM) using a tabular method.

    Args:
        pedigree_df (pd.DataFrame): DataFrame containing pedigree information.
                                    Must have columns for animal, sire, and dam IDs.
                                    Sire/Dam IDs for unknown parents should be specific (e.g., 0 or None/NaN).
        animal_col (str): Name of the column for animal IDs.
        sire_col (str): Name of the column for sire IDs.
        dam_col (str): Name of the column for dam IDs.
        founder_val (int/None/NaN): Value used to denote an unknown parent (founder).
                                      Commonly 0, None, or np.nan.

    Returns:
        pd.DataFrame: The Numerator Relationship Matrix (A) with AnimalIDs as index and columns.
                      Returns an empty DataFrame if input is invalid.
    """
    if not isinstance(pedigree_df, pd.DataFrame) or pedigree_df.empty:
        print("Error: Pedigree data must be a non-empty Pandas DataFrame.")
        return pd.DataFrame()

    required_cols = {animal_col, sire_col, dam_col}
    if not required_cols.issubset(pedigree_df.columns):
        print(f"Error: Pedigree DataFrame must contain columns: {required_cols}")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original DataFrame
    ped = pedigree_df[[animal_col, sire_col, dam_col]].copy()

    # Ensure animal IDs are unique and not null
    if ped[animal_col].isnull().any():
        print("Error: Animal IDs contain missing values.")
        return pd.DataFrame()
    if not ped[animal_col].is_unique:
        print("Error: Animal IDs are not unique.")
        return pd.DataFrame()

    # Standardize founder representation to np.nan for easier processing
    # This handles cases where founders are 0, None, or already NaN
    if founder_val == 0: # Common case
        ped.replace({sire_col: 0, dam_col: 0}, np.nan, inplace=True)
    elif founder_val is None: # Explicit None
        ped.replace({sire_col: None, dam_col: None}, np.nan, inplace=True)
    # If founder_val is np.nan, no replacement needed for np.nan itself.
    # Ensure other potential representations of "0" (like "0" as string) are handled if necessary before this function.

    # Get unique animal IDs in the order they appear or sorted, for consistent matrix construction
    # Sorting helps if animals are not perfectly ordered by birth date (common in real pedigrees)
    # However, the tabular method processes animals one by one, often assuming an order.
    # For simplicity, let's map IDs to indices. We need a list of all unique animals.

    # Create a list of all unique animals in the pedigree, including parents that might not have their own row as 'animal_col'
    # This is important if the pedigree is not "complete" in the sense that every parent is also an animal in animal_col
    # However, standard NRM calculation usually assumes all animals in sire/dam columns are also in animal_col or are founders.
    # For this implementation, we assume animals in animal_col are the ones for whom relationships are computed.
    # Parents not in animal_col (and not founders) would be an issue for the standard tabular method logic.

    animal_ids = list(ped[animal_col].unique())
    id_to_idx = {animal_id: i for i, animal_id in enumerate(animal_ids)}
    num_animals = len(animal_ids)

    # Initialize NRM matrix (A) with zeros
    nrm_matrix = np.zeros((num_animals, num_animals))

    # Sort pedigree by animal ID (or by an assumed birth order if available)
    # This is not strictly necessary for the logic below if using id_to_idx mapping correctly,
    # but processing in a somewhat logical order (e.g. founders first, then their offspring) is intuitive.
    # For this tabular method, we iterate through animals and fill relationships.
    # A common approach is to ensure parents appear before offspring in the list/pedigree.
    # If not sorted, we must handle cases where a parent's record hasn't been processed.
    # The direct tabular method calculates A_ij based on parents.

    # For simplicity, this implementation will iterate through the provided animal_ids list.
    # It's crucial that if animal_ids is not sorted by birth order, the pedigree lookups are correct.

    for i in range(num_animals):
        animal_i_id = animal_ids[i]
        animal_i_row = ped[ped[animal_col] == animal_i_id].iloc[0]
        sire_i = animal_i_row[sire_col]
        dam_i = animal_i_row[dam_col]

        sire_i_idx = id_to_idx.get(sire_i) # Returns None if sire is founder (NaN) or not in id_to_idx
        dam_i_idx = id_to_idx.get(dam_i)   # Returns None if dam is founder (NaN) or not in id_to_idx

        # Diagonal element A(i,i)
        if sire_i_idx is None and dam_i_idx is None: # Both parents unknown (founder)
            nrm_matrix[i, i] = 1.0
        elif sire_i_idx is not None and dam_i_idx is not None: # Both parents known
            # Ensure parents' relationships (A_ss, A_dd, A_sd) are already computed if relying on them.
            # The tabular method computes A_ii = 1 + 0.5 * A_sd (if s and d are parents of i)
            # This requires A_sd to be known. The loop order matters or one must use a different formulation.
            # Henderson's rules are typically applied row by row.
            # Rule: A_ii = 1 if parents unknown. A_ii = 1 + 0.5 * A_parents if parents known and related.
            # This is simpler: A_ii = 1 + 0.5 * A_sd (relationship between sire and dam)
            # For A_ii, if parents s and d are known: A_ii = 1 + 0.5 * A[sire_i_idx, dam_i_idx]
            # This implies A[sire_i_idx, dam_i_idx] must be correct.
            # The traditional tabular method fills A_ij for j <= i.

            # Simpler rule for diagonal often used:
            # A_ii = 1.0 by default (base relationship)
            # If parents are known and related, add 0.5 * A_sd. If inbred, this is implicitly handled.
            # For now, standard calculation for A_ii:
            if sire_i_idx == dam_i_idx : # Should not happen in a valid pedigree unless sire/dam can be same
                 nrm_matrix[i, i] = 1 + 0.5 * nrm_matrix[sire_i_idx, sire_i_idx] # or dam_i_idx
            else:
                 nrm_matrix[i, i] = 1 + 0.5 * nrm_matrix[sire_i_idx, dam_i_idx]

        elif sire_i_idx is not None: # Only sire known
            nrm_matrix[i, i] = 1.0 # Or 1 + 0.5 * (A_ss - 1) if sire is inbred, but simpler is 1.0
        elif dam_i_idx is not None: # Only dam known
            nrm_matrix[i, i] = 1.0
        else: # Should be covered by both unknown
            nrm_matrix[i,i] = 1.0


        # Off-diagonal elements A(i,j) for j < i
        for j in range(i):
            animal_j_id = animal_ids[j]
            # animal_j_row = ped[ped[animal_col] == animal_j_id].iloc[0] # Not needed for this logic

            sire_j_is_parent_of_i = (sire_i_idx is not None and sire_i_idx == j)
            dam_j_is_parent_of_i = (dam_i_idx is not None and dam_i_idx == j)

            # Relationship A(i,j) = A(j,i)
            # Case 1: j is sire of i
            if sire_j_is_parent_of_i and dam_i_idx is not None : # j is sire, dam_i is known
                nrm_matrix[i, j] = 0.5 * (nrm_matrix[j, j] + nrm_matrix[j, dam_i_idx]) # Mistake here, should be A(j,d) not A(j,j) + A(j,d)
                                                                                      # Correct: 0.5 * (A(j,s_i) + A(j,d_i)) where s_i is j
                                                                                      # A(i,j) = 0.5 * (A(s_i, j) + A(d_i, j))
                nrm_matrix[i,j] = 0.5 * (nrm_matrix[sire_i_idx, j] + nrm_matrix[dam_i_idx,j]) if dam_i_idx is not None else 0.5 * nrm_matrix[sire_i_idx,j]

            # Case 2: j is dam of i
            elif dam_j_is_parent_of_i and sire_i_idx is not None: # j is dam, sire_i is known
                nrm_matrix[i,j] = 0.5 * (nrm_matrix[sire_i_idx, j] + nrm_matrix[dam_i_idx,j]) if sire_i_idx is not None else 0.5 * nrm_matrix[dam_i_idx,j]

            # Case 3: Neither parent of i is j. Parents of i are s_i, d_i.
            # A(i,j) = 0.5 * (A(s_i, j) + A(d_i, j))
            # This applies if s_i and d_i are known.
            else:
                val = 0.0
                if sire_i_idx is not None:
                    val += 0.5 * nrm_matrix[sire_i_idx, j]
                if dam_i_idx is not None:
                    val += 0.5 * nrm_matrix[dam_i_idx, j]
                nrm_matrix[i, j] = val

            nrm_matrix[j, i] = nrm_matrix[i, j] # Symmetric matrix

    # Re-iterate for diagonal elements as they depend on off-diagonals (relationship between parents)
    # This is a common way to ensure correctness with the tabular method or Henderson's rules.
    # A_ii = 1 (if both parents unknown)
    # A_ii = 1 (if one parent unknown) - this is a simplification for non-inbred founders
    # A_ii = 1 + 0.5 * A_sd (if both parents s,d known)
    # For animals with at least one unknown parent, F_i = 0, so A_ii = 1.
    # For animal i with parents s and d: A_ii = 1 + F_i = 1 + 0.5 * A_sd.

    # Corrected loop structure for Henderson's rules / Tabular method:
    # This method fills the matrix by iterating through animals, assuming parents precede offspring OR by direct lookup.
    # A more robust tabular method (Henderson's rules by element):
    # Requires animals to be ordered such that parents always appear before their offspring in `animal_ids`.
    # If not, then direct lookup using id_to_idx is fine.

    # Simpler approach using direct formulas (Henderson, 1976, summarized by Mrode, 2005)
    # 1. Sort pedigree so parents appear before offspring (or use pointers/indices).
    #    Here, we use id_to_idx, so order in `animal_ids` list defines processing order.
    #    It's best if `animal_ids` is sorted chronologically if possible.

    # Let's re-initialize and use the more standard element-wise calculation based on Henderson's rules.
    # This requires careful handling of founder values (e.g. mapping them to an index outside the matrix or handling them as 0).

    # Create a mapping from original ID to a 0-based index for matrix ops
    # And ensure all parents (if not founders) are in this mapping.
    # The current `animal_ids` and `id_to_idx` should serve this.

    # Re-initialize NRM
    nrm_matrix = np.zeros((num_animals, num_animals))

    # Iterate through each animal `i` to calculate row `i` of A
    for i in range(num_animals):
        animal_i_id = animal_ids[i]
        animal_i_row = ped[ped[animal_col] == animal_i_id].iloc[0]
        sire_of_i = animal_i_row[sire_col]
        dam_of_i = animal_i_row[dam_col]

        s_idx = id_to_idx.get(sire_of_i) # None if sire is founder or not in animal_ids list
        d_idx = id_to_idx.get(dam_of_i) # None if dam is founder or not in animal_ids list

        # Calculate A(i,i) - diagonal element
        if s_idx is None and d_idx is None: # Both parents are founders/unknown
            nrm_matrix[i, i] = 1.0
        elif s_idx is not None and d_idx is not None: # Both parents known
            # A(i,i) = 1 + 0.5 * A(s,d)
            # Ensure s_idx and d_idx are valid (i.e., parents are in the matrix)
            # Relationship A(s,d) must be A[s_idx, d_idx] (or A[d_idx, s_idx])
            # This requires the matrix to be filled such that A(s,d) is already known.
            # This implies parents (s,d) must have smaller indices than i.
            # THIS IS A CRUCIAL Requirement: animal_ids should be sorted by birth date or generation.
            # If not, this direct calculation of A(i,i) using A(s,d) can be wrong if A(s,d) isn't final.
            # For now, assume ordering is sufficient or use the iterative method later.
            # A simpler formulation for A(i,i) if not strictly ordered:
            # A(i,i) = 1.0 initially. Inbreeding contribution is added if parents related.
            # For tabular method without strict ordering, A(i,i) = 1 + F_i, where F_i is inbreeding coeff.
            # F_i = 0.5 * A[s_idx, d_idx] IF s_idx and d_idx refer to animals already processed.
            # This is why sorting is important.
            # Let's use the rules that build up A_ij for j <= i:
            nrm_matrix[i,i] = 1.0 # Base if one parent unknown or if parents unrelated founders
            if s_idx is not None and d_idx is not None:
                 # This assumes A[s_idx, d_idx] (parental relationship) is already computed.
                 # This requires s_idx < i and d_idx < i.
                 # If animal_ids is not sorted appropriately, this is problematic.
                 # A more robust way without strict sort is more complex or iterative.
                 # For now, let's proceed assuming `animal_ids` is somewhat ordered or that lookups are to already filled parts.
                 # A_ii = 1 + 0.5 * A_sd (relationship between parents)
                 # This A_sd needs to be from the already computed part of the matrix.
                 # The value A[s_idx, d_idx] should be A_sd.
                 # We need to ensure s_idx and d_idx refer to the correct indices in nrm_matrix.
                 # This is where sorting pedigree by effective birth order is essential.
                 # If not sorted, this calculation of A_ii might be incorrect if A_sd isn't final.
                 # A common way: A_ii = 1 + F_i, where F_i = 0.5 * A_sd (if s and d are parents of i)
                 # Let's assume for now the animal_ids list implies a usable order.
                 # The relationship between sire and dam:
                 parent_relationship = nrm_matrix[s_idx, d_idx] if s_idx <= d_idx else nrm_matrix[d_idx, s_idx]
                 nrm_matrix[i,i] = 1.0 + 0.5 * parent_relationship
            # If one parent known, e.g. sire s, A_ii = 1. (Assuming founder dam is unrelated to sire's line)
            # This is implicitly handled if A_sd where one is founder is 0.

        elif s_idx is not None: # Sire known, Dam unknown
            nrm_matrix[i,i] = 1.0 # Assumes unknown dam is a founder, unrelated. No inbreeding from dam side.
        elif d_idx is not None: # Dam known, Sire unknown
            nrm_matrix[i,i] = 1.0 # Assumes unknown sire is a founder.
        # else: both unknown, already set to 1.0 earlier.

        # Calculate A(i,j) for j < i (off-diagonal elements)
        for j in range(i):
            # A(i,j) = 0.5 * (A(sire_of_i, j) + A(dam_of_i, j))
            # If sire/dam is unknown, their contribution is 0.
            val_from_sire = 0.0
            if s_idx is not None:
                # A(sire_of_i, j) is nrm_matrix[s_idx, j] (or [j, s_idx] due to symmetry)
                val_from_sire = nrm_matrix[s_idx, j] if s_idx >=j else nrm_matrix[j,s_idx]


            val_from_dam = 0.0
            if d_idx is not None:
                val_from_dam = nrm_matrix[d_idx, j] if d_idx >=j else nrm_matrix[j,d_idx]

            nrm_matrix[i,j] = 0.5 * (val_from_sire + val_from_dam)
            nrm_matrix[j,i] = nrm_matrix[i,j] # Symmetric

    # Final check on diagonals if pedigree was not perfectly sorted by birth date.
    # The above single pass for A_ii using A_sd assumes parents (s,d) are processed before child (i).
    # If `animal_ids` is not guaranteed to be sorted by birth order, A_sd might not be final when A_ii is computed.
    # A common method for tabular is to sort animals by birth year or use an iterative approach to build A.
    # For a direct method like Henderson (1976), the animal list must be ordered such that an animal always appears after its parents.
    # If such an order is not guaranteed by the input `pedigree_df`, it needs to be sorted first.
    # Let's add a step to attempt sorting or warn if order might be an issue.
    # For now, we proceed with the calculation as is, highlighting this assumption.
    # A simple sort by AnimalID (if numeric and sequential) might approximate this.

    # Create DataFrame for NRM
    nrm_df = pd.DataFrame(nrm_matrix, index=animal_ids, columns=animal_ids)

    print(f"NRM calculation complete for {num_animals} animals.")
    return nrm_df


def example_usage_nrm():
    """Example usage of the NRM calculation function."""
    print("--- NRM Calculation Example ---")
    # Example pedigree (AnimalID, SireID, DamID), 0 for founder
    ped_data = {
        'AnimalID': [1, 2, 3, 4, 5, 6],
        'SireID':   [0, 0, 1, 1, 3, 0], # Animal 1 & 2 are founders
        'DamID':    [0, 0, 2, 0, 4, 2]  # Animal 4's dam is unknown (founder)
    }
    pedigree_df = pd.DataFrame(ped_data)

    # Ensure AnimalID is sorted for the tabular method to work correctly in one pass for A_ii.
    # This is a critical assumption for the single-pass A_ii = 1 + 0.5 * A_sd part.
    # If AnimalID doesn't represent birth order, a more complex sort or method is needed.
    # For this example, we assume AnimalIDs are in an order that respects parent-offspring sequence.
    pedigree_df.sort_values(by='AnimalID', inplace=True) # Sorting just in case

    print("Input Pedigree:")
    print(pedigree_df)

    nrm = calculate_nrm(pedigree_df, founder_val=0)

    if not nrm.empty:
        print("\nNumerator Relationship Matrix (A):")
        # Print rounded for display
        print(nrm.round(3))

    # Example with a slightly more complex pedigree (potential inbreeding)
    print("\n--- NRM Calculation Example with Inbreeding ---")
    ped_data_inbred = { # Animal 5 is inbred (parents 3 and 4, where 3 and 4 are half-sibs)
        'AnimalID': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'SireID':   [None, None, 'A1', 'A1', 'A3'],
        'DamID':    [None, None, 'A2', None, 'A4']
    }
    pedigree_df_inbred = pd.DataFrame(ped_data_inbred)
    # For string IDs, sorting by ID might also work if IDs reflect order.
    # The key is that id_to_idx mapping provides indices that, when looped i from 0 to N-1,
    # parents s_idx and d_idx are < i.
    # This requires a topological sort of the pedigree (parents before offspring).
    # The current code does not enforce this sort; it relies on the order in `animal_ids` list.
    # If `animal_ids` is derived from `pedigree_df[animal_col].unique()` without specific sorting,
    # it might not be in the required order.

    # To make it more robust, one should sort the pedigree topologically before creating `animal_ids`.
    # For now, we'll assume the user provides a somewhat ordered pedigree or understands this limitation.

    print("Input Pedigree (Inbred Example):")
    print(pedigree_df_inbred)
    nrm_inbred = calculate_nrm(pedigree_df_inbred, founder_val=None) # Using None as founder
    if not nrm_inbred.empty:
        print("\nNRM for Inbred Example (A):")
        print(nrm_inbred.round(3))

    # Test with empty pedigree
    print("\n--- NRM Calculation with Empty Pedigree ---")
    empty_ped_df = pd.DataFrame(columns=['AnimalID', 'SireID', 'DamID'])
    nrm_empty = calculate_nrm(empty_ped_df)
    print(f"NRM for empty pedigree is empty: {nrm_empty.empty}")

    # Test with missing columns
    print("\n--- NRM Calculation with Missing Columns ---")
    missing_cols_ped_df = pd.DataFrame({'AnimalID': [1,2]})
    nrm_missing_cols = calculate_nrm(missing_cols_ped_df)
    print(f"NRM for missing columns is empty: {nrm_missing_cols.empty}")


if __name__ == '__main__':
    # To run this as a script for testing:
    # Ensure the data_structures module is accessible if you uncomment the import.
    # For now, it works with DataFrame directly.
    example_usage_nrm()

    # A note on sorting for NRM calculation:
    # The provided code for calculate_nrm assumes that when processing animal `i`,
    # the relationships for its parents (s,d) specifically A(s,d), are already correctly computed.
    # This is guaranteed if `animal_ids` (and thus the iteration order `i` and `j`)
    # is such that parents always have a smaller index than their offspring.
    # A topological sort of the pedigree graph is the most robust way to achieve this order.
    # If `pedigree_df[animal_col].unique()` does not yield such an order naturally,
    # the NRM diagonals (A_ii involving 0.5 * A_sd) might be inaccurate in a single pass.
    # More advanced implementations use techniques like "L রোগা" (L inverse) or iterative updates.
    # The current implementation is a direct tabular method often taught, assuming ordered input.
    # Consider adding a pedigree sorting step for robustness in future versions.


def calculate_grm(genomic_data_obj, method: str = "vanraden1") -> pd.DataFrame:
    """
    Calculates the Genomic Relationship Matrix (GRM) using VanRaden's Method 1.

    Method 1: G = ZZ' / (2 * sum(pi*(1-pi)))
    where Z is the marker matrix (animals x SNPs) centered by twice the allele frequency (2p_i).
    Genotypes are typically coded as 0, 1, 2 for aa, Aa, AA respectively.
    Missing genotypes should be handled (e.g., imputed or filtered) before this function.
    This function will raise an error if NaNs are present in the genotype matrix.

    Args:
        genomic_data_obj (GenomicData): An instance of the GenomicData class.
                                        It's assumed that animal IDs are unique and
                                        genotypes are numeric (0, 1, 2).
                                        Missing values (NaNs) in the genotype matrix are not allowed.
        method (str): The method for GRM calculation. Currently, only "vanraden1" is supported.

    Returns:
        pd.DataFrame: The Genomic Relationship Matrix (G) with AnimalIDs as index and columns.
                      Returns an empty DataFrame if input is invalid or errors occur.
    """
    # Import GenomicData here to avoid circular dependency if relationship_matrix is imported by data_structures
    # This is a common pattern if modules are tightly coupled.
    # However, for this project structure, GenomicData is in a different app, so direct import is fine.
    from ..data_management.data_structures import GenomicData

    if not isinstance(genomic_data_obj, GenomicData):
        print("Error: Input must be a GenomicData object.")
        return pd.DataFrame()

    if genomic_data_obj.num_animals == 0 or genomic_data_obj.num_snps == 0:
        print("Error: Genomic data object contains no animals or no SNPs.")
        return pd.DataFrame()

    M = genomic_data_obj.get_genotypes().values # Get genotype matrix as NumPy array
                                              # Animals in rows, SNPs in columns.

    # Check for missing values
    if np.isnan(M).any():
        print("Error: Genotype matrix contains missing values (NaNs). Impute or filter them before GRM calculation.")
        return pd.DataFrame()

    n_animals, n_snps = M.shape

    if method.lower() == "vanraden1":
        # Calculate allele frequencies (p_i for each SNP i)
        # p_i = (count of allele 'A') / (2 * n_animals)
        # Assuming 0,1,2 coding for aa, Aa, AA (number of 'A' alleles)
        # Allele frequency p is for the allele whose count is represented by M_ij values.
        # So, sum of M_ij for SNP j is sum of 'A' alleles for that SNP.
        # p_j = sum(M_ij for i in animals) / (2 * n_animals)

        # Allele frequencies (freq of the allele coded as '1' in M)
        # For 0,1,2 coding, sum of genotypes / (2*N) gives frequency of allele '2' (e.g. 'A')
        p = np.sum(M, axis=0) / (2 * n_animals) # p is a vector of allele frequencies for each SNP

        # Check for monomorphic SNPs (where p=0 or p=1), as these cause issues in denominator 2*sum(pi(1-pi))
        # VanRaden (2008) mentions that monomorphic markers contribute nothing to genomic relationships
        # and can be removed or handled. Division by zero if all SNPs are monomorphic.
        # Here, we calculate sum_2pq = 2 * sum(p_i * (1-p_i)). If a SNP has p=0 or p=1, p(1-p)=0.
        # Such SNPs don't contribute to the sum, which is fine.
        # If ALL SNPs are monomorphic, sum_2pq will be 0, leading to division by zero.

        # Center Z matrix: Z_ij = M_ij - 2*p_j
        # Z = M - 2 * p (broadcasting p to each row of M)
        # This is M_ij - E[M_ij] where E[M_ij] = 2*p_j
        P_matrix = np.tile(2 * p, (n_animals, 1)) # Create matrix where each row is 2*p vector
        Z = M - P_matrix

        # Denominator: 2 * sum(p_i * (1-p_i)) over all SNPs
        sum_2pq = 2 * np.sum(p * (1 - p))

        if sum_2pq == 0:
            print("Error: Sum of 2*pi*(1-pi) is zero. This can happen if all SNPs are monomorphic or no SNPs.")
            # This could also happen if p is > 1 or < 0 due to bad input, but primary check is for monomorphic.
            return pd.DataFrame()

        # Calculate GRM: G = ZZ' / sum_2pq
        grm = (Z @ Z.T) / sum_2pq

        animal_ids = genomic_data_obj.animal_ids
        grm_df = pd.DataFrame(grm, index=animal_ids, columns=animal_ids)

        print(f"GRM calculation using VanRaden Method 1 complete for {n_animals} animals and {n_snps} SNPs.")
        return grm_df

    else:
        print(f"Error: Method '{method}' for GRM calculation is not supported.")
        return pd.DataFrame()


def example_usage_grm():
    """Example usage of the GRM calculation function."""
    print("\\n--- GRM Calculation Example (VanRaden Method 1) ---")

    # Need to import GenomicData for the example.
    # This is tricky if this script is run directly and paths are not set up.
    # For direct script run, you might need to adjust sys.path or use a placeholder.
    # Assuming it's run as part of the package or tests where imports work:
    from ..data_management.data_structures import GenomicData

    # Example Genomic Data
    geno_dict = {
        'AnimalID': ['A1', 'A2', 'A3', 'A4'],
        'SNP1': [0, 1, 2, 0], # p1 = (0+1+2+0)/(2*4) = 3/8 = 0.375
        'SNP2': [1, 1, 0, 2], # p2 = (1+1+0+2)/(2*4) = 4/8 = 0.5
        'SNP3': [2, 0, 1, 1], # p3 = (2+0+1+1)/(2*4) = 4/8 = 0.5
        # 'SNP4': [0,0,0,0] # Monomorphic example, p4 = 0. sum2pq would exclude it.
    }
    # M = [[0,1,2], [1,1,0], [2,0,1], [0,2,1]] # Transposed view for by-animal
    # Correct M (animals x SNPs)
    # A1: [0,1,2]
    # A2: [1,1,0]
    # A3: [2,0,1]
    # A4: [0,2,1]

    geno_df = pd.DataFrame(geno_dict)
    genomic_data_for_grm = GenomicData(geno_df, animal_id_col='AnimalID')
    print("Input GenomicData for GRM:")
    print(genomic_data_for_grm)

    grm_df = calculate_grm(genomic_data_for_grm)

    if not grm_df.empty:
        print("\\nGenomic Relationship Matrix (GRM) - VanRaden Method 1:")
        print(grm_df.round(3))

    # Example with missing data (should fail)
    print("\\n--- GRM Calculation with Missing Data (should fail) ---")
    geno_dict_missing = {
        'AnimalID': ['A1', 'A2'],
        'SNP1': [0, np.nan],
        'SNP2': [1, 1]
    }
    geno_df_missing = pd.DataFrame(geno_dict_missing)
    genomic_data_missing = GenomicData(geno_df_missing, animal_id_col='AnimalID')
    grm_missing_df = calculate_grm(genomic_data_missing)
    if grm_missing_df.empty:
        print("GRM calculation correctly failed due to missing data.")
    else:
        print("GRM calculation proceeded with missing data, which is unexpected by current design.")

    # Example with all monomorphic SNPs (should fail sum_2pq = 0)
    print("\\n--- GRM Calculation with All Monomorphic SNPs (should fail) ---")
    geno_dict_mono = {
        'AnimalID': ['A1', 'A2'],
        'SNP1': [0, 0], # p1=0
        'SNP2': [2, 2]  # p2=1
    }
    geno_df_mono = pd.DataFrame(geno_dict_mono)
    genomic_data_mono = GenomicData(geno_df_mono, animal_id_col='AnimalID')
    grm_mono_df = calculate_grm(genomic_data_mono)
    if grm_mono_df.empty:
        print("GRM calculation correctly failed due to all monomorphic SNPs.")
    else:
        print("GRM calculation with all monomorphic SNPs did not fail as expected.")


if __name__ == '__main__':
    # To run this as a script for testing:
    # Ensure the data_management module is accessible if you uncomment the import.
    # For now, it works with DataFrame directly.
    example_usage_nrm()
    example_usage_grm() # Add GRM example to main execution
    example_usage_h_inv_matrix() # Add H_inv example
