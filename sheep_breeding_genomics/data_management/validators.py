# sheep_breeding_genomics/data_management/validators.py

import pandas as pd
import numpy as np
from .data_structures import PhenotypicData, PedigreeData, GenomicData

def validate_phenotypic_data(phen_data: PhenotypicData, required_columns: list = None, expected_dtypes: dict = None) -> bool:
    """
    Validates phenotypic data.
    - Checks if the data is an instance of PhenotypicData.
    - Checks for the presence of required columns (if specified).
    - Checks for expected data types of columns (if specified).
    - Checks for missing values in key columns (e.g., AnimalID).

    Args:
        phen_data (PhenotypicData): The phenotypic data object to validate.
        required_columns (list, optional): A list of column names that must be present.
        expected_dtypes (dict, optional): A dictionary where keys are column names
                                          and values are their expected data types (e.g., {'Age': int}).

    Returns:
        bool: True if data is valid, False otherwise.
    """
    if not isinstance(phen_data, PhenotypicData):
        print("Error: Input is not a valid PhenotypicData object.")
        return False

    df = phen_data.data
    if df.empty:
        print("Warning: Phenotypic data is empty. Validation may not be meaningful.")
        # Depending on requirements, an empty DataFrame might be considered valid or invalid.
        # For now, let's consider it "valid" but with a warning.
        return True

    # Check for AnimalID presence and uniqueness (assuming 'AnimalID' is a standard key)
    if 'AnimalID' not in df.columns:
        print("Error: 'AnimalID' column is missing in phenotypic data.")
        return False
    if df['AnimalID'].isnull().any():
        print("Error: 'AnimalID' column contains missing values in phenotypic data.")
        return False
    if not df['AnimalID'].is_unique:
        print("Warning: 'AnimalID' column contains duplicate values in phenotypic data.")
        # This might be a warning or an error depending on specific use cases.

    # Check for required columns
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' is missing in phenotypic data.")
                return False

    # Check for expected data types
    if expected_dtypes:
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                    try:
                        # Attempt to cast, could be useful for flexibility e.g. int64 vs int32
                        df[col] = df[col].astype(dtype)
                        phen_data.data = df # Update the object's data
                        print(f"Info: Column '{col}' was cast to {dtype}.")
                    except Exception as e:
                        print(f"Error: Column '{col}' in phenotypic data is not of expected type {dtype} (actual: {df[col].dtype}). Attempt to cast failed: {e}")
                        return False
            else:
                print(f"Warning: Column '{col}' for dtype check not found in phenotypic data.")

    print("Phenotypic data validation successful (or passed with warnings).")
    return True


def validate_pedigree_data(ped_data: PedigreeData, check_loops: bool = False) -> bool:
    """
    Validates pedigree data.
    - Checks if the data is an instance of PedigreeData.
    - Checks for essential columns: AnimalID, SireID, DamID.
    - Checks for missing AnimalIDs.
    - Checks that AnimalIDs are unique.
    - Checks that SireID and DamID exist as AnimalIDs in the pedigree (referential integrity), excluding founders (e.g., 0 or None).
    - Optional: Check for simple loops (e.g., animal being its own sire/dam).

    Args:
        ped_data (PedigreeData): The pedigree data object to validate.
        check_loops (bool): If True, perform basic loop checks.

    Returns:
        bool: True if data is valid, False otherwise.
    """
    if not isinstance(ped_data, PedigreeData):
        print("Error: Input is not a valid PedigreeData object.")
        return False

    df = ped_data.data
    if df.empty:
        print("Warning: Pedigree data is empty. Validation may not be meaningful.")
        return True # Similar to phenotypic data, consider empty as "valid" with warning.

    # Essential columns check already done in PedigreeData constructor, but good to re-verify
    required_cols = {'AnimalID', 'SireID', 'DamID'}
    if not required_cols.issubset(df.columns):
        print(f"Error: Pedigree DataFrame must contain columns: {required_cols}")
        return False

    # Check for missing or non-unique AnimalID
    if df['AnimalID'].isnull().any():
        print("Error: 'AnimalID' column contains missing values in pedigree data.")
        return False
    if not df['AnimalID'].is_unique:
        print("Error: 'AnimalID' column in pedigree data must contain unique values.")
        return False

    # Convert IDs to a common type for comparison, handling potential mixed types from CSV.
    # This is a common preprocessing step. Let's assume IDs can be numbers or strings representing numbers.
    # For simplicity, we'll work with them as they are but ensure comparisons are consistent.
    # A more robust solution might involve casting to a specific type (e.g., str or int) after assessing data.

    all_animal_ids = set(df['AnimalID'])

    # Check SireID and DamID referential integrity
    for index, row in df.iterrows():
        sire_id = row['SireID']
        dam_id = row['DamID']

        # Check if SireID is valid (exists in AnimalID column or is a founder indicator like 0, None, or NaN)
        if pd.notna(sire_id) and sire_id != 0 and sire_id not in all_animal_ids:
            print(f"Error: SireID '{sire_id}' for AnimalID '{row['AnimalID']}' does not exist as an AnimalID in the pedigree.")
            return False

        # Check if DamID is valid
        if pd.notna(dam_id) and dam_id != 0 and dam_id not in all_animal_ids:
            print(f"Error: DamID '{dam_id}' for AnimalID '{row['AnimalID']}' does not exist as an AnimalID in the pedigree.")
            return False

        # Basic loop checks (optional)
        if check_loops:
            if pd.notna(sire_id) and row['AnimalID'] == sire_id:
                print(f"Error: AnimalID '{row['AnimalID']}' is its own sire.")
                return False
            if pd.notna(dam_id) and row['AnimalID'] == dam_id:
                print(f"Error: AnimalID '{row['AnimalID']}' is its own dam.")
                return False
            if pd.notna(sire_id) and pd.notna(dam_id) and sire_id == dam_id and sire_id != 0: # Ensure sire and dam are not the same (unless founder)
                 print(f"Warning: AnimalID '{row['AnimalID']}' has identical SireID and DamID: '{sire_id}'.")


    print("Pedigree data validation successful.")
    return True


if __name__ == '__main__':
    # --- Test Phenotypic Data Validation ---
    print("--- Testing Phenotypic Data Validation ---")
    valid_phen_df = pd.DataFrame({
        'AnimalID': [1, 2, 3, 4],
        'TraitA': [10.1, 10.2, 10.3, 10.4],
        'Age': [24, 36, 20, 30],
        'Category': ['A', 'B', 'A', 'C']
    })
    phen_obj = PhenotypicData(valid_phen_df)
    print("\\nValidating good phenotypic data:")
    validate_phenotypic_data(phen_obj,
                             required_columns=['AnimalID', 'TraitA', 'Age'],
                             expected_dtypes={'Age': int, 'TraitA': float})

    missing_col_phen_df = pd.DataFrame({'AnimalID': [1, 2], 'TraitA': [10.1, 10.2]})
    phen_obj_missing_col = PhenotypicData(missing_col_phen_df)
    print("\\nValidating phenotypic data with missing required column 'Age':")
    validate_phenotypic_data(phen_obj_missing_col, required_columns=['Age'])

    wrong_dtype_phen_df = pd.DataFrame({'AnimalID': [1, 2], 'Age': ['24', '36']}) # Age as string
    phen_obj_wrong_dtype = PhenotypicData(wrong_dtype_phen_df)
    print("\\nValidating phenotypic data with wrong dtype for 'Age' (should be int, will attempt cast):")
    validate_phenotypic_data(phen_obj_wrong_dtype, expected_dtypes={'Age': int})
    print(f"Data type of 'Age' after validation attempt: {phen_obj_wrong_dtype.data['Age'].dtype}")


    duplicate_id_phen_df = pd.DataFrame({'AnimalID': [1, 1, 2], 'TraitA': [10,11,12]})
    phen_obj_dup_id = PhenotypicData(duplicate_id_phen_df)
    print("\\nValidating phenotypic data with duplicate AnimalIDs (warning):")
    validate_phenotypic_data(phen_obj_dup_id)

    # --- Test Pedigree Data Validation ---
    print("\\n--- Testing Pedigree Data Validation ---")
    valid_ped_df = pd.DataFrame({
        'AnimalID': [1, 2, 3, 4, 5],
        'SireID': [0, 1, 1, 0, 3], # 0 or None for founders
        'DamID': [0, 0, 2, 2, 4]
    })
    ped_obj = PedigreeData(valid_ped_df)
    print("\\nValidating good pedigree data:")
    validate_pedigree_data(ped_obj, check_loops=True)

    invalid_sire_ped_df = pd.DataFrame({
        'AnimalID': [1, 2, 3],
        'SireID': [0, 1, 99], # Sire 99 does not exist
        'DamID': [0, 0, 2]
    })
    ped_obj_invalid_sire = PedigreeData(invalid_sire_ped_df)
    print("\\nValidating pedigree data with non-existent SireID:")
    validate_pedigree_data(ped_obj_invalid_sire)

    loop_ped_df = pd.DataFrame({
        'AnimalID': [1, 2, 3],
        'SireID': [0, 1, 2],
        'DamID': [0, 2, 2] # Animal 2 is its own Dam
    })
    ped_obj_loop = PedigreeData(loop_ped_df) # This structure is valid for PedigreeData init as cols are fine
    print("\\nValidating pedigree data with an animal being its own dam (loop):")
    validate_pedigree_data(ped_obj_loop, check_loops=True) # Now check_loops should catch it

    ped_obj_loop_sire_dam_same = PedigreeData(pd.DataFrame({
        'AnimalID': [1, 2, 3],
        'SireID': [0, 1, 2],
        'DamID': [0, 1, 2] # Animal 2 has Sire 1, Dam 1; Animal 3 has Sire 2, Dam 2
    }))
    print("\\nValidating pedigree data where sire and dam are the same (warning):")
    validate_pedigree_data(ped_obj_loop_sire_dam_same, check_loops=True)


    missing_animalid_ped_df = pd.DataFrame({ # AnimalID is missing values
        'AnimalID': [1, None, 3],
        'SireID': [0,1,1],
        'DamID': [0,0,2]
    })
    ped_obj_missing_id = PedigreeData(missing_animalid_ped_df)
    print("\\nValidating pedigree data with missing AnimalID:")
    validate_pedigree_data(ped_obj_missing_id)

    duplicate_animalid_ped_df = pd.DataFrame({ # AnimalID is not unique
        'AnimalID': [1, 1, 3],
        'SireID': [0,1,1],
        'DamID': [0,0,2]
    })
    # This would ideally be caught by PedigreeData if IDs must be unique on init,
    # but current PedigreeData doesn't enforce this. validate_pedigree_data does.
    ped_obj_dup_id_ped = PedigreeData(duplicate_animalid_ped_df)
    print("\\nValidating pedigree data with duplicate AnimalID:")
    validate_pedigree_data(ped_obj_dup_id_ped)


def validate_genomic_data(genomic_data: GenomicData, valid_genotypes: list = [0, 1, 2, np.nan]) -> tuple[bool, pd.DataFrame]:
    """
    Validates genomic data.
    - Checks if data is a GenomicData object.
    - Checks for valid genotype coding (e.g., values are in [0, 1, 2, NaN]).
    - Calculates missing data rates per SNP and per animal.

    Args:
        genomic_data (GenomicData): The genomic data object to validate.
        valid_genotypes (list): A list of acceptable genotype values. np.nan is usually included for missing.

    Returns:
        tuple[bool, pd.DataFrame]:
            - bool: True if data passes basic validation checks (type, structure), False otherwise.
            - pd.DataFrame: A DataFrame with SNP-wise statistics (call rate, allele frequencies if calculable here).
                            Returns an empty DataFrame if basic validation fails.
    """
    if not isinstance(genomic_data, GenomicData):
        print("Error: Input is not a valid GenomicData object.")
        return False, pd.DataFrame()

    if genomic_data.data.empty:
        print("Warning: Genomic data is empty. Validation is trivial.")
        return True, pd.DataFrame(columns=['snp', 'call_rate', 'maf']) # Return empty stats df with expected columns

    genotypes_df = genomic_data.get_genotypes() # This excludes AnimalID column
    snp_stats_list = []

    # Check for valid genotype values
    all_genotypes_flat = genotypes_df.values.flatten()
    # Using a set for faster lookups of valid genotypes
    valid_genotype_set = set(valid_genotypes)

    invalid_values_found = []
    for val in np.unique(all_genotypes_flat): # Check unique values present in the data
        if val not in valid_genotype_set:
            invalid_values_found.append(val)

    if invalid_values_found:
        # Handle cases where comparison with np.nan might be tricky if not done right
        # np.isnan can check for actual nans, others are direct comparisons
        has_actual_invalid = any(not (isinstance(x, float) and np.isnan(x)) for x in invalid_values_found if x not in valid_genotype_set)

        # Refined check:
        truly_invalid = []
        for val in invalid_values_found:
            is_nan_in_data = isinstance(val, float) and np.isnan(val)
            is_nan_in_valid_set = any(isinstance(vg, float) and np.isnan(vg) for vg in valid_genotype_set)
            if is_nan_in_data and is_nan_in_valid_set:
                continue # It's a valid NaN
            elif not is_nan_in_data and val in valid_genotype_set: # Non-NaN value found in valid_genotype_set
                continue
            else: # Genuinely invalid or NaN not in valid_genotype_set
                truly_invalid.append(val)

        if truly_invalid:
             print(f"Error: Invalid genotype values found: {truly_invalid}. Expected values in {valid_genotypes}.")
             # Optionally, one might choose to proceed with other checks or stop. For now, let's stop critical validation.
             return False, pd.DataFrame()


    print("Genotype value check passed (all values are within the valid set).")

    # Calculate SNP-wise statistics: call rate, MAF
    num_animals = genomic_data.num_animals
    for snp_name in genomic_data.snp_names:
        snp_column = genotypes_df[snp_name]
        missing_count = snp_column.isnull().sum()
        call_rate = (num_animals - missing_count) / num_animals if num_animals > 0 else 0

        # Calculate MAF (Minor Allele Frequency)
        # Assumes genotypes 0, 1, 2 represent counts of one allele (e.g., allele 'B')
        # p = (2 * count(BB) + count(AB)) / (2 * num_called_genotypes)
        # MAF = min(p, 1-p)
        num_called_genotypes = num_animals - missing_count
        maf = np.nan # Default to NaN if cannot be calculated
        if num_called_genotypes > 0:
            count_0 = snp_column[snp_column == 0].count() # AA
            count_1 = snp_column[snp_column == 1].count() # AB
            count_2 = snp_column[snp_column == 2].count() # BB

            # Ensure only valid genotypes contribute to allele count
            if count_0 + count_1 + count_2 == num_called_genotypes: # All called genotypes are 0, 1, or 2
                allele_b_count = (2 * count_2) + count_1
                total_alleles = 2 * num_called_genotypes
                if total_alleles > 0:
                    p = allele_b_count / total_alleles
                    maf = min(p, 1 - p)
            else:
                print(f"Warning: SNP {snp_name} contains non-0,1,2 values among called genotypes. MAF calculation may be affected.")

        snp_stats_list.append({'snp': snp_name, 'call_rate': call_rate, 'maf': maf, 'missing_count': missing_count})

    snp_stats_df = pd.DataFrame(snp_stats_list)

    # Calculate animal-wise call rates
    # animal_call_rates = 1 - genotypes_df.isnull().sum(axis=1) / genomic_data.num_snps if genomic_data.num_snps > 0 else pd.Series([0]*genomic_data.num_animals)
    if genomic_data.num_snps > 0:
        called_per_animal = genotypes_df.notnull().sum(axis=1)
        animal_call_rates = called_per_animal / genomic_data.num_snps
    else:
        animal_call_rates = pd.Series([0.0]*genomic_data.num_animals, index=genotypes_df.index)

    print("\n--- SNP Statistics ---")
    print(snp_stats_df.head())
    print(f"\nOverall mean SNP call rate: {snp_stats_df['call_rate'].mean():.3f}" if not snp_stats_df.empty else "Overall mean SNP call rate: N/A")
    print(f"Overall mean MAF: {snp_stats_df['maf'].mean():.3f}" if not snp_stats_df.empty and snp_stats_df['maf'].notna().any() else "Overall mean MAF: N/A")


    print("\n--- Animal Call Rates ---")
    animal_summary_df = pd.DataFrame({
        'AnimalID': genomic_data.animal_ids,
        'call_rate': animal_call_rates.values if isinstance(animal_call_rates, pd.Series) else animal_call_rates
    })
    print(animal_summary_df.head())
    print(f"\nOverall mean animal call rate: {animal_summary_df['call_rate'].mean():.3f}" if not animal_summary_df.empty else "Overall mean animal call rate: N/A")

    print("\nGenomic data validation checks completed.")
    return True, snp_stats_df


def filter_by_call_rate_snps(genomic_data: GenomicData, min_call_rate: float = 0.90) -> GenomicData:
    """
    Filters SNPs based on call rate.

    Args:
        genomic_data (GenomicData): The GenomicData object to filter.
        min_call_rate (float): Minimum call rate for a SNP to be retained. (e.g., 0.9 means 90% call rate)

    Returns:
        GenomicData: A new GenomicData object with low call rate SNPs removed.
    """
    if not isinstance(genomic_data, GenomicData) or genomic_data.data.empty:
        print("Warning: Genomic data is empty or invalid. No SNP filtering by call rate applied.")
        return genomic_data

    genotypes_df_full = genomic_data.data # Includes AnimalID
    genotypes_only_df = genomic_data.get_genotypes() # Excludes AnimalID

    num_animals = genomic_data.num_animals
    if num_animals == 0:
        print("Warning: No animals in genomic data. No SNP filtering by call rate applied.")
        return genomic_data

    called_counts = genotypes_only_df.notnull().sum(axis=0) # Summing non-nulls per SNP (column)
    call_rates = called_counts / num_animals

    snps_to_keep = call_rates[call_rates >= min_call_rate].index.tolist()
    snps_removed = call_rates[call_rates < min_call_rate].index.tolist()

    if snps_removed:
        print(f"Filtering SNPs by call rate < {min_call_rate}: Removed {len(snps_removed)} SNPs: {snps_removed[:10]}...") # Print first 10 removed
    else:
        print(f"No SNPs removed by call rate filter (< {min_call_rate}).")

    # Reconstruct DataFrame with AnimalID and kept SNPs
    final_columns = [genomic_data.animal_id_col] + snps_to_keep
    filtered_df = genotypes_df_full[final_columns].copy()

    return GenomicData(filtered_df, animal_id_col=genomic_data.animal_id_col)


def filter_by_call_rate_animals(genomic_data: GenomicData, min_call_rate: float = 0.90) -> GenomicData:
    """
    Filters animals based on call rate.

    Args:
        genomic_data (GenomicData): The GenomicData object to filter.
        min_call_rate (float): Minimum call rate for an animal to be retained.

    Returns:
        GenomicData: A new GenomicData object with low call rate animals removed.
    """
    if not isinstance(genomic_data, GenomicData) or genomic_data.data.empty:
        print("Warning: Genomic data is empty or invalid. No animal filtering by call rate applied.")
        return genomic_data

    genotypes_only_df = genomic_data.get_genotypes() # Excludes AnimalID
    num_snps = genomic_data.num_snps

    if num_snps == 0:
        print("Warning: No SNPs in genomic data. No animal filtering by call rate applied.")
        return genomic_data

    called_counts = genotypes_only_df.notnull().sum(axis=1) # Summing non-nulls per animal (row)
    call_rates = called_counts / num_snps

    animals_to_keep_mask = call_rates >= min_call_rate

    # Use the mask on the original DataFrame which includes the animal ID column
    filtered_df = genomic_data.data[animals_to_keep_mask].copy()

    num_removed = genomic_data.num_animals - filtered_df.shape[0]
    if num_removed > 0:
        removed_animal_ids = genomic_data.data[~animals_to_keep_mask][genomic_data.animal_id_col].tolist()
        print(f"Filtering animals by call rate < {min_call_rate}: Removed {num_removed} animals: {removed_animal_ids[:10]}...")
    else:
        print(f"No animals removed by call rate filter (< {min_call_rate}).")

    return GenomicData(filtered_df, animal_id_col=genomic_data.animal_id_col)


def filter_by_maf(genomic_data: GenomicData, min_maf: float = 0.01, valid_genotypes_for_maf: list = [0,1,2]) -> GenomicData:
    """
    Filters SNPs based on Minor Allele Frequency (MAF).

    Args:
        genomic_data (GenomicData): The GenomicData object to filter.
        min_maf (float): Minimum MAF for a SNP to be retained.
        valid_genotypes_for_maf (list): Genotype codes to consider for MAF calculation (e.g. [0,1,2]). SNPs with other called genotypes will have MAF as NaN.


    Returns:
        GenomicData: A new GenomicData object with low MAF SNPs removed.
    """
    if not isinstance(genomic_data, GenomicData) or genomic_data.data.empty:
        print("Warning: Genomic data is empty or invalid. No MAF filtering applied.")
        return genomic_data

    genotypes_df_full = genomic_data.data # Includes AnimalID
    genotypes_only_df = genomic_data.get_genotypes() # Excludes AnimalID
    num_animals = genomic_data.num_animals

    if num_animals == 0 or genotypes_only_df.empty:
        print("Warning: No animals or SNP data available. No MAF filtering applied.")
        return genomic_data

    snps_to_keep = []
    snps_removed_low_maf = []

    valid_geno_set_maf = set(valid_genotypes_for_maf)

    for snp_name in genomic_data.snp_names:
        snp_column = genotypes_only_df[snp_name]
        called_genotypes = snp_column.dropna()

        num_called_genotypes = len(called_genotypes)
        maf = np.nan

        if num_called_genotypes > 0:
            # Check if all called genotypes are in the valid set for MAF (e.g. 0,1,2)
            are_all_called_valid = all(g in valid_geno_set_maf for g in called_genotypes)
            if not are_all_called_valid:
                print(f"Warning: SNP {snp_name} contains genotypes outside {valid_genotypes_for_maf} among called values. MAF will be NaN, SNP will be kept unless min_maf is NaN (not typical).")
                # Decide behavior: keep SNP if MAF is NaN, or remove? Typically keep unless min_maf is also NaN.
                # If min_maf is not NaN, this SNP will be kept because (np.nan >= min_maf) is False.
            else:
                count_0 = called_genotypes[called_genotypes == 0].count()
                count_1 = called_genotypes[called_genotypes == 1].count()
                count_2 = called_genotypes[called_genotypes == 2].count() # Assumes 0,1,2 coding

                allele_b_count = (2 * count_2) + count_1
                total_alleles = 2 * num_called_genotypes
                if total_alleles > 0:
                    p = allele_b_count / total_alleles
                    maf = min(p, 1 - p)

        # Decision to keep SNP
        if pd.isna(maf): # If MAF could not be calculated (e.g. non 0,1,2 genotypes or no called)
            snps_to_keep.append(snp_name) # Keep SNPs where MAF is NaN (conservative)
            # print(f"Info: SNP {snp_name} MAF is NaN. Kept by default.")
        elif maf >= min_maf:
            snps_to_keep.append(snp_name)
        else:
            snps_removed_low_maf.append(snp_name)

    if snps_removed_low_maf:
        print(f"Filtering SNPs by MAF < {min_maf}: Removed {len(snps_removed_low_maf)} SNPs: {snps_removed_low_maf[:10]}...")
    else:
        print(f"No SNPs removed by MAF filter (< {min_maf}).")

    final_columns = [genomic_data.animal_id_col] + snps_to_keep
    filtered_df = genotypes_df_full[final_columns].copy()

    return GenomicData(filtered_df, animal_id_col=genomic_data.animal_id_col)

# Hardy-Weinberg Equilibrium filter can be complex, often requiring p-value calculations (e.g. chi-squared test)
# For now, this is out of scope for the initial setup as per instructions.
# def filter_by_hwe(genomic_data: GenomicData, p_value_threshold: float = 0.001):
# ... implementation ...


if __name__ == '__main__':
    # --- Test Phenotypic Data Validation ---
    # (Existing phenotypic and pedigree validation examples remain)
    # ...
    print("\\n--- Phenotypic and Pedigree validation examples are above ---")

    # --- Test Genomic Data Validation and Filtering ---
    print("\\n\\n--- Testing Genomic Data Validation & Filtering ---")
    # Sample genomic data for testing
    geno_test_data_dict = {
        'SampleID': ['Ind1', 'Ind2', 'Ind3', 'Ind4', 'Ind5', 'Ind6'],
        'SNP1': [0, 1, 2, 0, 1, 0],          # Good SNP
        'SNP2': [1, np.nan, 0, 2, 1, 1],     # One missing
        'SNP3': [2, 2, np.nan, np.nan, 2, 2], # Two missing
        'SNP4': [0, 0, 0, 0, 0, 0],          # Monomorphic (MAF=0)
        'SNP5': [0, 1, 0, 1, np.nan, np.nan], # Low call rate SNP, also low MAF among called
        'SNP6': [0, 1, 2, 1, 0, 3]           # Contains an invalid genotype '3'
    }
    raw_geno_df = pd.DataFrame(geno_test_data_dict)
    genomic_data_obj = GenomicData(raw_geno_df, animal_id_col='SampleID')
    print("\\nOriginal GenomicData object:")
    print(genomic_data_obj)

    print("\\n--- Validating Genomic Data (with an invalid genotype) ---")
    is_valid, snp_stats = validate_genomic_data(genomic_data_obj)
    print(f"Is valid (initial check with invalid genotype): {is_valid}")
    if not snp_stats.empty:
        print("SNP stats from initial validation (should be empty due to invalid genotype):")
        print(snp_stats)

    # Correct the invalid genotype for further tests
    corrected_geno_df = raw_geno_df.copy()
    corrected_geno_df.loc[corrected_geno_df['SNP6'] == 3, 'SNP6'] = np.nan # Set '3' to NaN
    genomic_data_corrected = GenomicData(corrected_geno_df, animal_id_col='SampleID')
    print("\\nCorrected GenomicData object (invalid genotype 3 -> NaN):")
    print(genomic_data_corrected)

    print("\\n--- Validating Corrected Genomic Data ---")
    is_valid_corrected, snp_stats_corrected = validate_genomic_data(genomic_data_corrected)
    print(f"Is valid (corrected): {is_valid_corrected}")
    if not snp_stats_corrected.empty:
        print("SNP stats from corrected validation:")
        print(snp_stats_corrected)

    print("\\n--- Filtering SNPs by Call Rate (min_call_rate = 0.7, SNP5 should be removed) ---")
    # SNP1: 6/6=1.0, SNP2: 5/6=0.833, SNP3: 4/6=0.667, SNP4: 6/6=1.0, SNP5: 4/6=0.667, SNP6: 5/6=0.833 (after correction)
    # So SNP3 and SNP5 should be removed if min_call_rate = 0.7
    filtered_call_rate_snps = filter_by_call_rate_snps(genomic_data_corrected, min_call_rate=0.7)
    print("GenomicData after SNP call rate filtering (SNP3, SNP5 removed):")
    print(filtered_call_rate_snps)
    if not filtered_call_rate_snps.data.empty:
      print(f"Remaining SNPs: {filtered_call_rate_snps.snp_names}")


    print("\\n--- Filtering Animals by Call Rate (min_call_rate = 0.8) ---")
    # Original (corrected) data: 6 SNPs.
    # Ind1: 6/6=1.0
    # Ind2: 4/6=0.66 (SNP2, SNP5 are NaN) - Remove
    # Ind3: 4/6=0.66 (SNP3, SNP5 are NaN) - Remove
    # Ind4: 5/6=0.83
    # Ind5: 4/6=0.66 (SNP3, SNP5 are NaN) - Remove
    # Ind6: 5/6=0.83
    # So Ind2, Ind3, Ind5 should be removed if min_call_rate = 0.8
    # Using genomic_data_corrected for this test:
    filtered_call_rate_animals = filter_by_call_rate_animals(genomic_data_corrected, min_call_rate=0.8)
    print("GenomicData after animal call rate filtering (Ind2, Ind3, Ind5 removed):")
    print(filtered_call_rate_animals)
    if not filtered_call_rate_animals.data.empty:
      print(f"Remaining animals: {filtered_call_rate_animals.animal_ids}")

    print("\\n--- Filtering SNPs by MAF (min_maf = 0.1, using corrected data) ---")
    # On corrected data:
    # SNP1: [0,1,2,0,1,0]. p = (2*1+1*2)/(2*6) = 4/12 = 0.33. MAF = 0.33. Keep.
    # SNP2: [1,nan,0,2,1,1]. Called: [1,0,2,1,1]. N=5. p = (2*1+1*3)/(2*5) = 5/10 = 0.5. MAF = 0.5. Keep.
    # SNP3: [2,2,nan,nan,2,2]. Called: [2,2,2,2]. N=4. p = (2*4)/(2*4) = 1. MAF = 0. Keep. (Removed by MAF < 0.1)
    # SNP4: [0,0,0,0,0,0]. Called: [0,0,0,0,0,0]. N=6. p = 0. MAF = 0. Keep. (Removed by MAF < 0.1)
    # SNP5: [0,1,nan,nan,0,1]. Called: [0,1,0,1]. N=4. p = (2*0+1*2)/(2*4) = 2/8 = 0.25. MAF = 0.25. Keep.
    # SNP6: [0,1,2,1,0,nan]. Called: [0,1,2,1,0]. N=5. p = (2*1+1*2)/(2*5) = 4/10 = 0.4. MAF = 0.4. Keep.
    # So SNP3, SNP4 should be removed by MAF < 0.1
    filtered_maf_snps = filter_by_maf(genomic_data_corrected, min_maf=0.1)
    print("GenomicData after MAF filtering (SNP3, SNP4 removed):")
    print(filtered_maf_snps)
    if not filtered_maf_snps.data.empty:
      print(f"Remaining SNPs: {filtered_maf_snps.snp_names}")

    print("\\n--- Chained Filtering Example (Animal Call Rate -> SNP Call Rate -> MAF) ---")
    # Start with corrected data
    # 1. Animal filter (min_call_rate=0.8) -> Ind1, Ind4, Ind6 remain
    step1_animal_filter = filter_by_call_rate_animals(genomic_data_corrected, min_call_rate=0.8)
    print(f"Animals after step 1: {step1_animal_filter.animal_ids}")
    # Data for Ind1, Ind4, Ind6:
    # SampleID  SNP1  SNP2  SNP3  SNP4  SNP5  SNP6
    # Ind1       0     1   2.0     0   0.0   0.0
    # Ind4       0     2   NaN     0   1.0   1.0
    # Ind6       0     1   2.0     0   NaN   NaN

    # 2. SNP filter (min_call_rate=0.7) on result of step 1. (3 animals now)
    # SNP1: 3/3=1.0
    # SNP2: 3/3=1.0
    # SNP3: 2/3=0.66 -> remove SNP3
    # SNP4: 3/3=1.0
    # SNP5: 1/3=0.33 -> remove SNP5
    # SNP6: 1/3=0.33 -> remove SNP6
    step2_snp_call_filter = filter_by_call_rate_snps(step1_animal_filter, min_call_rate=0.7)
    print(f"SNPs after step 2 (SNP3, SNP5, SNP6 removed): {step2_snp_call_filter.snp_names}")
    # Remaining SNPs: SNP1, SNP2, SNP4. Data for Ind1, Ind4, Ind6:
    # SampleID  SNP1  SNP2  SNP4
    # Ind1         0     1     0
    # Ind4         0     2     0
    # Ind6         0     1     0

    # 3. MAF filter (min_maf=0.01) on result of step 2
    # SNP1: [0,0,0]. MAF=0. -> remove SNP1
    # SNP2: [1,2,1]. p=(2*1+1*2)/(2*3)=4/6=0.66. MAF=0.33. Keep.
    # SNP4: [0,0,0]. MAF=0. -> remove SNP4
    step3_maf_filter = filter_by_maf(step2_snp_call_filter, min_maf=0.01) # Stricter MAF might remove more
    print("GenomicData after chained filtering:")
    print(step3_maf_filter)
    if not step3_maf_filter.data.empty:
        print(f"Remaining animals: {step3_maf_filter.animal_ids}")
        print(f"Remaining SNPs: {step3_maf_filter.snp_names}")
        print("Final data head:")
        print(step3_maf_filter.data.head())

    print("\\n--- Test with empty GenomicData object for filters ---")
    empty_gdata = GenomicData(pd.DataFrame(columns=['AnimalID', 'SNP1']), animal_id_col='AnimalID')
    filter_by_call_rate_snps(empty_gdata)
    filter_by_call_rate_animals(empty_gdata)
    filter_by_maf(empty_gdata)
