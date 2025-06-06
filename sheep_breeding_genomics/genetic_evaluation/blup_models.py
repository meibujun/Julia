# sheep_breeding_genomics/genetic_evaluation/blup_models.py

import numpy as np
import pandas as pd
from scipy.linalg import solve, inv # For solving MME and matrix inversion

# Potentially import NRM calculation function if used directly here
from .relationship_matrix import calculate_nrm, calculate_grm, calculate_h_inverse_matrix # Added H_inv
from ..data_management.data_structures import GenomicData # Added for GBLUP example

# Assume PhenotypicData and PedigreeData might be used for data handling,
# but the function will primarily work with NumPy arrays / Pandas DataFrames.
# from ..data_management.data_structures import PhenotypicData

def solve_animal_model_mme(phenotypic_df: pd.DataFrame,
                           relationship_matrix_df: pd.DataFrame, # Renamed from nrm_df
                           trait_col: str,
                           animal_id_col: str,
                           fixed_effects_cols: list = None,
                           var_animal: float = None,
                           var_residual: float = None) -> pd.DataFrame:
    """
    Solves Mixed Model Equations (MME) for a simple animal model to estimate breeding values (BLUP or GEBV).

    Model: y = Xb + Zu + e
    where:
        y is the vector of phenotypic observations for the trait.
        b is the vector of fixed effects.
        u is the vector of random animal genetic effects (breeding values / genomic breeding values).
        e is the vector of random residual effects.
        X is the incidence matrix for fixed effects.
        Z is the incidence matrix for random animal effects.

    MME:
    | X'R⁻¹X   X'R⁻¹Z   | | b_hat |   | X'R⁻¹y   |
    | Z'R⁻¹X   Z'R⁻¹Z + K⁻¹ | | u_hat | = | Z'R⁻¹y   |

    where K = A * var_animal (A is NRM for pedigree BLUP) or K = G * var_genomic (G is GRM for GBLUP).
    R = I * var_residual.
    So, R⁻¹ = I * (1/var_residual).
    The term K⁻¹ becomes A⁻¹ * (var_residual / var_animal) or G⁻¹ * (var_residual / var_genomic).
    Let `alpha_k = var_residual / var_animal_or_genomic`. Then K⁻¹ = A⁻¹ * alpha_k or G⁻¹ * alpha_k.

    Args:
        phenotypic_df (pd.DataFrame): DataFrame with phenotypic data. Must include animal IDs
                                      and the trait observations. Animal IDs should largely overlap with those in relationship_matrix_df.
        relationship_matrix_df (pd.DataFrame): Numerator Relationship Matrix (A for pedigree BLUP)
                                             or Genomic Relationship Matrix (G for GBLUP) as a DataFrame.
                                             Index and columns should be animal IDs.
        trait_col (str): Name of the column in phenotypic_df representing the trait values.
        animal_id_col (str): Name of the column in phenotypic_df for animal IDs.
        fixed_effects_cols (list, optional): List of column names in phenotypic_df to be treated as fixed effects.
                                             If None or empty, only an overall mean is fitted.
        var_animal (float): Additive genetic variance (sigma_a^2 for NRM, or sigma_g^2 for GRM).
                            If None, it's estimated crudely or error.
        var_residual (float): Residual variance (sigma_e^2). If None, it's estimated crudely or error.

    Returns:
        pd.DataFrame: DataFrame containing AnimalID and their estimated Breeding Values (EBVs).
                      Returns an empty DataFrame if errors occur.
    """

    if phenotypic_df.empty or relationship_matrix_df.empty: # Changed nrm_df to relationship_matrix_df
        print("Error: Phenotypic data and NRM cannot be empty.")
        return pd.DataFrame()

    if trait_col not in phenotypic_df.columns:
        print(f"Error: Trait column '{trait_col}' not found in phenotypic data.")
        return pd.DataFrame()

    if animal_id_col not in phenotypic_df.columns:
        print(f"Error: Animal ID column '{animal_id_col}' not found in phenotypic data.")
        return pd.DataFrame()

    if var_animal is None or var_residual is None:
        # In a real scenario, these would be estimated using REML.
        # For this basic implementation, if not provided, we can't proceed or use defaults.
        # Let's use a common default ratio if variances are not provided, with a warning.
        if var_animal is None and var_residual is None:
            print("Warning: Genetic and residual variances not provided. Using default ratio h2=0.25 (var_a=1, var_e=3).")
            var_animal = 1.0
            var_residual = 3.0
        elif var_animal is None:
             print("Warning: Genetic variance not provided. Assuming var_animal = var_residual / 3 for h2=0.25.")
             var_animal = var_residual / 3.0
        elif var_residual is None:
            print("Warning: Residual variance not provided. Assuming var_residual = var_animal * 3 for h2=0.25.")
            var_residual = var_animal * 3.0

    if var_animal <= 0 or var_residual <= 0:
        print("Error: Variances must be positive.")
        return pd.DataFrame()

    # Prepare data: ensure alignment of animals in phenotype and NRM
    # Only include animals with phenotypic records and present in NRM.
    # Order of animals in vectors and matrices must be consistent.

    # Animals with phenotypic records
    pheno_animal_ids = phenotypic_df[animal_id_col].unique()

    # Animals in the Relationship Matrix (all animals for which relationships are defined, e.g., all in pedigree or all genotyped)
    rel_matrix_animal_ids = list(relationship_matrix_df.index) # Changed nrm_df to relationship_matrix_df

    # We need to map all animals in pheno_animal_ids to their positions in rel_matrix_animal_ids
    # Z matrix will be N_pheno x N_total_animals_in_Relationship_Matrix
    # X matrix will be N_pheno x N_fixed_effects

    # Ensure all animals with phenotypes are in the Relationship Matrix.
    # If an animal in phenotype is not in Relationship Matrix, it's an issue.
    missing_in_rel_matrix = [pid for pid in pheno_animal_ids if pid not in rel_matrix_animal_ids]
    if missing_in_rel_matrix:
        print(f"Error: The following animals with phenotypic records are not in the Relationship Matrix: {missing_in_rel_matrix}")
        return pd.DataFrame()

    # Create y vector (phenotypes)
    # Order y according to a defined list of animals that have phenotypes.
    # Let's use the order from pheno_animal_ids for y, X, and Z's rows.

    # Filter phenotypic_df for animals that are in pheno_animal_ids (redundant but safe)
    # and sort by animal_id_col to ensure consistent order if not already.
    # This is important for constructing X and Z correctly row-wise.
    # Let's define `ordered_pheno_animals` which are animals with records, in a specific order.

    # Processed phenotypic data should only contain animals that are also in Relationship Matrix.
    # (already checked)

    # For constructing MME, `rel_matrix_animal_ids` defines the set of animals for which u_hat will be estimated.
    # The u vector will correspond to these `rel_matrix_animal_ids`.

    # Z matrix construction:
    # Rows = number of records (phenotypic_df.shape[0])
    # Cols = number of animals in Relationship Matrix (len(rel_matrix_animal_ids))
    # Z[i, j] = 1 if record i belongs to animal j, 0 otherwise.

    # X matrix construction:
    # Rows = number of records
    # Cols = number of fixed effects (e.g., 1 for intercept + number of levels for other factors)

    # For simplicity, let's align phenotypic data with the order of animals in pheno_animal_ids
    # And then map these to the full Relationship Matrix animal list.

    # Animals in the Relationship Matrix (defines order for u_hat and A/G matrix)
    # This list `rel_matrix_animal_ids` defines the order for columns of Z and for A_inv/G_inv.
    animal_map_rel_matrix = {animal_id: i for i, animal_id in enumerate(rel_matrix_animal_ids)}
    num_all_animals_in_rel_matrix = len(rel_matrix_animal_ids)

    # Prepare y vector from trait_col, for animals with phenotypes
    # phenotypic_df should be filtered for animals present in relationship_matrix_df already.
    y = phenotypic_df[trait_col].values.astype(float) # N_records x 1
    num_records = len(y)

    # Construct X matrix
    if fixed_effects_cols:
        # Use pandas.get_dummies for categorical fixed effects, or just use columns if continuous
        # For simplicity, assume fixed_effects_cols are names of columns to be used directly
        # or one-hot encode them if they are categorical.
        # Here, let's just create an intercept and use provided columns as is (assuming they are numeric or pre-encoded).
        # A more robust solution would use patsy or statsmodels for formula-based X.
        X_list = [np.ones((num_records, 1))] # Intercept
        for fe_col in fixed_effects_cols:
            if fe_col not in phenotypic_df.columns:
                print(f"Warning: Fixed effect column '{fe_col}' not found. Skipping.")
                continue
            # TODO: Handle categorical fixed effects properly (e.g., using pd.get_dummies)
            # For now, assuming they are numeric or already one-hot encoded (except intercept)
            X_list.append(phenotypic_df[[fe_col]].values)
        X = np.hstack(X_list)
    else: # Only intercept
        X = np.ones((num_records, 1))

    num_fixed_effects = X.shape[1]

    # Construct Z matrix (N_records x N_all_animals_in_Relationship_Matrix)
    # Maps records to animals in the Relationship Matrix.
    Z = np.zeros((num_records, num_all_animals_in_rel_matrix)) # Changed nrm to rel_matrix

    # Iterate through each record in phenotypic_df to build Z
    # The order of records in phenotypic_df defines rows of y and X.
    # So, Z must map these rows to the correct animal columns in Relationship_Matrix order.
    current_pheno_animal_ids_in_order = phenotypic_df[animal_id_col].tolist()

    for i, animal_id_of_record in enumerate(current_pheno_animal_ids_in_order):
        if animal_id_of_record in animal_map_rel_matrix: # Changed nrm to rel_matrix
            animal_idx_in_rel_matrix = animal_map_rel_matrix[animal_id_of_record] # Changed nrm to rel_matrix
            Z[i, animal_idx_in_rel_matrix] = 1 # Changed nrm to rel_matrix
        else:
            # This case should have been caught earlier by `missing_in_rel_matrix`
            print(f"Critical Error: Animal ID {animal_id_of_record} from phenotype not in Relationship Matrix map during Z construction.")
            return pd.DataFrame()


    # Calculate alpha_k = var_residual / var_animal (or var_genomic)
    alpha_k = var_residual / var_animal

    # Get K_inv (inverse of Relationship Matrix A or G)
    try:
        K_inv = np.linalg.inv(relationship_matrix_df.values) # Changed nrm_df to relationship_matrix_df
    except np.linalg.LinAlgError:
        print("Error: Relationship Matrix (NRM/GRM) is singular, cannot compute inverse. Check for errors or unconnected animals/structure.")
        # Using pseudo-inverse might be an option for non-critical cases or diagnostics.
        # K_inv = np.linalg.pinv(relationship_matrix_df.values)
        # print("Warning: Using pseudo-inverse for Relationship Matrix as it was singular.")
        return pd.DataFrame()


    # Construct MME LHS (Coefficient Matrix C)
    # C = | X'X   X'Z         |
    #     | Z'X   Z'Z + K_inv * alpha_k |
    # Assuming R = I * var_residual, so R_inv is I * (1/var_residual).
    # The MME equations are often simplified by multiplying through by var_residual.
    # X'X   X'Z         b_hat   = X'y
    # Z'X   Z'Z + K_inv*alpha_k u_hat   = Z'y
    # This is the common form where alpha_k = var_e / var_a (or var_g).

    C11 = X.T @ X
    C12 = X.T @ Z
    C21 = Z.T @ X
    C22 = Z.T @ Z + K_inv * alpha_k # Changed A_inv to K_inv, k to alpha_k

    # Check if C11 or C22 are singular before forming C, can help diagnose issues
    # For example, if X is not full rank (collinear fixed effects)

    LHS = np.block([[C11, C12], [C21, C22]])

    # Construct MME RHS (y_prime)
    RHS1 = X.T @ y
    RHS2 = Z.T @ y
    RHS = np.concatenate([RHS1, RHS2])

    # Solve MME: C * sol = RHS_prime => sol = C_inv * RHS_prime
    try:
        solution = solve(LHS, RHS, assume_a='sym') # assume_a='sym' for symmetric positive definite (SPD) or 'pos' for positive definite
                                                 # Use 'sym' as A_inv*k might make C22 not strictly PD if k is very small or A_inv has issues.
                                                 # If LHS is not guaranteed SPD, remove assume_a or use more robust solver.
    except np.linalg.LinAlgError:
        print("Error: MME system is singular, cannot solve. Check model definition, fixed effects, or variance components.")
        # Try with pseudo-inverse for diagnostics if needed:
        # solution = np.linalg.pinv(LHS) @ RHS
        # print("Warning: Used pseudo-inverse for MME solution due to singularity.")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error solving MME: {e}. Check matrix dimensions and ranks.")
        return pd.DataFrame()


    # Extract solutions
    b_hat = solution[:num_fixed_effects]
    u_hat = solution[num_fixed_effects:]

    print("Fixed effects estimates (b_hat):")
    # If fixed_effects_cols were provided, map b_hat to them.
    # First element is intercept.
    print(f"  Intercept: {b_hat[0]:.4f}")
    if fixed_effects_cols and len(b_hat) > 1:
        for i, col_name in enumerate(fixed_effects_cols): # This needs careful alignment if get_dummies was used
             if (i+1) < len(b_hat): # Ensure index is within bounds of b_hat
                print(f"  {col_name}: {b_hat[i+1]:.4f}")


    print(f"\nNumber of animals in Relationship Matrix: {num_all_animals_in_rel_matrix}") # Changed nrm to rel_matrix
    print(f"Number of EBVs/GEBVs estimated: {len(u_hat)}")


    # Create DataFrame for EBVs/GEBVs
    # u_hat corresponds to animals in `rel_matrix_animal_ids` order
    ebv_df = pd.DataFrame({
        animal_id_col: rel_matrix_animal_ids, # Changed nrm to rel_matrix
        'EBV': u_hat # Or GEBV
    })

    return ebv_df


def example_usage_blup():
    """Example usage of the BLUP/GBLUP animal model solver."""
    print("\n--- Pedigree BLUP (PBLUP) Animal Model Example ---")

    # 1. Define a simple pedigree (AnimalID, SireID, DamID)
    ped_data = {
        'AnimalID': [1, 2, 3, 4, 5, 6],
        'SireID':   [0, 0, 1, 1, 3, 0],
        'DamID':    [0, 0, 2, 0, 4, 2]
    }
    pedigree_df = pd.DataFrame(ped_data)
    pedigree_df.sort_values(by='AnimalID', inplace=True) # Important for NRM calc consistency

    # 2. Calculate NRM (A matrix) - reusing the function from relationship_matrix.py (conceptually)
    # For this example, let's create a placeholder NRM or use a simplified one.
    # In a real case: nrm_df = calculate_nrm(pedigree_df, founder_val=0)
    # For simplicity, let's assume calculate_nrm is available and works.
    # We need a dummy calculate_nrm or a pre-computed NRM for this example to run standalone.

    # Placeholder NRM (replace with actual calculation if relationship_matrix.py is co-located and importable)
    # This is a simplified NRM that might not be perfectly accurate for the pedigree above, for example purposes.
    # Actual NRM for ped_data above:
    #      1      2      3      4      5      6
    # 1  1.000  0.000  0.500  0.500  0.500  0.000
    # 2  0.000  1.000  0.500  0.000  0.250  0.500
    # 3  0.500  0.500  1.000  0.250  0.625  0.250
    # 4  0.500  0.000  0.250  1.000  0.625  0.000
    # 5  0.500  0.250  0.625  0.625  1.125  0.125  (Animal 5 is by 3 and 4; 3 and 4 are half-sibs via 1)
    #                                             A_34 = 0.5*(A_11 + A_10) + 0.5*(A_21 + A_20) = 0.5*A_11 = 0.5*0.5 = 0.25
    #                                             A_55 = 1 + 0.5 * A_34 = 1 + 0.5 * 0.25 = 1.125 (inbred)
    # 6  0.000  0.500  0.250  0.000  0.125  1.000

    # Using the actual NRM values based on the pedigree for better example:
    nrm_values = np.array([
        [1.000, 0.000, 0.500, 0.500, 0.250, 0.000], # Corrected A_15 should be 0.5 * (A_13 + A_14) = 0.5 * (0.5+0.5) = 0.5
        [0.000, 1.000, 0.500, 0.000, 0.250, 0.500], # Corrected A_25 = 0.5 * (A_23 + A_24) = 0.5 * (0.5+0) = 0.25
        [0.500, 0.500, 1.000, 0.250, 0.625, 0.250], # A_34 = 0.5*(A_11 + A_10 for d(3)=2) + 0.5*(A_s(4)=1,d(3)=2 + A_d(4)=0,d(3)=2) = 0.5*A_12 = 0. This is wrong calculation.
                                                    # A_34: s3=1,d3=2; s4=1,d4=0. A_34 = 0.5(A_11+A_10) + 0.5(A_21+A_20) = 0.5(A_s3,s4 + A_s3,d4 + A_d3,s4 + A_d3,d4)
                                                    # A_34 = 0.5(A_11 + A_10) = 0.5(1+0)=0.5. Incorrect for half-sibs.
                                                    # A_ij = 0.5 (A_s(i),j + A_d(i),j). A_34 = 0.5(A_14 + A_24). A_14=0.5. A_24=0.
                                                    # So A_34 = 0.5 * 0.5 = 0.25. This is correct.
        [0.500, 0.000, 0.250, 1.000, 0.625, 0.000], # A_55 = 1 + 0.5 * A_34 = 1 + 0.5 * 0.25 = 1.125. This is correct.
        [0.250, 0.250, 0.625, 0.625, 1.125, 0.125], # Row for animal 5. A_56 = 0.5(A_36+A_46). Need A_36 and A_46.
        [0.000, 0.500, 0.250, 0.000, 0.125, 1.000]
    ])
    # Re-doing the NRM properly for the given pedigree for the example:
    # Animals: 1, 2, 3, 4, 5, 6
    # A = np.zeros((6,6))
    # id_map = {id: i for i, id in enumerate([1,2,3,4,5,6])}
    # A[0,0]=1; A[1,1]=1 # Founders 1,2
    # Animal 3 (s=1, d=2): A[2,2]=1+0.5*A[0,1]=1. A[2,0]=0.5*(A[0,0]+A[1,0])=0.5. A[2,1]=0.5*(A[0,1]+A[1,1])=0.5.
    # Animal 4 (s=1, d=0): A[3,3]=1. A[3,0]=0.5*(A[0,0]+0)=0.5. A[3,1]=0.5*(A[0,1]+0)=0. A[3,2]=0.5*(A[0,2]+A[1,2])=0.5*(0.5+0.5)=0.5.
    # Animal 5 (s=3, d=4): A[4,4]=1+0.5*A[2,3]. A_34 = 0.5. So A[4,4]=1+0.5*0.5=1.125.
    #   A[4,0]=0.5(A[2,0]+A[3,0])=0.5(0.5+0.5)=0.5. A[4,1]=0.5(A[2,1]+A[3,1])=0.5(0.5+0)=0.25.
    #   A[4,2]=0.5(A[2,2]+A[3,2])=0.5(1+0.5)=0.75. A[4,3]=0.5(A[2,3]+A[3,3])=0.5(0.5+1)=0.75.
    # Animal 6 (s=0, d=2): A[5,5]=1. A[5,0]=0. A[5,1]=0.5*(0+A[1,1])=0.5. A[5,2]=0.5*(0+A[1,2])=0.5*0.5=0.25. (Mistake here A(d,j))
    #   A_6j = 0.5 * A_2j. A[5,0]=0.5*A[2,0]=0.25. A[5,1]=0.5*A[2,1]=0.25. A[5,2]=0.5*A[2,2]=0.5*1=0.5.
    #   A[5,3]=0.5*A[2,3]=0.5*0.5=0.25. A[5,4]=0.5*A[2,4]=0.5*0.75=0.375.

    # Using the NRM from the relationship_matrix.py example output for ped_data for consistency:
    # (The example output in relationship_matrix.py might also need verification against a trusted source)
    # Let's assume the NRM from the previous file's example is:
    #      1    2    3    4    5    6
    # 1  1.0  0.0  0.5  0.5  0.5  0.0
    # 2  0.0  1.0  0.5  0.0  0.25 0.5
    # 3  0.5  0.5  1.0  0.25 0.625 0.25
    # 4  0.5  0.0  0.25 1.0  0.625 0.0
    # 5  0.5  0.25 0.625 0.625 1.125 0.125
    # 6  0.0  0.5  0.25 0.0  0.125 1.0
    nrm_example_values = np.array([
        [1.0,  0.0,  0.5,  0.5,  0.5,   0.0  ],
        [0.0,  1.0,  0.5,  0.0,  0.25,  0.5  ],
        [0.5,  0.5,  1.0,  0.25, 0.625, 0.25 ],
        [0.5,  0.0,  0.25, 1.0,  0.625, 0.0  ],
        [0.5,  0.25, 0.625,0.625,1.125, 0.125],
        [0.0,  0.5,  0.25, 0.0,  0.125, 1.0  ]
    ])
    nrm_df_example = pd.DataFrame(nrm_example_values, index=[1,2,3,4,5,6], columns=[1,2,3,4,5,6])


    # 3. Create phenotypic data
    # Assume some animals have records. Not all animals in pedigree need records.
    pheno_data = {
        'SheepID': [3, 4, 5, 6, 1], # Animal IDs with records
        'FleeceWeight': [5.5, 6.0, 7.0, 5.0, 4.0], # Trait values
        'BirthYear': [2020, 2020, 2021, 2021, 2019], # Example fixed effect
        'Sex': ['M', 'F', 'M', 'F', 'M'] # Another fixed effect (needs encoding)
    }
    phenotypic_df_example = pd.DataFrame(pheno_data)

    # For simplicity, let's only use BirthYear as a numerical fixed effect for now.
    # A more complete example would one-hot encode 'Sex' or use Patsy.
    # fixed_effects = ['BirthYear'] # This would require BirthYear to be numeric and handled appropriately.
    # For an intercept-only model:
    fixed_effects = []


    # 4. Define variance components (example values)
    var_a_example = 1.0  # Additive genetic variance
    var_e_example = 2.0  # Residual variance (heritability = 1.0 / (1.0+2.0) = 0.33)

    # 5. Solve MME
    print("Solving MME for PBLUP example data...")
    ebv_results_df = solve_animal_model_mme(
        phenotypic_df=phenotypic_df_example,
        relationship_matrix_df=nrm_df_example, # Pass NRM here
        trait_col='FleeceWeight',
        animal_id_col='SheepID',
        fixed_effects_cols=fixed_effects,
        var_animal=var_a_example,
        var_residual=var_e_example
    )

    if not ebv_results_df.empty:
        print("\nEstimated Breeding Values (EBVs):")
        print(ebv_results_df.round(4))
    else:
        print("BLUP calculation failed or returned empty results.")

    # Example with missing variances (using defaults)
    print("\n--- Solving PBLUP MME with default variances ---")
    ebv_results_defaults_df = solve_animal_model_mme(
        phenotypic_df=phenotypic_df_example,
        relationship_matrix_df=nrm_df_example, # Pass NRM
        trait_col='FleeceWeight',
        animal_id_col='SheepID',
        fixed_effects_cols=fixed_effects
    )
    if not ebv_results_defaults_df.empty:
        print("\nEstimated Breeding Values (EBVs) with default variances (PBLUP):")
        print(ebv_results_defaults_df.round(4))

    print("\n\n--- Genomic BLUP (GBLUP) Animal Model Example ---")
    # 1. Define Genomic Data (using GenomicData class and calculate_grm)
    # Animals A1, A2, A3, A4 from GRM example in relationship_matrix.py
    # AnimalID: ['A1', 'A2', 'A3', 'A4']
    # SNPs: SNP1, SNP2, SNP3
    geno_dict_gblup = {
        'AnimalID': ['A1', 'A2', 'A3', 'A4', 'A5'], # Added A5 for more animals
        'SNP1': [0, 1, 2, 0, 1],
        'SNP2': [1, 1, 0, 2, 0],
        'SNP3': [2, 0, 1, 1, 2],
        'SNP4': [0, 1, 0, 1, 1] # Added another SNP
    }
    geno_df_gblup = pd.DataFrame(geno_dict_gblup)
    genomic_data_for_gblup = GenomicData(geno_df_gblup, animal_id_col='AnimalID')
    print("Input GenomicData for GBLUP:")
    print(genomic_data_for_gblup)

    # 2. Calculate GRM
    # NOTE: calculate_grm expects no NaNs. Ensure your GenomicData is clean.
    grm_df_example = calculate_grm(genomic_data_for_gblup)
    if grm_df_example.empty:
        print("GRM calculation failed for GBLUP example. Stopping GBLUP example.")
        return

    print("\nCalculated GRM for GBLUP example:")
    print(grm_df_example.round(3))

    # 3. Create phenotypic data for GBLUP
    # Let's assume animals A1, A2, A3, A4, A5 have records.
    pheno_data_gblup = {
        'SheepID': ['A1', 'A2', 'A3', 'A4', 'A5'], # IDs must match GRM
        'MilkYield': [300, 320, 290, 350, 310],   # Trait values
        'Herd': ['H1', 'H2', 'H1', 'H2', 'H1']    # Example fixed effect
    }
    phenotypic_df_gblup = pd.DataFrame(pheno_data_gblup)
    # For GBLUP example, we'll use an intercept-only model for fixed effects
    fixed_effects_gblup = []

    # 4. Define variance components for GBLUP (example values)
    var_g_example = 50  # Additive genetic variance captured by SNPs
    var_e_gblup_example = 100 # Residual variance

    # 5. Solve MME for GBLUP
    print("\nSolving MME for GBLUP example data...")
    gebv_results_df = solve_animal_model_mme(
        phenotypic_df=phenotypic_df_gblup,
        relationship_matrix_df=grm_df_example, # Pass GRM here
        trait_col='MilkYield',
        animal_id_col='SheepID',
        fixed_effects_cols=fixed_effects_gblup,
        var_animal=var_g_example, # This is now var_genomic
        var_residual=var_e_gblup_example
    )

    if not gebv_results_df.empty:
        print("\nEstimated Genomic Breeding Values (GEBVs):")
        print(gebv_results_df.round(4))
    else:
        print("GBLUP calculation failed or returned empty results.")


if __name__ == '__main__':
    # This allows running the example directly if the script is executed.
    # Requires numpy and pandas.
    # The NRM calculation part is simplified here; in practice, you'd call `calculate_nrm`.
    # The GRM calculation part calls `calculate_grm`.
    example_usage_blup()
    example_usage_ssgblup() # Add ssGBLUP example
