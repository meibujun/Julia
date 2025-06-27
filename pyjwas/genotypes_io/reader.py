from typing import Union, Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import os

from ..core.genotypes import GenotypesData # The data structure class
from ..core.variance_covariance import VarianceCovariance # For G/genetic_variance
from .utils_geno import center_genotype_matrix_inplace, calculate_allele_frequencies_from_means, make_incidence_matrix_for_ids

# Placeholder for functions from tools4genotypes.jl that are more complex or MME-dependent
# e.g. align_genotypes, set_marker_hyperparameters_variances_and_pi
# These might become methods of MME or separate utility functions called during model setup.


def get_genotypes_data(
    source: Union[str, pd.DataFrame, np.ndarray],
    # Method parameters
    method: str = "BayesC",
    Pi: Union[float, Dict[Any, float]] = 0.0, # Default for ST, needs dict for MT
    estimate_pi: bool = True,
    # Variance parameters for G (genomic variance or marker effect variance)
    G_prior_value: Optional[Union[float, np.ndarray]] = None, # Prior mean for G
    G_is_marker_variance: bool = False, # If True, G_prior_value is for sigma_g^2, else for total sigma_a^2
    G_prior_df: float = 4.0,
    G_estimate_variance: bool = True,
    G_estimate_scale: bool = False, # Scale of prior for G
    G_constraint: bool = False, # For multi-trait G matrix
    # Formatting parameters for file reading
    file_separator: str = ',',
    file_has_header: bool = True, # True if marker IDs are in the first row
    # File/DataFrame structure
    obs_ids_col_name_or_index: Union[str, int] = 0, # For DataFrame or file with header
    marker_ids_header_row: Optional[int] = 0, # For file, row of marker IDs (0 if first line after main header)
                                              # None if no marker IDs in file and file_has_header=False
    # For direct array input
    obs_ids_list: Optional[List[str]] = None,
    marker_ids_list: Optional[List[str]] = None,
    # QC parameters
    quality_control: bool = True,
    maf_threshold: float = 0.01,
    missing_value_code: float = 9.0, # Numeric code for missing values in matrix
    # Other processing
    center_data: bool = True,
    double_precision: bool = False, # Output data type for matrix
    # Starting values (advanced)
    # starting_marker_effects: Optional[np.ndarray] = None # TODO: Add later if needed
    genotype_data_name: Optional[str] = "geno1" # Name for this GenotypesData object
) -> GenotypesData:
    """
    Reads, processes genotype data from various sources and returns a GenotypesData object.
    Combines logic from JWAS.jl's get_genotypes and parts of add_genotypes.

    Args:
        source: Path to genotype file (str), pandas DataFrame, or NumPy array.
                If file/DataFrame: first column is obs IDs, subsequent cols are markers.
                Marker IDs from header if file_has_header=True or DataFrame column names.
        method: Bayesian method (e.g., "BayesC", "RR-BLUP", "GBLUP").
        Pi: Prior for marker inclusion probability.
        estimate_pi: Whether to estimate Pi.
        G_prior_value: Prior mean for genomic variance (if G_is_marker_variance=False)
                       or marker effect variance (if G_is_marker_variance=True).
        G_is_marker_variance: Interprets G_prior_value accordingly.
        G_prior_df: Prior degrees of freedom for G.
        G_estimate_variance: Whether to estimate G.
        G_estimate_scale: Whether to estimate the scale of G's prior.
        G_constraint: For multi-trait, if G matrix is constrained (e.g., diagonal).
        file_separator: Delimiter for CSV file.
        file_has_header: If reading a file, indicates if the first row contains marker IDs.
                         The very first column is always assumed to be observation IDs.
        obs_ids_col_name_or_index: Column name or index for observation IDs if source is DataFrame.
                                   Not used for file reading (assumes first col).
        marker_ids_header_row: For file reading, if marker IDs are in a specific row (e.g. first actual data line).
                               Usually, if file_has_header=True, marker IDs are from the first line.
                               This gives more control. If None, and file_has_header=False, markers are 1..p.
        obs_ids_list: Provide if source is np.ndarray.
        marker_ids_list: Provide if source is np.ndarray.
        quality_control: Whether to perform QC (missing imputation, MAF filter).
        maf_threshold: MAF threshold for filtering markers.
        missing_value_code: Numeric value representing missing genotypes.
        center_data: Whether to center the genotype matrix.
        double_precision: If True, genotype matrix stored as float64, else float32.
        genotype_data_name: Name for this GenotypesData object.

    Returns:
        A configured GenotypesData object.
    """
    data_type = np.float64 if double_precision else np.float32

    raw_genotypes_matrix: np.ndarray
    current_obs_ids: List[str]
    current_marker_ids: List[str]

    if isinstance(source, str):
        print(f"Reading genotypes from file: {source}")
        # Read marker IDs from header line if file_has_header
        header_pd: Optional[int] = 0 if file_has_header else None # pandas header arg
        names_pd: Optional[List[str]] = None
        skiprows_pd: Optional[int] = None

        if file_has_header: # obsID in col 0, markerIDs in rest of row 0
             # Read just the header to get marker IDs
            try:
                header_df = pd.read_csv(source, delimiter=file_separator, header=None, nrows=1, dtype=str)
                # First element of first row is obs_id header, rest are marker_ids
                raw_marker_ids = header_df.iloc[0,1:].astype(str).tolist()
                skiprows_pd = 1 # Skip this header line when reading data
            except Exception as e:
                raise ValueError(f"Could not read header for marker IDs from {source}: {e}")
        else: # No header line for marker IDs
            raw_marker_ids = [] # Will be generated later based on number of columns

        # Read the data, first column as obs_id, rest as genotypes
        try:
            df = pd.read_csv(
                source,
                delimiter=file_separator,
                header=None, # We handle header separately or assume none
                skiprows=skiprows_pd,
                dtype=str, # Read all as string first to handle mixed types / missing strings
                na_values=str(missing_value_code) # Tell pandas what numeric missing is
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Genotype file not found: {source}")
        except Exception as e:
            raise ValueError(f"Error reading genotype file {source}: {e}")

        current_obs_ids = df.iloc[:, 0].astype(str).tolist()
        # Replace NA in genotype part with missing_value_code before converting to numeric
        genotype_df_part = df.iloc[:, 1:].fillna(str(missing_value_code))
        try:
            raw_genotypes_matrix = genotype_df_part.astype(data_type).values
        except ValueError as e:
             raise ValueError(f"Could not convert genotype data to numeric. Check for non-numeric values not matching missing_value_code ({missing_value_code}) or improper formatting. Error: {e}")


        if not raw_marker_ids: # If no header, generate marker IDs
            raw_marker_ids = [f"M{i+1}" for i in range(raw_genotypes_matrix.shape[1])]
        current_marker_ids = raw_marker_ids

    elif isinstance(source, pd.DataFrame):
        print("Reading genotypes from pandas DataFrame.")
        df_copy = source.copy()
        if isinstance(obs_ids_col_name_or_index, str): # Name
            current_obs_ids = df_copy.pop(obs_ids_col_name_or_index).astype(str).tolist()
        else: # Index
            current_obs_ids = df_copy.iloc[:, obs_ids_col_name_or_index].astype(str).tolist()
            df_copy.drop(df_copy.columns[obs_ids_col_name_or_index], axis=1, inplace=True)

        current_marker_ids = df_copy.columns.astype(str).tolist()
        raw_genotypes_matrix = df_copy.fillna(missing_value_code).astype(data_type).values

    elif isinstance(source, np.ndarray):
        print("Reading genotypes from NumPy array.")
        raw_genotypes_matrix = source.astype(data_type)
        n_obs, n_mkrs = raw_genotypes_matrix.shape
        current_obs_ids = obs_ids_list if obs_ids_list else [f"ID{i+1}" for i in range(n_obs)]
        current_marker_ids = marker_ids_list if marker_ids_list else [f"M{i+1}" for i in range(n_mkrs)]
        if len(current_obs_ids) != n_obs or len(current_marker_ids) != n_mkrs:
            raise ValueError("Provided obs_ids_list or marker_ids_list length mismatch with NumPy array dimensions.")
    else:
        raise TypeError("source must be a file path (str), pandas DataFrame, or NumPy array.")

    # --- Initial state ---
    processed_genotypes = raw_genotypes_matrix.copy()
    final_marker_ids = list(current_marker_ids) # Copy to allow modification

    # --- Quality Control ---
    if quality_control:
        print("Performing Quality Control...")
        # 1. Missing value imputation (with column means of non-missing)
        for j in range(processed_genotypes.shape[1]):
            col_data = processed_genotypes[:,j]
            missing_mask = (col_data == missing_value_code) | np.isnan(col_data) # Handle both numeric code and actual NaNs

            # Calculate mean from non-missing values
            non_missing_values = col_data[~missing_mask]
            if len(non_missing_values) > 0:
                col_mean = np.mean(non_missing_values)
                processed_genotypes[missing_mask, j] = col_mean
            else: # All values in column are missing
                processed_genotypes[missing_mask, j] = 0 # Impute with 0 or global mean? For now, 0.
                print(f"Warning: Marker {final_marker_ids[j]} has all missing values. Imputed with 0.")
        print(f"  Missing values ({missing_value_code} or NaN) imputed with column means.")

        # Check for out-of-range genotypes (0-2 expected for diploid biallelic)
        if np.any((processed_genotypes < 0) | (processed_genotypes > 2)):
            print("Warning: Genotype scores out of the typical 0-2 range found after imputation.")


    # --- Centering (before MAF, as MAF uses allele frequencies from means) ---
    # Calculate means on the (potentially imputed) 0,1,2 coded matrix
    marker_means_for_p = np.mean(processed_genotypes, axis=0)
    allele_freqs_p = calculate_allele_frequencies_from_means(marker_means_for_p, ploidy=2)

    if center_data and method != "GBLUP": # GBLUP handles centering differently if it builds GRM
        print("  Centering genotype data.")
        center_genotype_matrix_inplace(processed_genotypes) # Modifies in place

    is_centered_final = center_data if method != "GBLUP" else False # GRM is not "centered" in this sense

    # --- Quality Control Part 2 (MAF, Fixed Loci) - after initial p calculation ---
    if quality_control:
        # MAF filter (p or 1-p should be > maf_threshold)
        maf_filter = (allele_freqs_p > maf_threshold) & (allele_freqs_p < (1.0 - maf_threshold))

        # Fixed loci filter (variance of column != 0)
        # Use variance of original (0,1,2 potentially imputed) data before centering for this.
        # Or, use allele_freqs_p: fixed if p=0 or p=1.
        # Variance is zero if p=0 or p=1. So maf_filter already covers this.
        # var_filter = np.var(raw_genotypes_matrix, axis=0) > 1e-8 # Check on raw (or imputed)

        valid_marker_mask = maf_filter # & var_filter (if var_filter done separately)

        if not np.all(valid_marker_mask):
            n_removed = np.sum(~valid_marker_mask)
            processed_genotypes = processed_genotypes[:, valid_marker_mask]
            final_marker_ids = [mid for i, mid in enumerate(final_marker_ids) if valid_marker_mask[i]]
            allele_freqs_p = allele_freqs_p[valid_marker_mask] # Keep only p for selected markers
            print(f"  Removed {n_removed} markers due to MAF < {maf_threshold} or being fixed.")
        else:
            print(f"  All markers passed MAF filter (MAF > {maf_threshold}).")

        if processed_genotypes.shape[1] == 0:
            raise ValueError("No markers left after quality control. Check MAF threshold or input data.")

    # --- GBLUP GRM Calculation (if method is GBLUP and input is not already a GRM) ---
    is_grm_final = False
    if method == "GBLUP":
        # Check if input 'source' was already a GRM (symmetric square matrix)
        # This check should ideally happen on raw_genotypes_matrix before any processing
        if raw_genotypes_matrix.ndim == 2 and raw_genotypes_matrix.shape[0] == raw_genotypes_matrix.shape[1] and \
           np.allclose(raw_genotypes_matrix, raw_genotypes_matrix.T):
            print("  Input matrix detected as symmetric; assuming it's a GRM for GBLUP.")
            processed_genotypes = raw_genotypes_matrix # Use raw input as GRM
            final_marker_ids = list(current_obs_ids) # For GRM, "markers" are individuals
            is_grm_final = True
            is_centered_final = False # Centering not applicable if GRM is provided
            allele_freqs_p = None # Not applicable for GRM
        else:
            print("  Calculating GRM for GBLUP...")
            # Use already QC'd and centered (if center_data=True initially) `processed_genotypes`
            # The centering for GRM calc is typically done: Z = M - 2P
            # VanRaden Method 1: G = ZZ' / (2 * sum(p_i * (1-p_i)))
            # JWAS: genotypes ./ sqrt.(2*p.*(1 .- p)), then (genotypes*genotypes'+ I*eps)/nMarkers
            # This implies Z_standardized = (M_centered) / sqrt(2*p*(1-p)) per marker
            # Then G = Z_std @ Z_std.T / n_markers_effective

            if not is_centered_final: # If data wasn't centered before (e.g. center_data=False)
                # GBLUP GRM calculation needs centered genotypes (M - 2P)
                # marker_means_for_2P = np.mean(raw_genotypes_matrix[:, valid_marker_mask], axis=0) if quality_control else np.mean(raw_genotypes_matrix, axis=0)
                # Z_centered_for_grm = raw_genotypes_matrix[:, valid_marker_mask if quality_control else slice(None)] - marker_means_for_2P
                # This centering is different from just subtracting column mean if not 0,1,2 coded.
                # Assuming 0,1,2 coding, mean is 2p. M - 2P means using `processed_genotypes` if it was centered.
                # If `processed_genotypes` was not centered by user, center it now.
                # Let's use `allele_freqs_p` (from selected markers).
                # Z = M - 2P (where M is 0,1,2 coded markers, P is matrix of 2*p_j)
                # This requires original 0,1,2 coded markers, not the centered ones if `center_data=True` was used.
                # This part is tricky. Let's assume `processed_genotypes` before this step is M (original or imputed).
                # If `center_data` was true, `processed_genotypes` is M_centered.
                # It's better to use the `allele_freqs_p` for standardization.

                # Re-center if `center_data` was false, using 2p.
                # This requires original non-centered QC'd matrix.
                # For simplicity, if GBLUP and not GRM, assume we need to work from a raw (but QC'd) matrix.
                # This needs a refactor: QC -> then optionally center for Bayes methods OR build GRM for GBLUP.

                # Simplified: Assume `processed_genotypes` is M_centered (if center_data=True)
                # or M_original_qc (if center_data=False).
                # The formula `genotypes ./ sqrt.(2*p.*(1 .- p))` implies M_centered is divided by sqrt(var_j)
                # where var_j = 2*p_j*(1-p_j).

                if not center_data: # if data was not centered, do it now for GRM calc based on 2p
                    # This assumes processed_genotypes are 0,1,2.
                    # Allele freqs `p` are based on these.
                    mat_2p = 2 * allele_freqs_p
                    Z_centered_for_grm = processed_genotypes - mat_2p
                else: # data is already M - mean(M). If mean(M) = 2p, this is fine.
                    Z_centered_for_grm = processed_genotypes

                denom_std = np.sqrt(2 * allele_freqs_p * (1 - allele_freqs_p))
                denom_std[denom_std == 0] = 1e-8 # Avoid division by zero for fixed loci (should be removed by QC)

                Z_std = Z_centered_for_grm / denom_std

                grm = (Z_std @ Z_std.T) / Z_std.shape[1] # Divide by number of markers

                # Ensure positive definite (as in Julia)
                # Add small identity if not PD. Loop removed for brevity, direct check.
                if not np.all(np.linalg.eigvals(grm) > 1e-8): # Check with tolerance
                    print("  GRM not positive definite. Adding small value to diagonal.")
                    grm += np.eye(grm.shape[0]) * 1e-5
                    # Re-check, could loop a few times.
                    if not np.all(np.linalg.eigvals(grm) > 1e-8):
                         print("Warning: GRM still not robustly positive definite after small diag addition.")

                processed_genotypes = grm
                final_marker_ids = list(current_obs_ids) # For GRM, "markers" are individuals
                is_grm_final = True
                is_centered_final = False # GRM is not "centered" in the same way
                allele_freqs_p = None


    # --- Create GenotypesData object ---
    geno_obj = GenotypesData(
        name=genotype_data_name if genotype_data_name else "geno1",
        obs_ids=current_obs_ids,
        marker_ids=final_marker_ids,
        genotype_matrix=processed_genotypes, # This is the final matrix (centered, QC'd, or GRM)
        allele_freqs=allele_freqs_p if not is_grm_final else None,
        is_centered=is_centered_final,
        is_grm=is_grm_final,
        method=method,
        pi_value=Pi, # Will be processed further in MME/MCMC setup for multi-trait
        estimate_pi=estimate_pi
    )

    # Setup variance components for this genotype set
    # G_prior_value is for total genomic variance (sigma_a^2) or marker effect variance (sigma_g^2)
    # This needs to be mapped to geno_obj.genetic_variance (for sigma_a^2)
    # and geno_obj.marker_effect_variance (for sigma_g^2)

    # JWAS logic:
    # genotypes.G (marker_effect_variance) = Variance(G_is_marker_variance ? G : false, df, ...)
    # genotypes.genetic_variance = Variance(G_is_marker_variance ? false : G, df, ...)

    prior_val_for_marker_var = G_prior_value if G_is_marker_variance else None
    prior_val_for_genetic_var = G_prior_value if not G_is_marker_variance else None

    # Note: Julia code adjusted G_prior_df for multi-trait.
    # `genotypei.G.df = genotypei.G.df + mme.nModels` (in add_genotypes)
    # `mme.M[1].G.df = νG0` (in add_genotypes)
    # This implies number of traits might affect prior df. For now, use G_prior_df directly.
    # This should be handled when associating GenotypesData with an MME object.

    geno_obj.marker_effect_variance = VarianceCovariance(
        value=prior_val_for_marker_var, # Initial value (often None if estimated)
        df=G_prior_df,
        scale=prior_val_for_marker_var, # Assuming G_prior_value is also the prior scale S0^2 or Psi0
        estimate_variance=G_estimate_variance,
        estimate_scale=G_estimate_scale,
        constraint=G_constraint
    )
    geno_obj.genetic_variance = VarianceCovariance(
        value=prior_val_for_genetic_var,
        df=G_prior_df,
        scale=prior_val_for_genetic_var,
        estimate_variance=G_estimate_variance, # Typically true if G_prior_value is for total genetic var
        estimate_scale=G_estimate_scale,
        constraint=G_constraint
    )

    # Output IDs file
    try:
        ids_filepath = "IDs_for_individuals_with_genotypes.txt"
        # Check if output folder exists from MCMCInfo if passed, else current dir
        # This function doesn't have MCMCInfo. Output to current dir for now.
        with open(ids_filepath, "w") as f:
            for obs_id in current_obs_ids:
                f.write(f"{obs_id}\n")
        print(f"  Genotyped individual IDs written to {ids_filepath}")
    except IOError:
        print(f"Warning: Could not write genotyped IDs file to {ids_filepath}")

    print(f"Genotype processing complete for '{geno_obj.name}'.")
    print(f"  Final shape: {processed_genotypes.shape[0]} individuals, {processed_genotypes.shape[1]} {'individuals (GRM)' if is_grm_final else 'markers'}.")

    return geno_obj


if __name__ == '__main__':
    print("--- Testing Genotype Reader ---")

    # Create a dummy CSV file
    dummy_geno_content = """ObsID,M1,M2,M3,M4,M5
id1,0,1,2,0,9
id2,1,1,0,2,1
id3,2,0,1,1,0
id4,0,missing,1,2,2
id5,1,2,2,0,1
"""
    dummy_geno_filepath = "dummy_genotypes.csv"
    with open(dummy_geno_filepath, "w") as f:
        f.write(dummy_geno_content)

    print(f"\n--- Test 1: Read from CSV, QC, Center, RR-BLUP setup ---")
    try:
        geno_data1 = get_genotypes_data(
            dummy_geno_filepath,
            method="RR-BLUP",
            G_prior_value=0.5, G_is_marker_variance=False, # G is total genetic var
            file_separator=',', file_has_header=True,
            quality_control=True, maf_threshold=0.05, missing_value_code=9.0,
            center_data=True
        )
        print(f"  {geno_data1.name}: Centered={geno_data1.is_centered}, IsGRM={geno_data1.is_grm}")
        print(f"  Allele freqs (first 5): {geno_data1.allele_freqs[:5] if geno_data1.allele_freqs is not None else 'N/A'}")
        print(f"  Marker Var VC: {geno_data1.marker_effect_variance}") # Should be None if G is total genetic
        print(f"  Genetic Var VC: {geno_data1.genetic_variance}")
        if geno_data1.genotypes is not None:
             print(f"  Processed Genotypes (first 3x3):\n{geno_data1.genotypes[:3,:3]}")

    except Exception as e:
        print(f"Error in Test 1: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n--- Test 2: Read from CSV, GBLUP (build GRM) ---")
    try:
        # Need to use "missing" for missing_value_code if that's what's in file for this test
        geno_data2 = get_genotypes_data(
            dummy_geno_filepath,
            method="GBLUP",
            G_prior_value=0.4, G_is_marker_variance=False, # G is total genetic var
            file_separator=',', file_has_header=True,
            quality_control=True, maf_threshold=0.01,
            missing_value_code=9.0, # Matching the '9' in file
            center_data=True # Centering is part of GRM calc if not done before
        )
        print(f"  {geno_data2.name}: Centered={geno_data2.is_centered}, IsGRM={geno_data2.is_grm}")
        print(f"  Genetic Var VC: {geno_data2.genetic_variance}")
        if geno_data2.genotypes is not None:
            print(f"  Processed GRM (shape {geno_data2.genotypes.shape}):\n{geno_data2.genotypes[:3,:3]}")

    except Exception as e:
        print(f"Error in Test 2: {e}")
        import traceback
        traceback.print_exc()

    # Test with DataFrame input
    print(f"\n--- Test 3: Read from DataFrame ---")
    df_source = pd.read_csv(dummy_geno_filepath, sep=',', na_values=["missing", "9"]) # Read with NA handling
    df_source.rename(columns={'ObsID':'AnimalID'}, inplace=True) # Example different col name
    try:
        geno_data3 = get_genotypes_data(
            df_source,
            method="BayesA",
            obs_ids_col_name_or_index="AnimalID",
            G_prior_value=0.001, G_is_marker_variance=True,
            quality_control=True, maf_threshold=0.05, missing_value_code=np.nan, # Pandas already converted
            center_data=True
        )
        print(f"  {geno_data3.name}: Centered={geno_data3.is_centered}, IsGRM={geno_data3.is_grm}")
        print(f"  Marker Var VC: {geno_data3.marker_effect_variance}")
        if geno_data3.genotypes is not None:
             print(f"  Processed Genotypes (first 3x3):\n{geno_data3.genotypes[:3,:3]}")
    except Exception as e:
        print(f"Error in Test 3: {e}")
        import traceback
        traceback.print_exc()

    # Test with NumPy array input
    print(f"\n--- Test 4: Read from NumPy array ---")
    n_obs_np, n_mkrs_np = 50, 100
    np_array_source = np.random.choice([0,1,2,9], size=(n_obs_np, n_mkrs_np), p=[0.4,0.2,0.3,0.1])
    np_obs_ids = [f"S{i}" for i in range(n_obs_np)]
    np_mrk_ids = [f"SNP{j}" for j in range(n_mkrs_np)]
    try:
        geno_data4 = get_genotypes_data(
            np_array_source,
            obs_ids_list=np_obs_ids,
            marker_ids_list=np_mrk_ids,
            method="BayesB",
            G_prior_value=0.002, G_is_marker_variance=True, Pi=0.05,
            quality_control=True, maf_threshold=0.01, missing_value_code=9.0,
            center_data=False # Test without centering by default
        )
        print(f"  {geno_data4.name}: Centered={geno_data4.is_centered}, IsGRM={geno_data4.is_grm}")
        print(f"  Marker Var VC: {geno_data4.marker_effect_variance}")
        print(f"  Pi: {geno_data4.pi_value}")
    except Exception as e:
        print(f"Error in Test 4: {e}")
        import traceback
        traceback.print_exc()


    # Clean up dummy file
    if os.path.exists(dummy_geno_filepath):
        os.remove(dummy_geno_filepath)

```
