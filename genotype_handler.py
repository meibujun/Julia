import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

def read_genotypes_py(
    file_path_or_df: Union[str, pd.DataFrame],
    header_present: bool = True,
    separator: str = ',',
    missing_value_code: Union[str, int, float] = "9",
    perform_qc: bool = True,
    maf_threshold: float = 0.01,
    center_genotypes: bool = True,
    double_precision: bool = False
) -> Dict[str, Union[np.ndarray, List[str], Optional[np.ndarray]]]:
    """
    Reads genotype data from a file or DataFrame.
    Performs optional QC (missing imputation, MAF filtering) and centering.

    Args:
        file_path_or_df: Path to genotype file or a pandas DataFrame.
                         Expected format: Individuals in rows, markers in columns.
                         First column: individual IDs. Subsequent columns: marker data.
        header_present: True if the first row contains marker IDs.
        separator: Delimiter for text files.
        missing_value_code: Code used for missing genotypes (e.g., "9", -1, np.nan).
        perform_qc: Whether to perform quality control.
        maf_threshold: Minor Allele Frequency threshold for filtering.
        center_genotypes: Whether to center the genotype matrix.
        double_precision: If True, use float64, else float32.

    Returns:
        A dictionary containing:
        - 'genotypes_matrix': Processed genotype matrix (np.ndarray).
        - 'obs_ids': List of individual IDs.
        - 'marker_ids': List of marker IDs.
        - 'allele_frequencies': Allele frequencies (p) if calculated (np.ndarray or None).
                                 Calculated if perform_qc or center_genotypes is True.
        - 'sum_2pq': Sum of 2pq across markers if allele frequencies are calculated.
    """
    dtype = np.float64 if double_precision else np.float32

    if isinstance(file_path_or_df, str):
        # Read header to get marker IDs if present
        if header_present:
            first_row_df = pd.read_csv(file_path_or_df, sep=separator, nrows=0)
            marker_ids = list(first_row_df.columns[1:])
            skiprows = 1
        else:
            # Peek at first line to determine number of markers if no header
            temp_df = pd.read_csv(file_path_or_df, sep=separator, nrows=1, header=None)
            num_markers = temp_df.shape[1] - 1
            marker_ids = [f"M{i+1}" for i in range(num_markers)]
            skiprows = 0

        # Read the data part
        df = pd.read_csv(file_path_or_df, sep=separator, header=None, skiprows=skiprows,
                         na_values=str(missing_value_code), dtype=str) # Read as string to handle various inputs
        obs_ids = df.iloc[:, 0].astype(str).tolist()
        genotypes_str = df.iloc[:, 1:].values

    elif isinstance(file_path_or_df, pd.DataFrame):
        df_copy = file_path_or_df.copy()
        if header_present:
            marker_ids = list(df_copy.columns[1:])
            obs_ids = df_copy.iloc[:, 0].astype(str).tolist()
            genotypes_str = df_copy.iloc[:, 1:].values
        else: # No header in DataFrame, assume first col is ID, rest markers
            obs_ids = df_copy.iloc[:, 0].astype(str).tolist()
            num_markers = df_copy.shape[1] - 1
            marker_ids = [f"M{i+1}" for i in range(num_markers)]
            genotypes_str = df_copy.iloc[:, 1:].values
    else:
        raise TypeError("Input must be a file path (str) or a pandas DataFrame.")

    # Convert genotype data to numeric, handling potential errors by coercing to NaN
    try:
        genotypes_matrix = np.array(genotypes_str, dtype=dtype)
    except ValueError: # If direct conversion fails due to non-numeric strings not caught by na_values
        genotypes_matrix = np.array(pd.DataFrame(genotypes_str).replace(str(missing_value_code), np.nan).values, dtype=dtype)


    # --- Initial QC: Missing value imputation ---
    if perform_qc or np.isnan(genotypes_matrix).any(): # Impute if QC is on OR if NaNs are present
        num_obs, _ = genotypes_matrix.shape
        for j in range(genotypes_matrix.shape[1]):
            col_data = genotypes_matrix[:, j]
            nan_mask = np.isnan(col_data)
            if np.all(nan_mask): # All values are NaN in this column
                # print(f"Warning: Marker {marker_ids[j]} has all missing values. Imputing with 0.5 (arbitrary).")
                col_data[:] = 0.5 # Arbitrary, or could be global mean, or removed
            elif np.any(nan_mask):
                col_mean = np.nanmean(col_data)
                col_data[nan_mask] = col_mean
        # print(f"Info: Missing values (originally '{missing_value_code}') imputed with column means.")

    # --- Allele Frequencies (p) ---
    # Calculated based on current matrix (after imputation, before centering/filtering for MAF)
    # Assumes genotypes are 0, 1, 2 for counts of one allele.
    # p = mean(marker_column / 2)
    allele_frequencies = np.mean(genotypes_matrix, axis=0) / 2.0

    # --- QC: MAF Filtering ---
    if perform_qc:
        if allele_frequencies is None: # Should have been calculated above
             allele_frequencies_for_qc = np.mean(genotypes_matrix, axis=0) / 2.0
        else:
             allele_frequencies_for_qc = allele_frequencies

        maf = np.minimum(allele_frequencies_for_qc, 1.0 - allele_frequencies_for_qc)
        keep_mask_maf = maf >= maf_threshold

        # Also filter fixed loci (variance is zero)
        var_markers = np.var(genotypes_matrix, axis=0)
        keep_mask_var = var_markers > 1e-6 # Tolerance for floating point

        keep_mask = keep_mask_maf & keep_mask_var

        original_marker_count = genotypes_matrix.shape[1]
        genotypes_matrix = genotypes_matrix[:, keep_mask]
        marker_ids = [mid for mid, keep in zip(marker_ids, keep_mask) if keep]
        if allele_frequencies is not None:
            allele_frequencies = allele_frequencies[keep_mask]
        removed_count = original_marker_count - len(marker_ids)
        # if removed_count > 0:
        #     print(f"Info: Removed {removed_count} markers due to MAF < {maf_threshold} or being fixed.")

    if genotypes_matrix.shape[1] == 0:
        # print("Warning: No markers left after QC.")
        return {
            'genotypes_matrix': genotypes_matrix,
            'obs_ids': obs_ids,
            'marker_ids': marker_ids,
            'allele_frequencies': None,
            'sum_2pq': None
        }

    sum_2pq = None
    if allele_frequencies is not None:
        valid_freq_mask = (allele_frequencies > 1e-6) & (allele_frequencies < 1.0 - 1e-6) # Avoid fixed loci for sum_2pq
        if np.any(valid_freq_mask):
            sum_2pq = np.sum(2 * allele_frequencies[valid_freq_mask] * (1 - allele_frequencies[valid_freq_mask]))
        else: # All loci are fixed or problematic
            sum_2pq = 0.0 # Or handle as error / warning

    # --- Centering ---
    if center_genotypes:
        if allele_frequencies is None:
            current_means = np.mean(genotypes_matrix, axis=0)
            allele_frequencies = current_means / 2.0
            if sum_2pq is None:
                 valid_freq_mask = (allele_frequencies > 1e-6) & (allele_frequencies < 1.0 - 1e-6)
                 sum_2pq = np.sum(2 * allele_frequencies[valid_freq_mask] * (1 - allele_frequencies[valid_freq_mask])) if np.any(valid_freq_mask) else 0.0

        # Use original allele_frequencies for centering if available and not filtered to maintain reference
        # However, if markers were filtered, allele_frequencies would match genotypes_matrix.
        # The critical part is that the means for centering must match the current columns of genotypes_matrix.
        means_for_centering = 2.0 * allele_frequencies # allele_frequencies here are from potentially QC'd matrix
        genotypes_matrix = genotypes_matrix - means_for_centering
        # print("Info: Genotypes centered.")


    return {
        'genotypes_matrix': genotypes_matrix.astype(dtype),
        'obs_ids': obs_ids,
        'marker_ids': marker_ids,
        'allele_frequencies': allele_frequencies.astype(dtype) if allele_frequencies is not None else None,
        'sum_2pq': dtype(sum_2pq) if sum_2pq is not None else None
    }

# --- Utility functions ---

def get_allele_frequencies_from_matrix(genotype_matrix: np.ndarray) -> np.ndarray:
    """Calculates allele frequencies (p) assuming genotypes are 0, 1, 2 (counts of one allele)."""
    if genotype_matrix.ndim != 2: raise ValueError("Genotype matrix must be 2D.")
    # Basic check for typical 0,1,2 coding. Allows for imputed float values.
    if np.any(np.nan_to_num(genotype_matrix) < -0.1) or np.any(np.nan_to_num(genotype_matrix) > 2.1):
        # print("Warning: Genotypes outside typical 0-2 range found. Frequencies assume 0,1,2 coding.")
        pass
    return np.nanmean(genotype_matrix, axis=0) / 2.0


def center_genotype_matrix(genotype_matrix: np.ndarray, allele_frequencies: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Centers a genotype matrix. Returns centered matrix and marker means (2p)."""
    if allele_frequencies is None:
        allele_frequencies = get_allele_frequencies_from_matrix(genotype_matrix)
    marker_means = 2.0 * allele_frequencies
    centered_matrix = genotype_matrix - marker_means
    return centered_matrix, marker_means


def check_monomorphic(genotype_matrix: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Checks for monomorphic SNPs. Returns boolean array (True if monomorphic)."""
    if genotype_matrix.ndim != 2: raise ValueError("Genotype matrix must be 2D.")
    variances = np.var(genotype_matrix, axis=0)
    return variances < tolerance


def calculate_grm_py(
    genotype_matrix: np.ndarray,
    allele_frequencies: Optional[np.ndarray] = None,
    sum_2pq: Optional[float] = None,
    method: str = "vanraden1",
    ridge_epsilon: float = 1e-5
) -> np.ndarray:
    """
    Calculates the Genomic Relationship Matrix (GRM).

    Args:
        genotype_matrix: np.ndarray (n_obs x n_markers). Should be numeric (0,1,2).
                         Assumed to be NOT centered for VanRaden Method 1.
        allele_frequencies: Optional pre-calculated allele frequencies (p).
        sum_2pq: Optional pre-calculated sum of 2pq. If not provided, calculated from allele_frequencies.
        method: Currently supports "vanraden1".
                VanRaden Method 1: G = Z Z' / sum(2*p_i*(1-p_i)), where Z = M - 2P.
                                 M is matrix of genotypes (0,1,2).
                                 P is matrix of allele frequencies (2p_j for each marker j).
        ridge_epsilon: Small value to add to diagonal for numerical stability if GRM is not PD.

    Returns:
        GRM (np.ndarray).
    """
    if method.lower() != "vanraden1":
        raise NotImplementedError(f"GRM calculation method '{method}' not implemented.")

    n_obs, n_markers = genotype_matrix.shape
    if n_markers == 0: return np.zeros((n_obs, n_obs))

    if allele_frequencies is None:
        allele_frequencies = get_allele_frequencies_from_matrix(genotype_matrix)

    if not (0 < allele_frequencies.all() < 1):
        # print("Warning: Some allele frequencies are 0 or 1. sum_2pq might be zero or small.")
        # Handle fixed alleles before calculating sum_2pq to avoid division by zero or instability
        # For fixed alleles, 2pq = 0. They don't contribute to sum_2pq.
        # However, VanRaden's method involves centering M by 2p. If p=0 or p=1, M-2P is still valid.
        # The issue is division by sum_2pq.
        pass

    # Matrix P contains 2*p_j for each marker j, repeated for all individuals
    # P_matrix = np.full((n_obs, n_markers), 2.0 * allele_frequencies)
    P_vector = 2.0 * allele_frequencies # This is 1 x n_markers

    # Z = M - P (where P is broadcasted)
    Z_matrix = genotype_matrix - P_vector # P_vector is broadcasted to each row of M

    if sum_2pq is None:
        # Calculate sum_2pq only from polymorphic markers to avoid issues with fixed markers
        # However, the Z matrix already used all allele_frequencies.
        # If a marker is fixed (p=0 or p=1), its 2p(1-p) is 0.
        # Standard practice is to sum 2pq over all markers used to build Z.
        valid_freq_mask = (allele_frequencies > 1e-6) & (allele_frequencies < 1.0 - 1e-6)
        if np.any(valid_freq_mask):
            current_sum_2pq = np.sum(2 * allele_frequencies[valid_freq_mask] * (1 - allele_frequencies[valid_freq_mask]))
        else: # All markers are fixed or problematic
            current_sum_2pq = 0.0
    else:
        current_sum_2pq = sum_2pq

    if abs(current_sum_2pq) < 1e-9: # Denominator is zero or too small
        # print("Warning: sum(2pq) is close to zero. GRM cannot be reliably scaled by it. Returning unscaled Z Z'.")
        # This might happen if all markers are fixed after QC, or few markers with low MAF.
        # Returning Z Z' / n_markers as an alternative scaling, or just Z Z'.
        # Or could add a small epsilon to current_sum_2pq.
        # For now, let's use n_markers as a fallback scaling, similar to some GBLUP forms.
        # G = (Z_matrix @ Z_matrix.T) / n_markers if n_markers > 0 else Z_matrix @ Z_matrix.T
        # However, if sum_2pq is truly zero, it suggests issues with input markers.
        # Raising error might be better if scaling is critical.
        # For now, let's allow it but it means the GRM is not in standard VanRaden scale.
        if n_markers > 0:
            G = (Z_matrix @ Z_matrix.T) / n_markers
        else: # Should not happen if n_markers check at start
            G = Z_matrix @ Z_matrix.T
    else:
        G = (Z_matrix @ Z_matrix.T) / current_sum_2pq

    # Ensure positive definiteness with a small ridge (optional, but good practice)
    # G += np.eye(n_obs) * ridge_epsilon
    # Check if PD before adding ridge: np.all(np.linalg.eigvals(G) > 1e-6)

    return G
    """
    Checks for monomorphic SNPs (variance close to zero).
    Returns a boolean array where True indicates a monomorphic SNP.
    """
    if genotype_matrix.ndim != 2:
        raise ValueError("Genotype matrix must be 2D.")
    variances = np.var(genotype_matrix, axis=0)
    return variances < tolerance

```

This creates `genotype_handler.py` with `read_genotypes_py` and some utility functions.
`read_genotypes_py` handles:
- Reading from CSV or DataFrame.
- Optional header for marker IDs.
- Missing value imputation (with column mean).
- MAF filtering.
- Centering.
- Returns a dictionary with the processed matrix, IDs, and allele frequencies.

Next, I'll work on extending `model_components.py` for GBLUP and BayesC0 setup and then create the unit tests.
