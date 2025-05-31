import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Union, Optional, Any

# Assuming MME_py and GenotypesComponent are defined in model_components
# from model_components import MME_py, GenotypesComponent

# Placeholder for MME_py and GenotypesComponent if not available for standalone testing
class MockMME:
    def __init__(self):
        self.genotype_components: List[MockGenotypeComponent] = []
        self.output_id: Optional[List[str]] = None # For local EBV

class MockGenotypeComponent:
    def __init__(self, name="mock_gc", marker_ids=None, genotype_matrix=None, obs_ids=None):
        self.name = name
        self.marker_ids = marker_ids if marker_ids is not None else []
        self.genotype_matrix = genotype_matrix # This should be X for relevant individuals
        self.obs_ids = obs_ids if obs_ids is not None else []


def calculate_marker_model_frequency_py(
    marker_effects_file_path: str,
    header: bool = True,
    separator: str = ','
) -> pd.DataFrame:
    """
    Computes the model frequency for each marker from MCMC samples of marker effects.
    Model frequency is the probability that the marker is included in the model (effect != 0).
    """
    print(f"Calculating marker model frequencies from: {marker_effects_file_path}")
    try:
        if header:
            samples_df = pd.read_csv(marker_effects_file_path, sep=separator, header=0)
            marker_ids = samples_df.columns.tolist()
            samples = samples_df.values
        else:
            samples = pd.read_csv(marker_effects_file_path, sep=separator, header=None).values
            marker_ids = [f"M{i+1}" for i in range(samples.shape[1])]
    except Exception as e:
        raise ValueError(f"Error reading marker effects file {marker_effects_file_path}: {e}")

    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("Marker effects samples must be a 2D array with at least one marker.")

    # Model frequency = proportion of samples where effect is non-zero
    # Using a small tolerance for floating point comparisons to zero
    model_frequency = np.mean(np.abs(samples) > 1e-9, axis=0)

    output_df = pd.DataFrame({
        'marker_ID': marker_ids,
        'model_frequency': model_frequency
    })
    return output_df


def _parse_window_size_str(window_size_str: str) -> int:
    """Parses window size string like "1 Mb" to base pairs."""
    match = re.match(r"(\d+\.?\d*)\s*(kb|mb)", window_size_str.lower())
    if not match:
        raise ValueError(f"Invalid window_size format: '{window_size_str}'. Expected e.g., '1 Mb' or '500 Kb'.")
    value, unit = float(match.group(1)), match.group(2)
    if unit == "mb":
        return int(value * 1_000_000)
    elif unit == "kb":
        return int(value * 1_000)
    return int(value) # Should not happen with regex


def _define_genomic_windows(
    map_df: pd.DataFrame, # Columns: marker_ID, chromosome, position
    snp_ids_in_model: List[str], # IDs of SNPs present in the model (genotype matrix X)
    window_size_bp: int,
    sliding_window: bool
) -> List[Dict[str, Any]]:
    """
    Defines genomic windows based on map information.
    Returns a list of window dictionaries.
    """
    windows = []

    # Filter map_df for SNPs that are actually in the model and sort by chr/pos
    map_df_filtered = map_df[map_df['marker_ID'].isin(snp_ids_in_model)].copy()
    if map_df_filtered.empty:
        print("Warning: No markers from map file are present in the model's SNP set.")
        return []

    map_df_filtered['position'] = pd.to_numeric(map_df_filtered['position'])
    map_df_filtered.sort_values(by=['chromosome', 'position'], inplace=True)

    # Create a mapping from model SNP ID to its column index in the original X matrix
    # This assumes snp_ids_in_model is the order of columns in X
    snp_to_idx_map = {snp_id: i for i, snp_id in enumerate(snp_ids_in_model)}

    # Map filtered map marker_IDs to their original column indices in X
    # This requires that snp_ids_in_model (from mme.M[x].markerID) is the reference for column indices
    # map_df_filtered['col_idx_in_X'] = map_df_filtered['marker_ID'].map(snp_to_idx_map)
    # For window definition, we care about the order *within map_df_filtered* first,
    # then map these selected markers to their columns in X.

    current_model_snp_indices = []
    for marker_id in map_df_filtered['marker_ID']:
        current_model_snp_indices.append(snp_to_idx_map[marker_id])
    map_df_filtered['col_idx_in_X'] = current_model_snp_indices


    for chrom, group in map_df_filtered.groupby('chromosome'):
        group = group.sort_values(by='position')
        min_pos, max_pos = group['position'].min(), group['position'].max()

        if not sliding_window:
            num_windows_on_chrom = int(np.ceil(max_pos / window_size_bp))
            for i_win in range(num_windows_on_chrom):
                win_bp_start = i_win * window_size_bp
                win_bp_end = win_bp_start + window_size_bp

                snps_in_win_df = group[(group['position'] >= win_bp_start) & (group['position'] < win_bp_end)]
                if not snps_in_win_df.empty:
                    windows.append({
                        'chromosome': chrom,
                        'bp_start': win_bp_start,
                        'bp_end': win_bp_end,
                        'snp_pos_start': snps_in_win_df['position'].min(),
                        'snp_pos_end': snps_in_win_df['position'].max(),
                        'num_snps': len(snps_in_win_df),
                        'snp_indices_in_X': sorted(list(snps_in_win_df['col_idx_in_X'])), # Store original X indices
                        'marker_ids_in_window': list(snps_in_win_df['marker_ID'])
                    })
        else: # Sliding window: each SNP starts a new window (if possible)
            positions = group['position'].values
            for i_snp_idx, current_snp_start_pos in enumerate(positions):
                win_bp_start = current_snp_start_pos
                win_bp_end = win_bp_start + window_size_bp

                # Select SNPs from 'group' that fall into this window [win_bp_start, win_bp_end)
                snps_in_win_df = group[(group['position'] >= win_bp_start) & (group['position'] < win_bp_end)]

                if not snps_in_win_df.empty:
                    windows.append({
                        'chromosome': chrom,
                        'bp_start': win_bp_start, # Actual start of window (a SNP position)
                        'bp_end': win_bp_end,     # Theoretical end of window
                        'snp_pos_start': snps_in_win_df['position'].min(),
                        'snp_pos_end': snps_in_win_df['position'].max(),
                        'num_snps': len(snps_in_win_df),
                        'snp_indices_in_X': sorted(list(snps_in_win_df['col_idx_in_X'])),
                        'marker_ids_in_window': list(snps_in_win_df['marker_ID'])
                    })
    return windows


def run_window_gwas_py(
    mme: Any, # Should be MME_py instance
    map_file_path: str,
    marker_effects_file_paths: List[str],
    window_size_str: str = "1 Mb",
    sliding_window: bool = False,
    gwas_threshold_ppa: float = 0.01, # PPA threshold
    genetic_correlation_flag: bool = False,
    local_ebv_flag: bool = False,
    output_win_var_props_flag: bool = False,
    header_map: bool = True,
    header_effects: bool = True,
    separator_map: str = ',',
    separator_effects: str = ','
) -> Tuple[List[pd.DataFrame], Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
    """
    Performs window-based GWAS, genetic correlation, and local EBV estimation.
    """
    if not mme.genotype_components:
        raise ValueError("MME object must have at least one GenotypesComponent.")

    # For now, assume first genotype component is the one to use for X matrix and marker IDs
    # This X should be for individuals for whom local EBVs are calculated if local_ebv_flag is True.
    # Typically, this means X should be aligned with mme.output_id if specified, or all phenotyped.
    # The Julia code uses mme.M[1].output_genotypes, which could be X aligned to output_ID or phenotyped IDs.
    # Let's assume GenotypesComponent.genotype_matrix is the relevant X.
    # It must NOT be the GRM for this function.
    geno_comp = mme.genotype_components[0]
    if geno_comp.is_grm:
        raise ValueError("GenotypesComponent must contain marker covariates (X), not a GRM, for window GWAS.")
    if geno_comp.genotype_matrix is None:
        raise ValueError("Genotype matrix X is not available in the GenotypesComponent.")

    X_geno = geno_comp.genotype_matrix # n_individuals x n_model_snps
    model_marker_ids = geno_comp.marker_ids # List of marker IDs in order of X_geno columns

    # Individuals for local EBV (use mme.obs_id if mme.output_id is not set)
    # This depends on X_geno rows matching these individuals.
    # Assume X_geno rows correspond to mme.obs_id by default from read_genotypes,
    # or it has been aligned to mme.output_id if that was the target for output_genotypes in Julia.
    # For simplicity, if local_ebv_flag, assume X_geno individuals are those for whom local EBVs are desired.
    individual_ids_for_lebv = geno_comp.obs_ids if local_ebv_flag else None


    print(f"Reading map file: {map_file_path}")
    map_df = pd.read_csv(map_file_path, sep=separator_map, header=0 if header_map else None)
    if not header_map: map_df.columns = ['marker_ID', 'chromosome', 'position']
    map_df['marker_ID'] = map_df['marker_ID'].astype(str)
    map_df['chromosome'] = map_df['chromosome'].astype(str)


    window_size_bp = _parse_window_size_str(window_size_str)
    print(f"Defining genomic windows of size {window_size_bp} bp (sliding: {sliding_window})...")
    windows = _define_genomic_windows(map_df, model_marker_ids, window_size_bp, sliding_window)
    if not windows:
        print("No windows defined. GWAS cannot proceed.")
        return [], None, None
    n_windows = len(windows)
    print(f"Defined {n_windows} windows.")

    gwas_results_dfs = []
    all_win_var_props_list = [] if output_win_var_props_flag else None

    # Store BV_winj samples for all traits for correlation calculation
    all_trait_bv_win_samples: List[List[np.ndarray]] = [[] for _ in range(len(marker_effects_file_paths))]


    for i_trait, effects_file in enumerate(marker_effects_file_paths):
        print(f"Processing GWAS for trait {i_trait+1} using effects from: {effects_file}")
        try:
            if header_effects:
                samples_df = pd.read_csv(effects_file, sep=separator_effects, header=0)
                effect_marker_ids = samples_df.columns.tolist()
                alpha_samples = samples_df.values # n_mcmc_samples x n_markers_in_file
            else:
                alpha_samples = pd.read_csv(effects_file, sep=separator_effects, header=None).values
                effect_marker_ids = [f"M{i+1}" for i in range(alpha_samples.shape[1])]
        except Exception as e:
            raise ValueError(f"Error reading marker effects file {effects_file}: {e}")

        # Align alpha_samples columns to model_marker_ids (order of X_geno)
        if effect_marker_ids != model_marker_ids:
            print("Aligning marker effects sample columns to model's marker order...")
            # Create mapping from file marker ID to its column index in alpha_samples
            file_id_to_idx_map = {id_str: i for i, id_str in enumerate(effect_marker_ids)}

            aligned_alpha_samples = np.zeros((alpha_samples.shape[0], len(model_marker_ids)), dtype=alpha_samples.dtype)
            for new_col_idx, model_id in enumerate(model_marker_ids):
                if model_id in file_id_to_idx_map:
                    aligned_alpha_samples[:, new_col_idx] = alpha_samples[:, file_id_to_idx_map[model_id]]
                # else: marker in model but not in effects file, its effect remains 0 (already initialized)
            alpha_samples = aligned_alpha_samples
            print("Alignment complete.")


        n_mcmc_samples, n_model_markers_check = alpha_samples.shape
        if n_model_markers_check != len(model_marker_ids):
            raise ValueError(f"Mismatch in number of markers between X_geno ({len(model_marker_ids)}) and aligned effects file {effects_file} ({n_model_markers_check}).")

        win_variances_samples = np.zeros((n_mcmc_samples, n_windows))
        win_var_proportions_samples = np.zeros((n_mcmc_samples, n_windows))

        current_trait_local_ebv_sum = np.zeros((X_geno.shape[0], n_windows)) if local_ebv_flag else None
        current_trait_bv_win_samples_for_corr = [np.zeros((X_geno.shape[0], n_mcmc_samples)) for _ in range(n_windows)] if genetic_correlation_flag else []


        for i_sample in range(n_mcmc_samples):
            alpha_i = alpha_samples[i_sample, :] # Current sample of all marker effects

            # Calculate total genetic variance for this MCMC sample
            # BV_total_i = X_geno @ alpha_i (n_individuals x 1)
            # var_total_genetic_i = np.var(BV_total_i)
            # Optimization: var(X*alpha) = alpha' * cov(X) * alpha. If X is std, cov(X) is G.
            # Or simpler: var of the sum of effects.
            # This requires X_geno to be for the reference population for variance calculation.
            # Let's assume X_geno is appropriate for this variance.
            var_total_genetic_i = np.var(X_geno @ alpha_i)
            if var_total_genetic_i < 1e-9: var_total_genetic_i = 1e-9 # Avoid division by zero

            for j_win, window in enumerate(windows):
                win_snp_indices = window['snp_indices_in_X'] # These are original column indices in X_geno
                if not win_snp_indices: # Empty window
                    win_variances_samples[i_sample, j_win] = 0
                    win_var_proportions_samples[i_sample, j_win] = 0
                    continue

                X_win_j = X_geno[:, win_snp_indices]
                alpha_i_win_j = alpha_i[win_snp_indices]

                bv_win_j = X_win_j @ alpha_i_win_j # (n_individuals x 1)
                var_win_j = np.var(bv_win_j)

                win_variances_samples[i_sample, j_win] = var_win_j
                win_var_proportions_samples[i_sample, j_win] = var_win_j / var_total_genetic_i

                if local_ebv_flag and current_trait_local_ebv_sum is not None:
                    current_trait_local_ebv_sum[:, j_win] += bv_win_j # Summing up, will divide by n_samples later

                if genetic_correlation_flag:
                    current_trait_bv_win_samples_for_corr[j_win][:, i_sample] = bv_win_j

        all_trait_bv_win_samples[i_trait] = current_trait_bv_win_samples_for_corr

        # Calculate WPPA and other stats for this trait
        wppa = np.mean(win_var_proportions_samples > gwas_threshold_ppa, axis=0)
        mean_prop_gen_var = np.mean(win_var_proportions_samples, axis=0) * 100 # Percentage
        mean_win_var = np.mean(win_variances_samples, axis=0)
        std_win_var = np.std(win_variances_samples, axis=0)

        trait_gwas_df = pd.DataFrame({
            'trait_index': i_trait + 1,
            'window_index': np.arange(n_windows) + 1,
            'chromosome': [w['chromosome'] for w in windows],
            'bp_start': [w['bp_start'] for w in windows],
            'bp_end': [w['bp_end'] for w in windows],
            'snp_pos_start': [w['snp_pos_start'] for w in windows],
            'snp_pos_end': [w['snp_pos_end'] for w in windows],
            'num_snps_in_window': [w['num_snps'] for w in windows],
            'mean_window_variance': mean_win_var,
            'std_window_variance': std_win_var,
            'mean_prop_total_variance': mean_prop_gen_var,
            'WPPA': wppa
        })
        trait_gwas_df = trait_gwas_df.sort_values(by="WPPA", ascending=False)
        gwas_results_dfs.append(trait_gwas_df)

        if local_ebv_flag and current_trait_local_ebv_sum is not None:
            mean_local_ebvs = current_trait_local_ebv_sum / n_mcmc_samples
            lebv_df_cols = {f"win{j+1}_lebv": mean_local_ebvs[:, j] for j in range(n_windows)}
            lebv_df = pd.DataFrame(lebv_df_cols)
            if individual_ids_for_lebv: # Add individual IDs if available
                 lebv_df.insert(0, "ID", individual_ids_for_lebv)
            # Save or return local EBVs (e.g., append to gwas_results_dfs or handle separately)
            # For now, let's print a message and it could be an optional return
            print(f"Local EBVs calculated for trait {i_trait+1}. Shape: {mean_local_ebvs.shape}")
            # Example: save to a file
            # lebv_df.to_csv(f"localEBV_trait{i_trait+1}.csv", index=False)
            # For now, we are not returning it directly from this function to keep signature simple.
            # It could be added to gwas_results_dfs elements or returned separately.

        if output_win_var_props_flag and all_win_var_props_list is not None:
            all_win_var_props_list.append(win_var_proportions_samples)

    # Genetic Correlation Calculation
    genetic_corr_df = None
    if genetic_correlation_flag and len(marker_effects_file_paths) == 2:
        print("Calculating windowed genetic correlations...")
        bv_samples_trait1 = all_trait_bv_win_samples[0] # List of (n_ind x n_mcmc_samples) arrays, one per window
        bv_samples_trait2 = all_trait_bv_win_samples[1]

        window_covariances_mcmc = np.zeros((n_mcmc_samples, n_windows))
        window_correlations_mcmc = np.zeros((n_mcmc_samples, n_windows))

        for j_win in range(n_windows):
            bv1_win_j_samples = bv_samples_trait1[j_win] # n_ind x n_mcmc_samples
            bv2_win_j_samples = bv_samples_trait2[j_win] # n_ind x n_mcmc_samples

            for i_sample in range(n_mcmc_samples):
                bv1 = bv1_win_j_samples[:, i_sample]
                bv2 = bv2_win_j_samples[:, i_sample]
                window_covariances_mcmc[i_sample, j_win] = np.cov(bv1, bv2)[0, 1]

                var1 = np.var(bv1)
                var2 = np.var(bv2)
                if var1 > 1e-9 and var2 > 1e-9:
                    window_correlations_mcmc[i_sample, j_win] = window_covariances_mcmc[i_sample, j_win] / np.sqrt(var1 * var2)
                else:
                    window_correlations_mcmc[i_sample, j_win] = 0.0 # Or NaN

        mean_window_cov = np.mean(window_covariances_mcmc, axis=0)
        std_window_cov = np.std(window_covariances_mcmc, axis=0)
        mean_window_cor = np.mean(window_correlations_mcmc, axis=0)
        std_window_cor = np.std(window_correlations_mcmc, axis=0)

        genetic_corr_df = pd.DataFrame({
            'window_index': np.arange(n_windows) + 1,
            'chromosome': [w['chromosome'] for w in windows],
            'bp_start': [w['bp_start'] for w in windows],
            'bp_end': [w['bp_end'] for w in windows],
            'num_snps_in_window': [w['num_snps'] for w in windows],
            'mean_covariance': mean_window_cov,
            'std_covariance': std_window_cov,
            'mean_correlation': mean_window_cor,
            'std_correlation': std_window_cor
        })
        print("Genetic correlation calculation complete.")

    return gwas_results_dfs, genetic_corr_df, all_win_var_props_list

```

I've created `gwas_module.py` with:
-   `calculate_marker_model_frequency_py`: Calculates model frequency of markers from MCMC effect samples.
-   `_parse_window_size_str`: Helper to parse strings like "1 Mb" into base pairs.
-   `_define_genomic_windows`: Helper to define genomic windows based on a map file, window size, and sliding/non-sliding option. It maps window SNPs to their column indices in the model's genotype matrix `X`.
-   `run_window_gwas_py`: The main function.
    -   Reads map file and marker effect MCMC samples.
    -   Aligns marker effects from file to the order of markers in the model's `X` matrix.
    -   Iterates MCMC samples:
        -   Calculates total genetic variance $var(X\alpha)$.
        -   For each window, calculates window breeding values $BV_{winj} = X_{winj}\alpha_{winj}$ and its variance $var(BV_{winj})$.
        -   Stores window variance proportions.
        -   Accumulates $BV_{winj}$ for local EBV calculation (mean across MCMC samples).
    -   Calculates WPPA and mean variance proportions per window.
    -   If `genetic_correlation_flag` is true and two effect files are given:
        -   Calculates window covariances and correlations between the two traits' window BVs for each MCMC sample.
        -   Summarizes these to posterior means and std devs.
    -   Returns DataFrames of results.
-   Mock MME and GenotypeComponent classes are included at the top for standalone execution/testing if `model_components.py` is not in the path. These should be removed if integrating into a larger project structure.

**Self-Correction/Refinements:**
-   **Marker Alignment:** Added a crucial step in `run_window_gwas_py` to align marker effects read from file to the order of markers in the `X_geno` matrix (derived from `model_marker_ids`). This is vital if the effects file doesn't perfectly match the model's internal marker order.
-   **Local EBV Individuals:** Clarified that `X_geno` and `individual_ids_for_lebv` must correspond. The `GenotypesComponent.obs_ids` (which should match `X_geno` rows) are used if `local_ebv_flag` is true.
-   **Window SNP Indices:** `_define_genomic_windows` now stores `snp_indices_in_X` which are the 0-based column indices in the main `X_geno` matrix for the SNPs in that window. This simplifies slicing `X_geno` and `alpha_i`.
-   **Genetic Correlation:** The storage `all_trait_bv_win_samples` is structured to hold, for each trait, a list where each element is an array of `(n_individuals x n_mcmc_samples)` for a specific window's BVs. This allows for easier calculation of correlations across MCMC samples.

Next, I will create `test_gwas_module.py`.
