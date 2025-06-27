import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from typing import List, Optional, Dict, Any, Tuple, Union # Added Dict, Any, Tuple, Union
from ..core.genotypes import GenotypesData # Assuming GenotypesData is in core
from ..core.variance_covariance import VarianceCovariance
# If MME and MCMCInfo are needed for context:
# from ..core.mme import MixedModelEquations
# from ..core.mcmc_info import MCMCInfo


def center_genotype_matrix_inplace(genotype_matrix: np.ndarray) -> Optional[np.ndarray]:
    if genotype_matrix.size == 0: return None
    if genotype_matrix.ndim != 2: raise ValueError("Genotype matrix must be 2D.")
    col_means = np.mean(genotype_matrix, axis=0)
    genotype_matrix -= col_means
    return col_means

def calculate_allele_frequencies_from_means(marker_means: np.ndarray, ploidy: int = 2) -> np.ndarray:
    if ploidy <= 0: raise ValueError("Ploidy must be positive.")
    return marker_means / float(ploidy)

def make_incidence_matrix_for_ids(target_ids: List[str], source_ids: List[str]) -> csc_matrix:
    if not target_ids: return csc_matrix((0, len(source_ids) if source_ids else 0))
    if not source_ids: return csc_matrix((len(target_ids), 0))

    source_id_to_index_map = {id_val: i for i, id_val in enumerate(source_ids)}
    row_indices, col_indices, data_values = [], [], []

    for i, target_id in enumerate(target_ids):
        if target_id in source_id_to_index_map:
            row_indices.append(i)
            col_indices.append(source_id_to_index_map[target_id])
            data_values.append(1.0)
        else:
            raise ValueError(f"Target ID '{target_id}' not found in source IDs.")

    Z_shape = (len(target_ids), len(source_ids))
    if not data_values: return csc_matrix(Z_shape)

    Z_lil = lil_matrix(Z_shape, dtype=np.float64)
    for r, c, v in zip(row_indices, col_indices, data_values): Z_lil[r, c] = v
    return Z_lil.tocsc()

def align_genotypes_with_phenotypes(
    genotypes_data_list: List[GenotypesData],
    phenotype_obs_ids: List[str],
    output_pred_ids: Optional[List[str]] = None
):
    if not phenotype_obs_ids: print("Warning: phenotype_obs_ids is empty."); return

    for geno_data in genotypes_data_list:
        if geno_data.genotypes is None or not geno_data.obs_ids:
            print(f"Warning: Skipping alignment for '{geno_data.name}', missing genotypes or obs_ids."); continue

        original_obs_ids = list(geno_data.obs_ids) # Keep a copy of original IDs for output_pred_ids logic
        original_genotypes = geno_data.genotypes.copy() # Keep a copy of original genotypes

        if set(geno_data.obs_ids) != set(phenotype_obs_ids) or list(geno_data.obs_ids) != list(phenotype_obs_ids) :
            print(f"  Aligning genotypes for '{geno_data.name}' with {len(phenotype_obs_ids)} phenotype IDs...")
            try:
                Z_pheno = make_incidence_matrix_for_ids(phenotype_obs_ids, original_obs_ids) # Align to original
                if geno_data.is_grm:
                    geno_data.genotypes = Z_pheno @ original_genotypes @ Z_pheno.T
                else:
                    geno_data.genotypes = Z_pheno @ original_genotypes
                geno_data.obs_ids = list(phenotype_obs_ids)
                geno_data.n_obs = len(phenotype_obs_ids)
            except ValueError as e:
                print(f"Error aligning '{geno_data.name}' with phenotype IDs: {e}. Keeping original."); continue

        if output_pred_ids:
            print(f"  Preparing output genotypes for '{geno_data.name}' for {len(output_pred_ids)} prediction IDs...")
            try:
                # Use original full genotype matrix and original IDs for output alignment
                Z_output = make_incidence_matrix_for_ids(output_pred_ids, original_obs_ids)
                if geno_data.is_grm: # If original was GRM
                    # Output GRM for prediction IDs: Z_out @ G_original @ Z_out.T
                    geno_data.output_genotypes = Z_output @ original_genotypes @ Z_output.T
                else: # Standard marker matrix
                    geno_data.output_genotypes = Z_output @ original_genotypes
            except ValueError as e:
                 print(f"Error preparing output genotypes for '{geno_data.name}': {e}.")
                 geno_data.output_genotypes = None
        elif geno_data.genotypes is not None: # No output_pred_ids, output_genotypes is the phenotype-aligned one
            geno_data.output_genotypes = geno_data.genotypes


def _genetic_to_marker_variance_st(
    genetic_variance_val: float,
    sum_2pq: float,
    pi_val: float # Proportion of markers *not* having zero effect (1-pi_0 from some notations)
) -> float:
    """Calculates marker effect variance from total genetic variance for single trait."""
    if sum_2pq == 0: return 0.0 # Avoid division by zero
    # Julia: M.G.val = M.genetic_variance.val/((1-π)*M.sum2pq)
    # Here, pi_val is equivalent to (1-π) if π was prob of zero effect.
    # If Pi in JWAS is prob of *having* an effect, then it's pi_val.
    # Let's assume pi_val is the proportion of markers with effects.
    # If pi_val = 0 (no markers have effects), this would be problematic.
    # The Julia code `(1-π)` suggests π is the BayesC prior prob of marker having zero effect.
    # So, if `Pi` in `get_genotypes` is `Prob(effect=0)`, then `1-Pi` is `Prob(effect!=0)`.
    # Let's assume `pi_val` passed here is `Prob(effect != 0)`.
    if pi_val == 0: # No markers have effects, marker variance is undefined or effectively infinite if genetic var > 0
        # This case should ideally be handled by the model (e.g. no marker term)
        # For safety, return 0 or raise error. Let's return a large number if genetic_variance_val > 0.
        return 1e12 if genetic_variance_val > 0 else 0.0
    return genetic_variance_val / (pi_val * sum_2pq)


def _genetic_to_marker_variance_mt(
    genetic_cov_matrix: np.ndarray, # n_traits x n_traits
    sum_2pq: float,
    pi_dict: Dict[Tuple[float, ...], float] # Pi from JWAS for multi-trait
) -> np.ndarray:
    """Calculates marker effect covariance matrix from genetic covariance matrix for multi-trait."""
    n_traits = genetic_cov_matrix.shape[0]
    denom_matrix = np.zeros((n_traits, n_traits))

    # Pi_dict key is a tuple of (0.0 or 1.0) indicating effect pattern across traits
    # Value is the probability of that pattern.
    for i in range(n_traits):
        for j in range(i, n_traits):
            # Sum probabilities of patterns where both trait i and trait j have an effect
            sum_prob_both_effect = 0.0
            for pattern_key, prob_val in pi_dict.items():
                if len(pattern_key) == n_traits and pattern_key[i] == 1.0 and pattern_key[j] == 1.0:
                    sum_prob_both_effect += prob_val

            if sum_2pq == 0: denom_val = 1e-12 # Effectively infinite if genetic_cov_matrix is non-zero
            else: denom_val = sum_2pq * sum_prob_both_effect

            if denom_val == 0: # Avoid division by zero
                denom_matrix[i,j] = 1e-12 if genetic_cov_matrix[i,j] !=0 else 1.0 # make marker var huge or G_ij if G_ij=0
            else:
                denom_matrix[i,j] = denom_val
            if i != j: denom_matrix[j,i] = denom_matrix[i,j]

    marker_cov_matrix = genetic_cov_matrix / denom_matrix # Element-wise division
    return marker_cov_matrix


def setup_marker_hyperparameters(
    geno_data: GenotypesData,
    n_model_traits: int, # Number of traits in the current model context
    # is_rrm: bool = False # TODO: Handle RRM specific ntraits for Pi later
):
    """
    Initializes marker effect variance (geno_data.marker_effect_variance) if not directly provided,
    by deriving it from total genetic variance (geno_data.genetic_variance) and Pi.
    Also sets default multi-trait Pi if needed.
    Modifies geno_data in place.
    """

    # Default Pi for multi-trait if not BayesL/RR-BLUP and Pi is simple float
    if n_model_traits > 1 and isinstance(geno_data.pi_value, float) and \
       geno_data.method not in ["RR-BLUP", "BayesL", "GBLUP"]: # GBLUP doesn't use Pi
        print(f"  Info: Defaulting multi-trait Pi for '{geno_data.name}' assuming all markers affect all traits.")
        default_pi_mt: Dict[Tuple[float,...], float] = {}
        all_one_pattern = tuple([1.0] * n_model_traits)
        all_zero_pattern = tuple([0.0] * n_model_traits) # Not explicitly in Julia default but good for sum to 1

        # Simplified default: only pattern with all 1s has prob 1.0
        # More complex default from Julia: iterate all 2^n_traits patterns.
        # For now, just the all-ones pattern. This needs to match MCMC expectation.
        # The Julia default was: Pi[ones(ntraits)]=1.0, others 0.0
        for k_int in range(1 << n_model_traits):
            pattern_tuple = tuple(float(b) for b in bin(k_int)[2:].zfill(n_model_traits))
            if pattern_tuple == all_one_pattern:
                default_pi_mt[pattern_tuple] = 1.0
            else:
                default_pi_mt[pattern_tuple] = 0.0
        geno_data.pi_value = default_pi_mt

    # Calculate marker effect variance (Mi.G.val) from genetic variance if not set
    if geno_data.marker_effect_variance and geno_data.marker_effect_variance.value is None:
        if geno_data.genetic_variance and geno_data.genetic_variance.value is not None:
            if geno_data.method == "GBLUP":
                # For GBLUP, marker_effect_variance (sigma_g^2) is same as genetic_variance (sigma_a^2)
                # if the GRM is used directly. If it's marker-based GBLUP, then it's different.
                # Assuming if method=GBLUP, genetic_variance is sigma_a^2 for the GRM.
                # And marker_effect_variance field is used to store this.
                geno_data.marker_effect_variance.value = np.copy(geno_data.genetic_variance.value)
                print(f"  Info: For GBLUP '{geno_data.name}', marker variance set from genetic variance.")
            elif geno_data.sum_2pq is not None and geno_data.sum_2pq > 0:
                print(f"  Info: Calculating marker effect variance from genetic variance for '{geno_data.name}'.")
                if n_model_traits == 1:
                    if isinstance(geno_data.pi_value, float) and isinstance(geno_data.genetic_variance.value, (float, int, np.floating)):
                        geno_data.marker_effect_variance.value = _genetic_to_marker_variance_st(
                            float(geno_data.genetic_variance.value), geno_data.sum_2pq, geno_data.pi_value
                        )
                    else: print(f"Warning: Cannot calculate marker variance for ST {geno_data.name}, Pi or genetic_variance type mismatch.")
                else: # Multi-trait
                    if isinstance(geno_data.pi_value, dict) and isinstance(geno_data.genetic_variance.value, np.ndarray):
                        geno_data.marker_effect_variance.value = _genetic_to_marker_variance_mt(
                            geno_data.genetic_variance.value, geno_data.sum_2pq, geno_data.pi_value
                        )
                    else: print(f"Warning: Cannot calculate marker variance for MT {geno_data.name}, Pi or genetic_variance type mismatch.")

                # Check for positive definiteness if matrix, or positivity if scalar
                if geno_data.marker_effect_variance.value is not None:
                    is_pd_ok = False
                    if isinstance(geno_data.marker_effect_variance.value, np.ndarray):
                        try: is_pd_ok = np.all(np.linalg.eigvals(geno_data.marker_effect_variance.value) > 1e-12)
                        except np.linalg.LinAlgError: is_pd_ok = False
                    else: # scalar
                        is_pd_ok = float(geno_data.marker_effect_variance.value) > 0

                    if not is_pd_ok:
                        print(f"Error: Calculated marker effect (co)variance for '{geno_data.name}' is not positive definite/positive. Check Pi and genetic variance priors.")
                        geno_data.marker_effect_variance.value = None # Reset to avoid issues
            else:
                print(f"Warning: Cannot calculate marker variance for '{geno_data.name}', sum_2pq is missing or zero.")
        else:
            print(f"Warning: Marker variance for '{geno_data.name}' not set and cannot be derived (no genetic variance provided).")

    # Calculate prior scale for marker effect variance (Mi.G.scale)
    # Julia: Mi.G.scale = Mi.G.val*(Mi.G.df-Mi.ntraits-1)/Mi.G.df (single-trait like, but uses ntraits)
    #        or Mi.G.scale = Mi.G.val*(Mi.G.df-Mi.ntraits-1) (multi-trait like)
    # This is for an Inverse Wishart E[Sigma] = Scale / (df - p - 1) -> Scale = E[Sigma]*(df-p-1)
    # Or for Scaled Inv Chi2 E[sigma^2]=S^2. If S^2 is prior mean (Mi.G.val), and df is nu.
    # Scale param for scipy.invgamma(a=df/2, scale=beta) is beta = nu*S^2/2.
    # Julia's logic `val*(df-p-1)` for scale seems to assume val is E[Sigma] and it's calculating Scale_matrix for IW.
    # And `val*(df-p-1)/df` for single trait is unusual.
    # Let's assume Mi.G.val is the prior mean E[sigma_g^2] or E[Sigma_g].
    # And Mi.G.df is nu_0.
    # For InvGamma(alpha=nu_0/2, beta=nu_0*S0^2/2), S0^2 is the prior scale parameter.
    # If Mi.G.value is this S0^2 (prior scale parameter itself), then Mi.G.scale should be Mi.G.value.
    # The `VarianceCovariance` already has `scale` which is intended for this prior scale.
    # If `G_prior_value` was this prior scale, it's already set.
    # The Julia code might be setting `Mi.G.scale` based on `Mi.G.val` IF `Mi.G.val` was derived
    # (e.g. from total genetic variance) and is now considered the prior mean, and then it calculates
    # the scale parameter for the prior distribution.
    # This is complex. For now, if `geno_data.marker_effect_variance.scale` is None and `.value` is set,
    # we can use the Julia logic as a heuristic to set the prior scale.

    if geno_data.marker_effect_variance and \
       geno_data.marker_effect_variance.value is not None and \
       geno_data.marker_effect_variance.scale is None: # If prior scale wasn't set directly

        prior_mean_val = geno_data.marker_effect_variance.value
        prior_df = geno_data.marker_effect_variance.df
        p = n_model_traits # Number of "traits" for this variance component

        if prior_df > p + 1: # Condition for Inverse Wishart mean
            # Assuming prior_mean_val is E[Sigma_g], calculate Psi_0 for IW(nu_0, Psi_0)
            calculated_prior_scale = prior_mean_val * (prior_df - p - 1)
            geno_data.marker_effect_variance.scale = calculated_prior_scale
            print(f"  Info: Calculated prior scale for marker variance of '{geno_data.name}' based on its value and df.")
        else:
            # Fallback: use value as scale if df is too small for IW mean formula
            # This is often done if value is considered the scale parameter S_0 directly.
            geno_data.marker_effect_variance.scale = np.copy(prior_mean_val)
            print(f"  Info: Setting prior scale for marker variance of '{geno_data.name}' to its value (df too small for IW mean formula).")


if __name__ == '__main__':
    print("--- Testing Genotype Utilities ---")
    from ..core.genotypes import GenotypesData # For test instances

    print("\nTesting centering:")
    # ... (rest of __main__ from previous state) ...
    target = ["C", "A", "D", "B"]
    source = ["A", "B", "C", "D", "E"]
    Z = make_incidence_matrix_for_ids(target, source)
    print(f"Target IDs: {target}")
    print(f"Source IDs: {source}")
    print(f"Z matrix ({Z.shape}):\n{Z.toarray()}")
    source_data_vec = np.array([10, 20, 30, 40, 50])
    target_data_vec = Z @ source_data_vec
    print(f"Source data: {source_data_vec}")
    print(f"Target data (Z @ source): {target_data_vec}")
    np.testing.assert_allclose(target_data_vec, np.array([30, 10, 40, 20]))

    try: make_incidence_matrix_for_ids(["A", "F"], ["A", "B"])
    except ValueError as e: print(f"Caught expected error: {e}")

    print("\n--- Testing Hyperparameter Setup ---")
    geno_st = GenotypesData(name="st_geno", method="BayesC")
    geno_st.sum_2pq = 250.0
    geno_st.pi_value = 0.95 # Prob of effect for ST (1-pi_0)
    geno_st.genetic_variance = VarianceCovariance(value=50.0, df=4.0, scale=50.0)
    geno_st.marker_effect_variance = VarianceCovariance(value=None, df=4.0, scale=None) # To be derived

    setup_marker_hyperparameters(geno_st, n_model_traits=1)
    print(f"ST Geno: Marker Var Value: {geno_st.marker_effect_variance.value:.4f}, Scale: {geno_st.marker_effect_variance.scale:.4f if geno_st.marker_effect_variance.scale is not None else 'None'}")
    # Expected marker var value: 50.0 / (0.95 * 250.0) = 50.0 / 237.5 = 0.2105
    # Expected scale: value * (df - p - 1) = 0.2105 * (4 - 1 - 1) = 0.2105 * 2 = 0.4210

    geno_mt = GenotypesData(name="mt_geno", method="BayesC")
    geno_mt.sum_2pq = 300.0
    # Pi not set, should default for MT
    geno_mt.genetic_variance = VarianceCovariance(value=np.array([[10.,2.],[2.,8.]]), df=5.0, scale=np.array([[10.,2.],[2.,8.]]))
    geno_mt.marker_effect_variance = VarianceCovariance(value=None, df=5.0, scale=None)

    setup_marker_hyperparameters(geno_mt, n_model_traits=2)
    print(f"MT Geno: Pi: {geno_mt.pi_value}")
    print(f"MT Geno: Marker Var Value:\n{geno_mt.marker_effect_variance.value}\nScale:\n{geno_data.marker_effect_variance.scale if geno_mt.marker_effect_variance.scale is not None else 'None'}")
    # Default Pi: {(1,1):1.0, others:0.0}. sum_prob_both_effect for [0,0] is 1.0. For [0,1] is 0. For [1,1] is 1.0.
    # Denom_00 = 300*1=300. Denom_11=300. Denom_01=0 (problematic if G_01 !=0).
    # This default Pi logic needs to be robust. If sum_prob_both_effect is 0 for an off-diagonal G_ij, marker_var_ij will be huge.
    # Julia's default: Pi[ones(ntraits)]=1.0. So only G_diag elements are affected if genetic_variance is diag.
    # If genetic_variance has off-diagonals, and pi_dict makes denom_ij zero, then marker_var_ij is inf.
    # This implies genetic_variance and Pi must be compatible.
    # Current _genetic_to_marker_variance_mt uses 1e-12 if denom_val is 0 and G_ij!=0.
```
