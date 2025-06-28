import numpy as np
from scipy.sparse import spmatrix # For type hinting Vi
from .distributions import sample_scaled_inverse_chi_squared, sample_inverse_wishart
from ..core.variance_covariance import VarianceCovariance # For type hinting Gi objects

# Need to import List, Optional, Union, TYPE_CHECKING for type hints
from typing import List, Optional, Union, TYPE_CHECKING
if TYPE_CHECKING: # To avoid circular import issues for type hinting MME
    from ..core.mme import MixedModelEquations


def sample_scalar_variance_component(
    data_vector: np.ndarray,
    prior_df: float,
    prior_scale_parameter: float,
    inv_weights: Optional[np.ndarray] = None
) -> float:
    """
    Samples a scalar variance component (e.g., residual variance, marker effect variance for single trait).
    Assumes a Scaled Inverse Chi-squared posterior distribution.
    Equivalent to Julia's `sample_variance(x, n, df, scale)` which samples (dot(x,x) + df*scale)/rand(Chisq(n+df)).

    Args:
        data_vector: Numpy array of data points (e.g., residuals, effects).
        prior_df: Degrees of freedom for the prior (nu_0).
        prior_scale_parameter: Scale parameter (S_0^2) for the Scaled Inverse Chi-squared prior.
        inv_weights: Optional numpy array of inverse variance weights for observations in data_vector.

    Returns:
        A float sample from the posterior distribution of the variance component.
    """
    n = len(data_vector)
    if n == 0 and prior_df <=0 :
        if prior_scale_parameter > 0 : return prior_scale_parameter
        raise ValueError("Cannot sample variance with no data and non-positive prior_df/scale.")

    if inv_weights is not None:
        if len(inv_weights) != n:
            raise ValueError("Length of inv_weights must match data_vector.")
        weighted_data_vector = data_vector * np.sqrt(inv_weights)
        sum_sq_data = np.dot(weighted_data_vector, weighted_data_vector)
    else:
        sum_sq_data = np.dot(data_vector, data_vector)

    posterior_df = prior_df + n

    if posterior_df <=0 :
        if prior_scale_parameter > 0: return prior_scale_parameter
        raise ValueError(f"Posterior degrees of freedom ({posterior_df}) must be positive.")

    posterior_scale_sq_param = (sum_sq_data + prior_df * prior_scale_parameter) / posterior_df

    if posterior_scale_sq_param < 0:
        print(f"Warning: Calculated posterior scale parameter S2_post ({posterior_scale_sq_param}) is negative. Check priors. Returning prior scale parameter if positive, else error.")
        if prior_scale_parameter > 0: return prior_scale_parameter
        raise ValueError("Negative posterior scale parameter S2_post for ScaledInvChi2.")
    if posterior_scale_sq_param == 0 and posterior_df > 0:
        return 0.0

    return sample_scaled_inverse_chi_squared(df=posterior_df, scale_sq=posterior_scale_sq_param)


def sample_matrix_variance_component(
    data_arrays: List[np.ndarray],
    n_observations: int,
    prior_df: float,
    prior_scale_matrix: np.ndarray,
    inv_weights: Optional[np.ndarray] = None,
    constraint: bool = False
) -> np.ndarray:
    """
    Samples a matrix variance component (e.g., residual covariance matrix R for multi-trait).
    Assumes an Inverse Wishart posterior distribution if constraint=False.
    If constraint=True, samples diagonal elements independently.
    """
    n_traits = len(data_arrays)
    if n_traits == 0: raise ValueError("data_arrays cannot be empty.")
    if not all(len(arr) == n_observations for arr in data_arrays):
        raise ValueError("All arrays in data_arrays must have length n_observations.")
    if prior_scale_matrix.shape != (n_traits, n_traits):
        raise ValueError(f"prior_scale_matrix must be {n_traits}x{n_traits}.")

    SSCP_data = np.zeros((n_traits, n_traits))
    weighted_data_arrays = data_arrays
    if inv_weights is not None:
        if len(inv_weights) != n_observations: raise ValueError("Length of inv_weights mismatch.")
        sqrt_inv_weights = np.sqrt(inv_weights)
        weighted_data_arrays = [data_arrays[i] * sqrt_inv_weights for i in range(n_traits)]

    for i in range(n_traits):
        for j in range(i, n_traits):
            dot_product = np.dot(weighted_data_arrays[i], weighted_data_arrays[j])
            SSCP_data[i, j] = dot_product
            if i != j: SSCP_data[j, i] = dot_product

    posterior_df_iw = prior_df + n_observations
    posterior_scale_matrix_iw = prior_scale_matrix + SSCP_data

    if constraint:
        sampled_variances = np.zeros(n_traits)
        for i in range(n_traits):
            scalar_prior_df_eff = prior_df
            scalar_prior_scale_param = prior_scale_matrix[i,i]

            nu_post_scalar = scalar_prior_df_eff + n_observations
            # For diagonal sampling, the "data" for variance i is just that trait's vector of weighted effects/residuals.
            # sum_sq_data for variance i is SSCP_data[i,i] (already sum of w_k * x_ik^2)
            sse_post_scalar = SSCP_data[i,i] + scalar_prior_df_eff * scalar_prior_scale_param

            if nu_post_scalar <= 0:
                if scalar_prior_scale_param > 0: sampled_variances[i] = scalar_prior_scale_param; continue
                raise ValueError(f"Scalar posterior df ({nu_post_scalar}) for trait {i} must be positive.")

            s2_post_scalar_param = sse_post_scalar / nu_post_scalar
            if s2_post_scalar_param < 0:
                if scalar_prior_scale_param > 0: sampled_variances[i] = scalar_prior_scale_param; continue
                raise ValueError(f"Negative S2_post_scalar_param for trait {i}.")
            if s2_post_scalar_param == 0 and nu_post_scalar > 0:
                sampled_variances[i] = 0.0; continue

            sampled_variances[i] = sample_scaled_inverse_chi_squared(df=nu_post_scalar, scale_sq=s2_post_scalar_param)
        return np.diag(sampled_variances)
    else:
        return sample_inverse_wishart(df=posterior_df_iw, scale_matrix=posterior_scale_matrix_iw)


def sample_general_random_effect_variances(mme: 'MixedModelEquations'):
    """
    Samples variance/covariance components for general random effects
    (e.g., polygenic, other user-defined random effects) stored in mme.random_effect_terms.
    Updates rt.Gi_new.value (and rt.Gi.value if appropriate) for each term with the new G_inverse.
    This is equivalent to Julia's `sampleVCs`.

    Args:
        mme: The MixedModelEquations object.
    """
    if mme.solutions is None:
        # print("Warning: Solutions vector not available, cannot sample general random effect VCs.")
        return

    for rt in mme.random_effect_terms:
        # Use rt.Gi for prior information (df, scale which is Psi_0)
        # Sampled G_inv updates rt.Gi_new (for ST MCMC state) and also rt.Gi (as current G_inv)

        if rt.Gi is None or not rt.Gi.estimate_variance:
            continue
        if rt.Gi.df is None or rt.Gi.scale is None:
            print(f"Warning: Prior df or scale (Psi_0) not set for random effect {rt.term_array} in rt.Gi. Skipping VC sampling.")
            continue

        n_levels_term = 0
        if not rt.term_array or not rt.term_array[0] in mme.model_term_dict:
            print(f"Warning: Could not find model term for random effect {rt.term_array}. Skipping VC sampling.")
            continue
        first_model_term_in_rt = mme.model_term_dict[rt.term_array[0]]
        n_levels_term = first_model_term_in_rt.n_levels

        Vi: Union[np.ndarray, spmatrix]
        if rt.V_inv is not None: Vi = rt.V_inv
        else:
            if n_levels_term > 0:
                from scipy.sparse import identity as sparse_identity
                Vi = sparse_identity(n_levels_term, format="csc", dtype=np.float64)
            else:
                # print(f"Warning: Random effect {rt.term_array} has no levels. Skipping VC sampling.")
                continue # Or handle as error if this state is unexpected

        n_effect_traits = len(rt.term_array)
        S_matrix = np.zeros((n_effect_traits, n_effect_traits), dtype=np.float64)

        solution_vectors_for_rt: List[Optional[np.ndarray]] = [None] * n_effect_traits
        valid_solutions_found = True
        for i, term_i_str in enumerate(rt.term_array):
            model_term_i = mme.model_term_dict.get(term_i_str)
            if not model_term_i or model_term_i.n_levels == 0: valid_solutions_found = False; break
            solution_vectors_for_rt[i] = mme.solutions[model_term_i.start_pos : model_term_i.start_pos + model_term_i.n_levels]
        if not valid_solutions_found: continue


        for i in range(n_effect_traits):
            sol_i = solution_vectors_for_rt[i]
            if sol_i is None: continue # Should have been caught by valid_solutions_found

            for j in range(i, n_effect_traits):
                sol_j = solution_vectors_for_rt[j]
                if sol_j is None: continue

                # Check conformability for Vi with sol_i and sol_j
                # Vi is (n_levels_term x n_levels_term)
                # sol_i, sol_j are (n_levels_term,)
                if Vi.shape[0] != len(sol_i) or Vi.shape[1] != len(sol_j):
                     print(f"Warning: Shape mismatch for Vi ({Vi.shape}) and solution vectors for RE {rt.term_array}. Skipping S_ij.")
                     continue

                s_ij = sol_i.T @ Vi @ sol_j
                S_matrix[i, j] = s_ij
                if i != j: S_matrix[j, i] = s_ij

        prior_df_for_G = rt.Gi.df
        prior_scale_matrix_for_G = rt.Gi.scale # This is Psi_0

        posterior_df_iw = prior_df_for_G + n_levels_term
        posterior_scale_matrix_iw = prior_scale_matrix_for_G + S_matrix

        try:
            sampled_G_matrix = sample_inverse_wishart(df=posterior_df_iw, scale_matrix=posterior_scale_matrix_iw)
        except ValueError as e:
            print(f"Error sampling Inverse Wishart for RE {rt.term_array} (df={posterior_df_iw}): {e}. Scale matrix:\n{posterior_scale_matrix_iw}\nSkipping update.")
            continue

        try:
            sampled_G_inv_matrix = np.linalg.inv(sampled_G_matrix)
        except np.linalg.LinAlgError:
            print(f"Error inverting sampled G matrix for RE {rt.term_array}. Sampled G:\n{sampled_G_matrix}\nSkipping update.")
            continue

        # Update the primary VC store (rt.Gi) which holds current G_inv and prior info
        rt.Gi.value = sampled_G_inv_matrix

        # If Gi_new is used (typically for ST MCMC state), update it too.
        if rt.Gi_new is not None:
            rt.Gi_new.value = np.copy(sampled_G_inv_matrix)
            # Gi_new's .df and .scale are priors, should not change unless Empirical Bayes.

        # Special handling for pedigree term to update mme.pedigree_inv_covariance
        if rt.random_type == "A":
            if mme.pedigree_inv_covariance is None:
                mme.pedigree_inv_covariance = VarianceCovariance(value=np.copy(sampled_G_inv_matrix),
                                                                 df=rt.Gi.df, scale=rt.Gi.scale) # Ensure it exists with priors
            else:
                mme.pedigree_inv_covariance.value = np.copy(sampled_G_inv_matrix)


if __name__ == '__main__':
    print("Testing variance component samplers:")
    np.random.seed(123)
    residuals = np.random.randn(100) * np.sqrt(2.0); prior_df_s = 4.0; prior_scale_s = 1.0
    samples_s = [sample_scalar_variance_component(residuals, prior_df_s, prior_scale_s) for _ in range(1000)]
    print(f"Scalar variance: Mean={np.mean(samples_s):.3f}")

    print("\nConceptual test for sample_general_random_effect_variances:")
    if callable(sample_general_random_effect_variances):
        print("sample_general_random_effect_variances function is defined.")

```
