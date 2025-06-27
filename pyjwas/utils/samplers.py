import numpy as np
from .distributions import sample_scaled_inverse_chi_squared, sample_inverse_wishart

def sample_scalar_variance_component(
    data_vector: np.ndarray,
    prior_df: float,
    prior_scale_parameter: float, # This is 'scale' in Julia's sample_variance, S0_nu in some texts for Inv-Chi2
    inv_weights: Optional[np.ndarray] = None
) -> float:
    """
    Samples a scalar variance component (e.g., residual variance, marker effect variance for single trait).
    Assumes a Scaled Inverse Chi-squared posterior distribution.
    Equivalent to Julia's `sample_variance(x, n, df, scale)` which samples (dot(x,x) + df*scale)/rand(Chisq(n+df)).

    This posterior arises from a likelihood N(0, sigma^2/w_i) and prior sigma^2 ~ ScaledInvChi2(prior_df, prior_scale_parameter).
    The posterior for sigma^2 is ScaledInvChi2(nu_post, S2_post)
    where nu_post = prior_df + n
          S2_post = (sum(w_i * x_i^2) + prior_df * prior_scale_parameter) / nu_post

    Args:
        data_vector: Numpy array of data points (e.g., residuals, effects) from which to estimate variance.
        prior_df: Degrees of freedom for the prior (nu_0).
        prior_scale_parameter: Scale parameter (S_0^2 or tau_0^2) for the Scaled Inverse Chi-squared prior.
                               In Julia's (dot(x,x) + df*scale), 'scale' is this prior_scale_parameter.
        inv_weights: Optional numpy array of inverse variance weights for observations in data_vector.
                     If provided, data_vector elements are effectively x_i * sqrt(w_i).

    Returns:
        A float sample from the posterior distribution of the variance component.
    """
    n = len(data_vector)
    if n == 0 and prior_df <=0 : # Cannot sample if no data and no informative prior
        # This case needs careful handling, depends on model assumptions
        # Return prior scale, or raise error, or return large variance?
        # JWAS likely assumes n > 0 or prior_df is reasonably positive.
        if prior_scale_parameter > 0 : return prior_scale_parameter # Fallback to prior scale
        raise ValueError("Cannot sample variance with no data and non-positive prior_df/scale.")


    if inv_weights is not None:
        if len(inv_weights) != n:
            raise ValueError("Length of inv_weights must match data_vector.")
        # Apply weights: sum_sq_data = sum( (x_i * sqrt(w_i))^2 ) = sum(x_i^2 * w_i)
        # In Julia: sample_variance(x.*sqrt.(invweights), n, df, scale)
        # So, the input 'x' to Julia's underlying logic is already x*sqrt(w)
        # Here, data_vector is 'x', inv_weights is 'w' (not sqrt(w)).
        # sum_sq_data = np.sum((data_vector**2) * inv_weights)
        # Let's match Julia's effective input:
        weighted_data_vector = data_vector * np.sqrt(inv_weights)
        sum_sq_data = np.dot(weighted_data_vector, weighted_data_vector)
    else:
        sum_sq_data = np.dot(data_vector, data_vector)

    posterior_df = prior_df + n

    # SSE_posterior = sum_sq_data + prior_df * prior_scale_parameter
    # This is the numerator of the fraction that gets divided by ChiSq(posterior_df)
    # This SSE_posterior is the scale parameter for InvGamma(posterior_df/2, SSE_posterior/2)
    # Or, for ScaledInvChi2(posterior_df, S2_posterior), where S2_posterior = SSE_posterior / posterior_df

    posterior_scale_sq = (sum_sq_data + prior_df * prior_scale_parameter) / posterior_df
    if posterior_df <=0 : # Should not happen if n>0 or prior_df>0
        if prior_scale_parameter > 0: return prior_scale_parameter
        raise ValueError(f"Posterior degrees of freedom ({posterior_df}) must be positive.")
    if posterior_scale_sq <=0 and posterior_df > 0: # Can happen if sum_sq is tiny and prior_scale_parameter is zero/negative
         # This implies data is all zero and prior is not informative enough to prevent zero variance.
         # Small positive variance to avoid issues, or handle based on context.
         # print(f"Warning: posterior_scale_sq is {posterior_scale_sq}. Sampling might fail or yield zero.")
         # Let sample_scaled_inverse_chi_squared handle it, it might raise error if scale_sq is not positive.
         # If posterior_scale_sq is zero, it means SSE_posterior is zero.
         # Then variance sample will be zero. This is often acceptable.
         pass


    return sample_scaled_inverse_chi_squared(df=posterior_df, scale_sq=posterior_scale_sq)


def sample_matrix_variance_component(
    data_arrays: List[np.ndarray], # List of vectors, one per trait: [y1_corr, y2_corr, ...]
    n_observations: int, # Number of observations per trait (common)
    prior_df: float,
    prior_scale_matrix: np.ndarray, # Prior scale matrix (Psi_0) for Inverse Wishart
    inv_weights: Optional[np.ndarray] = None, # Common inverse weights for observations
    constraint: bool = False # If true, sample diagonal elements independently
) -> np.ndarray:
    """
    Samples a matrix variance component (e.g., residual covariance matrix R for multi-trait).
    Assumes an Inverse Wishart posterior distribution if constraint=False.
    If constraint=True, samples diagonal elements independently using sample_scalar_variance_component logic.

    Args:
        data_arrays: List of 1D numpy arrays. Each array contains data for one trait
                     (e.g., residuals y_corrected_trait_1, y_corrected_trait_2, ...).
                     All arrays must have length `n_observations`.
        n_observations: Number of observations for each trait.
        prior_df: Degrees of freedom for the prior (nu_0).
        prior_scale_matrix: Scale matrix (Psi_0) for the Inverse Wishart prior (p x p).
                            If constraint=True, this should be a diagonal matrix, and its
                            diagonal elements are used as prior_scale_parameter for scalar sampling.
        inv_weights: Optional 1D numpy array of common inverse variance weights for observations.
        constraint: If True, R is diagonal, and variances are sampled independently.

    Returns:
        A p x p numpy array for the sampled covariance matrix.
    """
    n_traits = len(data_arrays)
    if n_traits == 0:
        raise ValueError("data_arrays cannot be empty.")
    if not all(len(arr) == n_observations for arr in data_arrays):
        raise ValueError("All arrays in data_arrays must have length n_observations.")
    if prior_scale_matrix.shape != (n_traits, n_traits):
        raise ValueError(f"prior_scale_matrix must be {n_traits}x{n_traits}.")

    # Calculate Sum of Squares and Cross-Products (SSCP) matrix from data_arrays
    # SSCP_ij = y_i' * W * y_j where W is diag(inv_weights) or Identity
    SSCP_data = np.zeros((n_traits, n_traits))

    # Apply weights to data_arrays first if inv_weights are provided
    # y_weighted_i = y_i * sqrt(w)
    # SSCP_ij = y_weighted_i' * y_weighted_j

    weighted_data_arrays = []
    if inv_weights is not None:
        if len(inv_weights) != n_observations:
            raise ValueError("Length of inv_weights must match n_observations.")
        sqrt_inv_weights = np.sqrt(inv_weights)
        for i in range(n_traits):
            weighted_data_arrays.append(data_arrays[i] * sqrt_inv_weights)
    else:
        weighted_data_arrays = data_arrays

    for i in range(n_traits):
        for j in range(i, n_traits):
            dot_product = np.dot(weighted_data_arrays[i], weighted_data_arrays[j])
            SSCP_data[i, j] = dot_product
            if i != j:
                SSCP_data[j, i] = dot_product

    posterior_df = prior_df + n_observations
    posterior_scale_matrix = prior_scale_matrix + SSCP_data

    if constraint:
        # Sample diagonal elements independently
        # prior_scale_matrix is expected to be diagonal, use its diagonal elements
        # for the scalar prior_scale_parameter.
        sampled_variances = np.zeros(n_traits)
        for i in range(n_traits):
            # For scalar sampling, data_vector is weighted_data_arrays[i]
            # prior_df is for the multivariate prior, so for each scalar variance,
            # the effective prior_df might be different if derived from a Wishart.
            # Julia's code uses `df` (multivariate prior_df) for scalar sampling too in this path.
            # And `scale[i,i]` from the multivariate prior_scale_matrix.
            scalar_prior_scale = prior_scale_matrix[i,i]
            # The data for variance i is data_arrays[i] (or weighted_data_arrays[i])
            # sum_sq_data for variance i is SSCP_data[i,i]

            # Using the formula from sample_scalar_variance_component's core logic:
            # nu_post_scalar = prior_df_scalar + n_obs
            # SSE_post_scalar = SSCP_data[i,i] + prior_df_scalar * scalar_prior_scale
            # Here, Julia seems to use the multivariate prior_df for each scalar sample's prior_df part.
            # (SSE[traiti,traiti]+df*scale[traiti,traiti])/rand(Chisq(nobs+df))
            # This means nu_post_scalar = prior_df + n_observations
            # And SSE_post_scalar = SSCP_data[i,i] + prior_df * scalar_prior_scale

            nu_post_scalar = prior_df + n_observations # Using multivariate prior_df
            sse_post_scalar = SSCP_data[i,i] + prior_df * scalar_prior_scale

            if nu_post_scalar <=0 :
                if scalar_prior_scale > 0 : sampled_variances[i] = scalar_prior_scale; continue
                raise ValueError(f"Scalar posterior df ({nu_post_scalar}) for trait {i} must be positive.")

            s2_post_scalar = sse_post_scalar / nu_post_scalar
            if s2_post_scalar <=0 and nu_post_scalar > 0:
                 # Let sample_scaled_inverse_chi_squared handle/error if s2_post_scalar is not positive
                 pass

            sampled_variances[i] = sample_scaled_inverse_chi_squared(df=nu_post_scalar, scale_sq=s2_post_scalar)
        return np.diag(sampled_variances)
    else:
        # Sample from Inverse Wishart
        # df for sample_inverse_wishart is posterior_df
        # scale_matrix for sample_inverse_wishart is posterior_scale_matrix
        return sample_inverse_wishart(df=posterior_df, scale_matrix=posterior_scale_matrix)


if __name__ == '__main__':
    print("Testing variance component samplers:")
    np.random.seed(123)

    # Test sample_scalar_variance_component
    residuals = np.random.randn(100) * np.sqrt(2.0) # True variance = 2.0
    prior_df_s = 4.0
    prior_scale_s = 1.0 # S0^2

    samples_s = [sample_scalar_variance_component(residuals, prior_df_s, prior_scale_s) for _ in range(1000)]
    print(f"Scalar variance: Mean={np.mean(samples_s):.3f} (Prior mean if data weak, converges to data var)")
    # Expected posterior mean approx = (sum(res^2) + prior_df*prior_scale) / (n + prior_df - 2) (for InvGamma mean)
    # Or for ScaledInvChi2(nu,S^2), mean is S^2 if nu very large, else S^2 * nu / (nu-2) (approx)
    # Here, S2_post = (dot(res,res) + prior_df_s*prior_scale_s)/(100+prior_df_s)
    # Mean approx S2_post if (100+prior_df_s) is large.

    residuals_weighted = np.random.randn(100) * np.sqrt(np.repeat([2.0, 0.5], 50))
    weights = 1.0 / np.repeat([2.0, 0.5], 50) # variances are 2.0 and 0.5
    inv_var_weights = weights # inverse of individual error variances

    samples_sw = [sample_scalar_variance_component(residuals_weighted, prior_df_s, prior_scale_s, inv_weights=inv_var_weights) for _ in range(1000)]
    # This samples a common sigma^2, assuming y_i ~ N(mu_i, sigma^2 / w_i)
    # If true sigma^2 = 1, then var(y_i) = 1/w_i.
    # Here, true var(y_i) are 2.0 and 0.5. If sigma^2 = 1, then w_i should be 0.5 and 2.0.
    # So inv_var_weights is correct.
    print(f"Scalar variance (weighted): Mean={np.mean(samples_sw):.3f} (should be around 1.0 if data matches model)")


    # Test sample_matrix_variance_component
    n_obs_m = 200
    n_traits_m = 2
    true_R = np.array([[2.0, 0.5], [0.5, 1.0]])
    L = np.linalg.cholesky(true_R)
    res_multi = (L @ np.random.randn(n_traits_m, n_obs_m)).T # each row is an obs [res1, res2]

    data_arrays_m = [res_multi[:, i] for i in range(n_traits_m)]

    prior_df_m = float(n_traits_m + 2) # nu_0
    prior_scale_m = np.eye(n_traits_m) * 1.0 # Psi_0

    # Unconstrained
    samples_m = [sample_matrix_variance_component(data_arrays_m, n_obs_m, prior_df_m, prior_scale_m) for _ in range(2000)]
    mean_R_sampled = np.mean(samples_m, axis=0)
    print(f"Matrix variance (unconstrained): Mean=\n{mean_R_sampled}")
    # Expected mean = (Psi_0 + SSCP_data) / (nu_0 + n_obs - n_traits - 1)

    # Constrained
    prior_scale_m_diag = np.diag(np.diag(prior_scale_m)) # Ensure it's diagonal for test
    samples_m_diag = [sample_matrix_variance_component(data_arrays_m, n_obs_m, prior_df_m, prior_scale_m_diag, constraint=True) for _ in range(1000)]
    mean_R_sampled_diag = np.mean(samples_m_diag, axis=0)
    print(f"Matrix variance (constrained): Mean=\n{mean_R_sampled_diag}")
    self.assertTrue(np.allclose(mean_R_sampled_diag - np.diag(np.diag(mean_R_sampled_diag)), 0), "Constrained sample not diagonal")

# Need to import List, Optional, Union for type hints if not already there
from typing import List, Optional, Union
```
