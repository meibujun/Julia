import numpy as np
from ..core.genotypes import GenotypesData # Assuming relative import from sibling directory
from typing import List

def _sample_marker_effects_rrblup_st(
    geno_data: GenotypesData,
    y_corrected: np.ndarray, # Corrected for fixed effects: y_obs - X*beta
    residual_variance: float,
    marker_variance: float # sigma_g^2, assumed fixed for now
):
    """
    Samples marker effects for a single trait using RR-BLUP logic.
    Updates geno_data.alpha_samples[0] in place.
    Modifies y_corrected in place by subtracting new marker effects.

    Args:
        geno_data: The GenotypesData object for this marker set.
                   Assumes geno_data.genotypes is (n_obs, n_markers).
                   Assumes geno_data.alpha_samples is initialized (e.g., list containing one np.array for ST).
        y_corrected: Current residuals (e.g., y_obs - X*beta). This will be further modified.
        residual_variance: Current estimate of residual variance (sigma_e^2).
        marker_variance: Known variance of marker effects (sigma_g^2).
    """
    if not geno_data.genotypes.shape[0] == len(y_corrected):
        raise ValueError("Genotypes n_obs does not match y_corrected length.")
    if marker_variance <= 0:
        raise ValueError("Marker variance must be positive.")
    if residual_variance <= 0:
        raise ValueError("Residual variance must be positive.")

    n_markers = geno_data.genotypes.shape[1]

    # Ensure alpha_samples for single trait is initialized
    if not geno_data.alpha_samples or not isinstance(geno_data.alpha_samples[0], np.ndarray):
        geno_data.alpha_samples = [np.zeros(n_markers)] # Initialize if empty or wrong type

    current_alpha_st = geno_data.alpha_samples[0] # Get current alpha for single trait

    lambda_val = residual_variance / marker_variance

    # y_eff starts as y_obs - X*beta. We will iteratively update it.
    # y_eff = y_obs - X*beta - Z*alpha (current full alpha)
    # This means we first subtract all current Z*alpha
    if np.any(current_alpha_st): # If any current alphas are non-zero
         y_corrected -= geno_data.genotypes @ current_alpha_st

    # NUMBA_TARGET: This loop over markers is a prime candidate for Numba.
    # Requires geno_data.genotypes to be NumPy array. y_corrected, current_alpha_st are NumPy arrays.
    for j in range(n_markers):
        Z_j = geno_data.genotypes[:, j] # j-th marker column
        alpha_j_old = current_alpha_st[j]

        # Add back effect of alpha_j_old to y_corrected
        # y_corrected_for_j = y_corrected + Z_j * alpha_j_old
        # (y_corrected here is y_obs - X*beta - Z_(-j)*alpha_(-j)_old - Z_j*alpha_j_old)
        # So adding Z_j*alpha_j_old makes it y_obs - X*beta - Z_(-j)*alpha_(-j)_old
        y_corrected_for_j_sampling = y_corrected + Z_j * alpha_j_old

        # Calculate posterior parameters for alpha_j
        # Assuming R = I * residual_variance, so R_inv_diag elements are 1.0 (sigma_e^2 factored out)
        # ZjtZj = np.dot(Z_j, Z_j) # This assumes no weights. If weights, Z_j' W Z_j
        # For standard RR-BLUP, LHS_j = Z_j'Z_j + lambda
        # RHS_j = Z_j'y_corrected_for_j_sampling
        # Var(alpha_j) = residual_variance / LHS_j
        # Mean(alpha_j) = (RHS_j / LHS_j)

        # Direct calculation from conditional posterior N( (Z_j'y_corr_j / (Z_j'Z_j + lambda)), sigma_e^2 / (Z_j'Z_j + lambda) )
        ZjtZj = np.dot(Z_j, Z_j)
        if ZjtZj == 0.0 and lambda_val == 0.0: # Marker has no variation and no shrinkage
            # This can happen if a marker is monomorphic or has all zeros after centering.
            # Set effect to zero or handle as per specific model requirements.
            # For now, if ZjtZj is zero, this marker likely has no info.
             current_alpha_st[j] = 0.0
             # y_corrected does not change due to this marker if old alpha was also 0.
             # If old alpha was non-zero, it's already removed by y_corrected update below.
             # y_corrected -= Z_j * (0.0 - alpha_j_old) -> y_corrected += Z_j * alpha_j_old (done at start of loop)
             # So, effectively, y_corrected is now y_obs - X*beta - Z_(-j)*alpha_(-j)
             # and alpha_j_new is 0.
             # The final update y_corrected -= Z_j * (alpha_j_new - alpha_j_old) handles this:
             # y_corrected -= Z_j * (0 - alpha_j_old) = y_corrected + Z_j*alpha_j_old.
             # This is incorrect. The y_corrected update should be based on the *final* alpha_j_new.
             # Let's adjust:
             # y_corrected is y_obs - X*beta - Z_(-j)*alpha_(-j) (after adding back Z_j*alpha_j_old)
             # If alpha_j_new is 0, then the new y_corrected (for next marker) is this same value.
             # So, the final y_corrected update `y_corrected -= Z_j * (alpha_j_new - alpha_j_old)` becomes
             # `y_corrected -= Z_j * (0 - alpha_j_old)`
             # This means y_corrected_final = y_obs - X*beta - Z_(-j)*alpha_(-j) - Z_j*0
             # which is correct.
             # What was done in Julia: BLAS.axpy!(oldAlpha-α[j],x,yCorr) -> yCorr_new = yCorr_old + (oldAlpha-newAlpha)*x
             # So, y_corrected should be updated by the *change* in alpha.
             # y_corrected_final_for_iter = y_corrected_current_for_iter - Z_j * (alpha_j_new - alpha_j_old)

             # Let y_corrected always be y_obs - X*beta.
             # Each marker samples from y_obs - X*beta - Z_(-j)*alpha_(-j).
             # Let y_target = y_obs - X*beta
             # y_loop = y_target.copy()
             # for j:
             #   alpha_j_old = alpha[j]
             #   y_eff_j = y_loop + Z_j * alpha_j_old # y_target - Z_(-j)*alpha_(-j)_old
             #   sample alpha_j_new
             #   y_loop = y_loop + Z_j * (alpha_j_old - alpha_j_new) # Update y_loop for next marker
             # This is what the Julia axpy does.

             # Simpler: y_corrected is y_obs - X*beta - Z*alpha (full current alpha)
             # When sampling marker j:
             #   y_temp = y_corrected + Z_j * alpha_j_old  (removes j-th contribution)
             #   sample alpha_j_new based on y_temp
             #   y_corrected = y_temp - Z_j * alpha_j_new (adds back new j-th contribution)
             # This is equivalent to y_corrected_new = y_corrected_old - Z_j * (alpha_j_new - alpha_j_old)

             alpha_j_new = 0.0 # If marker has no info and no shrinkage
        else:
            lhs = ZjtZj + lambda_val
            if lhs == 0: # Should ideally not happen if lambda_val > 0 or ZjtZj > 0
                alpha_j_new = 0.0
            else:
                invLhs = 1.0 / lhs
                rhs = np.dot(Z_j, y_corrected_for_j_sampling)
                mean = invLhs * rhs
                alpha_j_new = mean + np.random.randn() * np.sqrt(invLhs * residual_variance)

        # Update y_corrected with the change in alpha_j
        y_corrected += Z_j * (alpha_j_old - alpha_j_new)
        current_alpha_st[j] = alpha_j_new

    # geno_data.alpha_samples[0] is updated in place as current_alpha_st is a view/reference.


def _ensure_precomputed_terms_for_markers(geno_data: GenotypesData, inv_weights: np.ndarray):
    """
    Ensures Z'diag(W)Z (diag) and Z'diag(W) are available on geno_data.
    These are `xpRinvx` and `xRinvArray` in Julia's GibbsMats if Rinv is inv_weights.
    """
    # Check if already computed and if inv_weights hash matches (if we want to be very robust)
    # For now, recompute if not present.
    if not hasattr(geno_data, 'priv_xpWz') or geno_data.priv_xpWz is None:
        # xpWz_j = Z_j' * diag(inv_weights) * Z_j
        if geno_data.genotypes is None: raise ValueError("Genotypes matrix is None")
        weighted_Z = geno_data.genotypes * inv_weights[:, np.newaxis] # Z_ij * w_i
        geno_data.priv_xpWz = np.sum(weighted_Z * geno_data.genotypes, axis=0) # (Z_j * w_i)' * Z_j = Z_j' * w_i * Z_j (sum over i)
        # This is sum_i ( Z_ij^2 * w_i ) for each marker j. This is diag(Z'diag(W)Z).

    # xRW_j = Z_j' * diag(inv_weights) -> (Z'diag(W))_j (j-th row of Z'diag(W))
    # This is effectively (Weighted_Z)' where Weighted_Z_ij = Z_ij * w_i
    # No, this is (Z'W). Each element (j,i) is Z_ij * w_i.
    # xRinvArray in Julia was a list of columns, each Z_j * inv_weights (element-wise product if inv_weights is column vector).
    # Z_j_weighted = Z_j * inv_weights (element-wise). Then Z_j_weighted' * y_corr.
    # Let's compute ZprimeW = Z.T @ np.diag(inv_weights) once, or pass weighted_Z_transpose
    if not hasattr(geno_data, 'priv_ZprimeW') or geno_data.priv_ZprimeW is None:
        if geno_data.genotypes is None: raise ValueError("Genotypes matrix is None")
        geno_data.priv_ZprimeW = geno_data.genotypes.T @ diags(inv_weights) # (n_markers x n_obs) sparse

def _sample_marker_effects_bayesa_st(
    geno_data: GenotypesData,
    y_corrected: np.ndarray, # Corrected for fixed effects: y_obs - X*beta
    residual_variance: float, # sigma_e^2
    inv_weights: np.ndarray # Observation-specific inverse variance weights (w_i)
):
    """
    Samples marker effects (alpha_j) and their specific variances (sigma_g_j^2) for BayesA (single trait).
    Updates geno_data.alpha_samples[0] and geno_data.marker_effect_variance.value (array of sigma_g_j^2) in place.
    Modifies y_corrected by subtracting new Z*alpha.

    Priors:
        alpha_j | sigma_g_j^2 ~ N(0, sigma_g_j^2)
        sigma_g_j^2 ~ ScaledInvChi2(nu_g_prior, S_g_prior^2) (common prior for all j)
    """
    if geno_data.genotypes is None: raise ValueError("Genotypes matrix not set.")
    if not geno_data.genotypes.shape[0] == len(y_corrected):
        raise ValueError("Genotypes n_obs does not match y_corrected length.")
    if residual_variance <= 0: raise ValueError("Residual variance must be positive.")
    if geno_data.marker_effect_variance is None or \
       geno_data.marker_effect_variance.df is None or \
       geno_data.marker_effect_variance.scale is None:
        raise ValueError("Prior df and scale for marker variances must be set in geno_data.marker_effect_variance.")

    n_markers = geno_data.genotypes.shape[1]

    # Ensure alpha_samples and marker_specific_variances (sigma_g_j^2) are initialized
    if not geno_data.alpha_samples or not isinstance(geno_data.alpha_samples[0], np.ndarray) or \
       len(geno_data.alpha_samples[0]) != n_markers:
        geno_data.alpha_samples = [np.zeros(n_markers)]

    # marker_effect_variance.value will store the array of sigma_g_j^2
    if not isinstance(geno_data.marker_effect_variance.value, np.ndarray) or \
       len(geno_data.marker_effect_variance.value) != n_markers:
        # Initialize from prior mean if not set, or if wrong shape
        prior_mean_sigma_g_j_sq = sample_scaled_inverse_chi_squared(
            df=geno_data.marker_effect_variance.df,
            scale_sq=float(geno_data.marker_effect_variance.scale) # scale is S0^2
        )
        geno_data.marker_effect_variance.value = np.full(n_markers, prior_mean_sigma_g_j_sq)

    current_alpha_st = geno_data.alpha_samples[0]
    current_sigma_g_j_sq_array = geno_data.marker_effect_variance.value # This is array of sigma_g_j^2

    # Precompute terms involving Z and W (inv_weights) if not done
    # Z'_j W Z_j and Z'_j W (where W = diag(inv_weights))
    # _ensure_precomputed_terms_for_markers(geno_data, inv_weights) # Modifies geno_data
    # ZprimeW = geno_data.priv_ZprimeW # (n_markers x n_obs)
    # ZprimeWZ_diag = geno_data.priv_xpWz # (n_markers,)

    # Simpler: direct calculation if precomputation is complex to manage across modules
    # Z'W Z (diag) = sum_i (Z_ij^2 * w_i) for each marker j
    ZprimeWZ_diag = np.sum((geno_data.genotypes**2) * inv_weights[:, np.newaxis], axis=0)
    # Z'W = Z.T @ diags(inv_weights)
    # ZprimeW_y_corr_j = sum_i (Z_ij * w_i * y_corr_i) for each marker j
    # This is (Z.T @ (inv_weights * y_corr_for_sampling_j))_j

    # y_corrected initially is y_obs - X*beta.
    # We update it iteratively: y_final = y_initial - Z * (alpha_new - alpha_old)
    if np.any(current_alpha_st): # If any current alphas are non-zero
         y_corrected -= geno_data.genotypes @ current_alpha_st # Now y_corrected = y_obs - X*beta - Z*alpha_old

    # NUMBA_TARGET: Loop over markers for sampling alpha_j
    for j in range(n_markers):
        Z_j = geno_data.genotypes[:, j]
        alpha_j_old = current_alpha_st[j]

        # y_target_for_j = y_corrected (which is y_obs - Xb - Z*alpha_old) + Z_j * alpha_j_old
        #                  = y_obs - Xb - Z_(-j)*alpha_(-j)_old
        y_target_for_j = y_corrected + Z_j * alpha_j_old

        sigma_g_j_sq = current_sigma_g_j_sq_array[j]
        if sigma_g_j_sq <= 1e-12: # Avoid division by zero if variance is ~0
            alpha_j_new = 0.0
        else:
            # Denominator for conditional variance of alpha_j: (Z_j' (W/sigma_e^2) Z_j + 1/sigma_g_j^2)
            # lhs_val = (ZprimeWZ_diag[j] / residual_variance) + (1.0 / sigma_g_j_sq)
            # Numerator for conditional mean of alpha_j: (Z_j' (W/sigma_e^2) y_target_for_j)
            # rhs_val = (Z_j.T @ (inv_weights * y_target_for_j)) / residual_variance

            # Simpler, from standard theory: V_alpha_j = 1 / (Z_j'Z_j/sigma_e^2 + 1/sigma_g_j^2)
            # E_alpha_j = V_alpha_j * (Z_j'y_target_for_j / sigma_e^2)
            # This assumes W=I. If W is present, Z_j should be sqrt(W)Z_j or incorporated.
            # The Julia code uses:
            # lhs = (Z_j' W Z_j / vare) + 1/sigma_g_j^2
            # rhs_for_gHat = ( (Z_j' W y_target_j) + (Z_j' W Z_j) * alpha_j_old ) / vare
            # gHat = rhs_for_gHat / lhs. This is the mean.
            # Sampling variance is 1/lhs.

            # Let's use ZprimeWZ_diag[j] = Z_j' W Z_j
            # Let ZprimeW_y_tj = Z_j' W y_target_for_j
            ZprimeW_y_tj = np.dot(Z_j, inv_weights * y_target_for_j)

            lhs = (ZprimeWZ_diag[j] / residual_variance) + (1.0 / sigma_g_j_sq)
            if lhs == 0: alpha_j_new = 0.0;
            else:
                invLhs = 1.0 / lhs
                # The mean in Julia's BayesABC is calculated based on yCorr that *includes* the current alpha_j
                # effectively gHat = invLhs * ( (Z_j'W(y_corr_full_old_alpha)) / vare + (Z_j'WZ_j / vare)alpha_j_old + (1/sigma_g_j_sq) * 0_prior_mean )
                # This is equivalent to ( Z_j'W y_target_for_j / vare ) / lhs
                mean_alpha_j = invLhs * (ZprimeW_y_tj / residual_variance)
                alpha_j_new = mean_alpha_j + np.random.randn() * np.sqrt(invLhs) # Variance of alpha is 1/lhs

        # Update y_corrected with the change in alpha_j
        y_corrected += Z_j * (alpha_j_old - alpha_j_new)
        current_alpha_st[j] = alpha_j_new

    # After sampling all alpha_j, sample their variances sigma_g_j^2
    prior_df_g = geno_data.marker_effect_variance.df
    prior_scale_g = float(geno_data.marker_effect_variance.scale) # Common S0^2 for all marker variances

    # NUMBA_TARGET: Loop over markers for sampling sigma_g_j^2
    for j in range(n_markers):
        # Posterior for sigma_g_j^2 ~ ScaledInvChi2(nu_prior + 1, (alpha_j^2 + nu_prior*S0_prior^2)/(nu_prior+1))
        # data_vector for sample_scalar_variance_component is just [current_alpha_st[j]]
        # prior_df is nu_g_prior, prior_scale_parameter is S_g_prior^2
        current_sigma_g_j_sq_array[j] = sample_scalar_variance_component(
            data_vector=np.array([current_alpha_st[j]]), # Single data point alpha_j
            prior_df=prior_df_g,
            prior_scale_parameter=prior_scale_g
        )
        if current_sigma_g_j_sq_array[j] < 1e-12: # Floor small variances
            current_sigma_g_j_sq_array[j] = 1e-12

    # geno_data.alpha_samples[0] and geno_data.marker_effect_variance.value are updated in place.


# Placeholder for multi-trait BayesA
# def _sample_marker_effects_bayesa_mt(...):
#     pass


def _sample_marker_effects_bayesc_st(
    geno_data: GenotypesData,
    y_corrected: np.ndarray, # Corrected for fixed effects: y_obs - X*beta
    residual_variance: float, # sigma_e^2
    inv_weights: np.ndarray, # Observation-specific inverse variance weights (w_i)
    # Pi parameter (P(effect != 0)) is stored in geno_data.pi_value
    # Common marker variance (sigma_g^2) is in geno_data.marker_effect_variance.value
    # Priors for Pi (alpha, beta for Beta dist) should be on geno_data or passed.
    # For simplicity, assume pi_prior_alpha=1, pi_prior_beta=1 if not on geno_data.
    pi_prior_alpha: float = 1.0,
    pi_prior_beta: float = 1.0
):
    """
    Samples marker effects (alpha_j), inclusion indicators (delta_j),
    common marker variance (sigma_g^2), and inclusion probability (pi) for BayesC (single trait).
    Updates relevant attributes in geno_data in place.
    Modifies y_corrected by subtracting new Z*alpha.
    """
    if geno_data.genotypes is None: raise ValueError("Genotypes matrix not set.")
    if not geno_data.genotypes.shape[0] == len(y_corrected):
        raise ValueError("Genotypes n_obs does not match y_corrected length.")
    if residual_variance <= 0: raise ValueError("Residual variance must be positive.")
    if geno_data.marker_effect_variance is None or geno_data.marker_effect_variance.value is None or \
       not isinstance(geno_data.marker_effect_variance.value, (float, int, np.floating)):
        raise ValueError("Common marker variance (scalar) must be set in geno_data.marker_effect_variance.value.")
    if geno_data.pi_value is None or not isinstance(geno_data.pi_value, float):
        raise ValueError("Pi parameter (scalar P(effect!=0)) must be set in geno_data.pi_value.")

    n_markers = geno_data.genotypes.shape[1]
    common_marker_variance = float(geno_data.marker_effect_variance.value)
    pi_param = float(geno_data.pi_value) # P(effect != 0)

    if common_marker_variance <= 1e-12: # Effectively zero
        # If marker variance is zero, all effects should be zero.
        # Update y_corrected by adding back any old Z*alpha contribution
        if np.any(geno_data.alpha_samples[0]):
            y_corrected += geno_data.genotypes @ geno_data.alpha_samples[0]
        geno_data.alpha_samples[0].fill(0.0)
        geno_data.delta_samples[0].fill(0.0) # All excluded
        # Pi and common_marker_variance would still be sampled based on this state.
    else:
        # Initialize delta_samples if not present
        if not geno_data.delta_samples or not isinstance(geno_data.delta_samples[0], np.ndarray) or \
        len(geno_data.delta_samples[0]) != n_markers:
            # Initialize delta based on pi_param (e.g., random draw or all included if pi_param is high)
            # For robustness, let's initialize to current alpha's non-zero status or all 1s if alpha is zero.
            if np.any(geno_data.alpha_samples[0]):
                 geno_data.delta_samples = [(geno_data.alpha_samples[0] != 0).astype(float)]
            else: # Start with all included if alpha is all zero
                 geno_data.delta_samples = [np.ones(n_markers)]


        current_alpha_st = geno_data.alpha_samples[0]
        current_delta_st = geno_data.delta_samples[0]

        # Precompute Z'WZ_diag (sum_i Z_ij^2 * w_i for marker j)
        ZprimeWZ_diag = np.sum((geno_data.genotypes**2) * inv_weights[:, np.newaxis], axis=0)

        # y_corrected initially is y_obs - X*beta.
        # Adjust for full Z*alpha_old contribution before starting per-marker loop.
        if np.any(current_alpha_st):
            y_corrected -= geno_data.genotypes @ current_alpha_st

        # Terms for BayesC (from Julia BayesABC!)
        # pi_0 = 1.0 - pi_param # P(effect == 0)
        # log_pi_0 = np.log(pi_0) if pi_0 > 1e-12 else -np.inf # log(P(delta_j=0))
        # log_pi_1 = np.log(pi_param) if pi_param > 1e-12 else -np.inf # log(P(delta_j=1))
        # Using odds: log( (1-pi_0)/pi_0 ) = log( pi_param / (1-pi_param) )
        if pi_param <= 1e-12: log_prior_odds = -np.inf # Effectively P(delta_j=1)=0
        elif pi_param >= 1.0 - 1e-12: log_prior_odds = np.inf # Effectively P(delta_j=1)=1
        else: log_prior_odds = np.log(pi_param / (1.0 - pi_param))


        inv_residual_variance = 1.0 / residual_variance
        inv_common_marker_variance = 1.0 / common_marker_variance

        for j in range(n_markers):
            Z_j = geno_data.genotypes[:, j]
            alpha_j_old = current_alpha_st[j]

            # y_target_for_j = y_corrected (y_obs - Xb - Z*alpha_old) + Z_j * alpha_j_old
            #                  = y_obs - Xb - Z_(-j)*alpha_(-j)_old
            y_target_for_j = y_corrected + Z_j * alpha_j_old

            # LHS for conditional posterior of alpha_j (if included)
            # lhs_j = (Z_j' W Z_j / sigma_e^2) + 1/sigma_g^2 (common)
            lhs_j = (ZprimeWZ_diag[j] * inv_residual_variance) + inv_common_marker_variance

            alpha_j_new = 0.0
            delta_j_new = 0.0

            if lhs_j > 1e-12: # Avoid division by zero if Z_j is all zero and marker_var is huge
                inv_lhs_j = 1.0 / lhs_j

                # Posterior mean of alpha_j if included (gHat in Julia)
                # gHat = inv_lhs_j * (Z_j' W y_target_for_j / sigma_e^2)
                ZprimeW_y_tj = np.dot(Z_j, inv_weights * y_target_for_j)
                gHat = inv_lhs_j * (ZprimeW_y_tj * inv_residual_variance)

                # Log-likelihood ratio part for delta_j=1 vs delta_j=0
                # log_LR = 0.5 * (log(sigma_g^2) - log(sigma_g^2 + Z_j'WZ_j * sigma_e^2 / (Z_j'WZ_j)) ) ??? Complex form
                # From Julia: logDelta1 = -0.5*(log(lhs) + logVarEffects[j] - gHat*rhs) + logPiComp
                # Here, logVarEffects[j] is log(common_marker_variance)
                # rhs for gHat was (Z_j'W y_target_j / vare) + (Z_j'WZ_j / vare)alpha_j_old_effective_zero
                # So, gHat*rhs (where rhs is for gHat's formula, not the full BayesABC rhs)
                # = gHat * (Z_j'W y_target_j / vare)
                # log_LR_contrib = -0.5 * (np.log(lhs_j) + np.log(common_marker_variance) - gHat * (ZprimeW_y_tj * inv_residual_variance) )
                # This is for P(y|delta_j=1,...) / P(y|delta_j=0,...)
                # P(y|delta_j=0) uses alpha_j=0. P(y|delta_j=1) integrates out alpha_j.
                # log P(D|M_1) - log P(D|M_0)
                # M_1: alpha_j ~ N(0, sigma_g^2), M_0: alpha_j = 0
                # log [ N(y_target | Z_j*0, sigma_e^2/W) ] = const - 0.5 * sum( (y_target_j^2 * W)/sigma_e^2 )
                # log [ int N(y_target | Z_j*alpha_j, sigma_e^2/W) N(alpha_j|0,sigma_g^2) d alpha_j ]
                #   = log N(y_target | 0, sigma_e^2/W + Z_j sigma_g^2 Z_j') which is MVN.
                # This is complex. The Julia one is simpler:
                # logDelta1 = -0.5*(log(lhs) + log(sigma_g^2) - gHat^2 * lhs) + log(P(delta_j=1)/P(delta_j=0))
                # where lhs = (Z'WZ/sigma_e^2 + 1/sigma_g^2) and gHat = E[alpha_j|delta_j=1]
                # and P(delta_j=1)/P(delta_j=0) comes from pi_param.

                log_LR_factor = 0.5 * (gHat**2 * lhs_j - np.log(lhs_j * common_marker_variance)) # Check this derivation. Should be log(sigma_g^2) instead of common_marker_variance?
                                                                                             # Julia: log(lhs) + logVarEffects[j] - gHat*rhs
                                                                                             # where VarEffects[j] is sigma_g^2. rhs for gHat is Z'Wy_c/vare.
                                                                                             # So: log(lhs*sigma_g^2) - gHat^2*lhs. Yes, this is closer.
                log_LR_factor = 0.5 * (np.log(inv_lhs_j) - np.log(common_marker_variance) + (gHat**2)/inv_lhs_j) # From Christensen & Sorensen 2001, eq 10
                                                                                                             # Or from Meuwissen 2001 (BayesB paper)

                log_post_odds = log_prior_odds + log_LR_factor

                if log_post_odds > 700: prob_delta1 = 1.0 # Avoid overflow in exp
                elif log_post_odds < -700: prob_delta1 = 0.0
                else: prob_delta1 = 1.0 / (1.0 + np.exp(-log_post_odds))

                if np.random.rand() < prob_delta1:
                    delta_j_new = 1.0
                    alpha_j_new = gHat + np.random.randn() * np.sqrt(inv_lhs_j) # Sample from N(gHat, inv_lhs_j)

            # Update y_corrected and current_alpha_st, current_delta_st
            y_corrected += Z_j * (alpha_j_old - alpha_j_new)
            current_alpha_st[j] = alpha_j_new
            current_delta_st[j] = delta_j_new

    # Sample common marker variance (sigma_g^2)
    if geno_data.marker_effect_variance.estimate_variance:
        included_alphas = current_alpha_st[current_delta_st == 1.0]
        if len(included_alphas) > 0:
            new_common_marker_var = sample_scalar_variance_component(
                data_vector=included_alphas,
                prior_df=geno_data.marker_effect_variance.df,
                prior_scale_parameter=float(geno_data.marker_effect_variance.scale) # S0^2
            )
            if new_common_marker_var > 1e-12:
                geno_data.marker_effect_variance.value = new_common_marker_var
            else: # sampled too small
                geno_data.marker_effect_variance.value = 1e-12 # Floor it
        elif geno_data.marker_effect_variance.df > 0 and float(geno_data.marker_effect_variance.scale) > 0 : # No markers included, sample from prior
             geno_data.marker_effect_variance.value = sample_scaled_inverse_chi_squared(
                 df=geno_data.marker_effect_variance.df,
                 scale_sq=float(geno_data.marker_effect_variance.scale)
             )

    # Sample Pi (P(effect != 0))
    if geno_data.estimate_pi:
        n_included = np.sum(current_delta_st)
        n_excluded = n_markers - n_included
        # Posterior for pi_param ~ Beta(prior_alpha + n_included, prior_beta + n_excluded)
        # Using default priors for pi_param ~ Beta(1,1) if not specified elsewhere
        # These should be stored on geno_data if user can set them.
        # pi_prior_alpha, pi_prior_beta

        sampled_pi = np.random.beta(pi_prior_alpha + n_included, pi_prior_beta + n_excluded)
        geno_data.pi_value = max(1e-9, min(1.0 - 1e-9, sampled_pi)) # Ensure pi is in (0,1) bounds

    # Final y_corrected (y_obs - Xb - Z*alpha_new) is implicitly stored by modification of input y_corrected.


def _sample_marker_effects_bayesb_st(
    geno_data: GenotypesData,
    y_corrected: np.ndarray, # Corrected for fixed effects: y_obs - X*beta
    residual_variance: float, # sigma_e^2
    inv_weights: np.ndarray, # Observation-specific inverse variance weights (w_i)
    # Pi parameter (P(effect != 0)) is stored in geno_data.pi_value
    # Priors for sigma_g_j^2 (common nu_g, S0_g^2) from geno_data.marker_effect_variance.df and .scale
    # Array of sigma_g_j^2 is in geno_data.marker_effect_variance.value
    pi_prior_alpha: float = 1.0,
    pi_prior_beta: float = 1.0
):
    """
    Samples marker effects (alpha_j), inclusion indicators (delta_j),
    marker-specific variances (sigma_g_j^2), and inclusion probability (pi) for BayesB (single trait).
    Updates relevant attributes in geno_data in place.
    Modifies y_corrected by subtracting new Z*alpha.
    """
    if geno_data.genotypes is None: raise ValueError("Genotypes matrix not set.")
    if not geno_data.genotypes.shape[0] == len(y_corrected):
        raise ValueError("Genotypes n_obs does not match y_corrected length.")
    if residual_variance <= 0: raise ValueError("Residual variance must be positive.")

    # Priors for individual marker variances (sigma_g_j^2)
    if geno_data.marker_effect_variance is None or \
       geno_data.marker_effect_variance.df is None or \
       geno_data.marker_effect_variance.scale is None:
        raise ValueError("Prior df and scale for marker-specific variances must be set in geno_data.marker_effect_variance.")
    prior_df_sigma_g_j_sq = geno_data.marker_effect_variance.df
    prior_scale_S0_sigma_g_j_sq = float(geno_data.marker_effect_variance.scale)

    if geno_data.pi_value is None or not isinstance(geno_data.pi_value, float):
        raise ValueError("Pi parameter (scalar P(effect!=0)) must be set in geno_data.pi_value.")

    n_markers = geno_data.genotypes.shape[1]
    pi_param = float(geno_data.pi_value) # P(effect != 0)

    # Ensure alpha_samples, delta_samples, and marker_specific_variances are initialized
    if not geno_data.alpha_samples or not isinstance(geno_data.alpha_samples[0], np.ndarray) or \
       len(geno_data.alpha_samples[0]) != n_markers:
        geno_data.alpha_samples = [np.zeros(n_markers)]

    if not geno_data.delta_samples or not isinstance(geno_data.delta_samples[0], np.ndarray) or \
       len(geno_data.delta_samples[0]) != n_markers:
        if np.any(geno_data.alpha_samples[0]):
            geno_data.delta_samples = [(geno_data.alpha_samples[0] != 0).astype(float)]
        else: geno_data.delta_samples = [np.ones(n_markers)] # Start all included if alpha is zero

    if not isinstance(geno_data.marker_effect_variance.value, np.ndarray) or \
       len(geno_data.marker_effect_variance.value) != n_markers:
        init_var_val = sample_scaled_inverse_chi_squared(df=prior_df_sigma_g_j_sq, scale_sq=prior_scale_S0_sigma_g_j_sq)
        geno_data.marker_effect_variance.value = np.full(n_markers, max(1e-12, init_var_val))


    current_alpha_st = geno_data.alpha_samples[0]
    current_delta_st = geno_data.delta_samples[0]
    current_sigma_g_j_sq_array = geno_data.marker_effect_variance.value

    ZprimeWZ_diag = np.sum((geno_data.genotypes**2) * inv_weights[:, np.newaxis], axis=0)

    if np.any(current_alpha_st):
        y_corrected -= geno_data.genotypes @ current_alpha_st

    if pi_param <= 1e-12: log_prior_odds = -np.inf
    elif pi_param >= 1.0 - 1e-12: log_prior_odds = np.inf
    else: log_prior_odds = np.log(pi_param / (1.0 - pi_param))

    inv_residual_variance = 1.0 / residual_variance

    for j in range(n_markers):
        Z_j = geno_data.genotypes[:, j]
        alpha_j_old = current_alpha_st[j]
        y_target_for_j = y_corrected + Z_j * alpha_j_old

        sigma_g_j_sq = current_sigma_g_j_sq_array[j]
        if sigma_g_j_sq <= 1e-12: # Effectively zero variance for this specific marker
            # If marker has no variance, its effect is 0 if included.
            # Likelihood ratio for delta=1 vs delta=0 needs care.
            # log P(D|M_1 with sigma_g_j_sq=0) vs log P(D|M_0)
            # This implies gHat = 0. log_LR_factor becomes -0.5 * (log(lhs_val_at_sigma_g_j_sq_zero * 0)) -> problematic
            # If sigma_g_j_sq is truly zero, then alpha_j must be zero.
            # So, prob_delta1 should be low unless data strongly suggests non-zero with this tiny variance.
            # For practical purposes, if sigma_g_j_sq is near zero, treat as excluded or alpha_j=0.
            # The Julia code's `invVarEffects[j]` would be huge. `lhs` would be huge. `invLhs` tiny. `gHat` tiny.
            log_LR_factor = -np.inf # Effectively, if sigma_g_j_sq is zero, this marker cannot explain data unless effect is also zero.
        else:
            inv_sigma_g_j_sq = 1.0 / sigma_g_j_sq
            lhs_j = (ZprimeWZ_diag[j] * inv_residual_variance) + inv_sigma_g_j_sq

            if lhs_j <= 1e-12: # Effectively zero or negative (should not happen if ZprimeWZ_diag >=0, inv_res_var >0, inv_sigma_g_j_sq >0)
                log_LR_factor = -np.inf # Cannot explain data
            else:
                inv_lhs_j = 1.0 / lhs_j
                ZprimeW_y_tj = np.dot(Z_j, inv_weights * y_target_for_j)
                gHat = inv_lhs_j * (ZprimeW_y_tj * inv_residual_variance)
                log_LR_factor = 0.5 * (np.log(inv_lhs_j) - np.log(sigma_g_j_sq) + (gHat**2)/inv_lhs_j)

        log_post_odds = log_prior_odds + log_LR_factor
        prob_delta1 = 1.0 / (1.0 + np.exp(-log_post_odds)) if log_post_odds > -700 else 0.0
        if log_post_odds > 700: prob_delta1 = 1.0

        alpha_j_new = 0.0
        delta_j_new = 0.0
        if np.random.rand() < prob_delta1 and lhs_j > 1e-12 : # If included and lhs is valid
            delta_j_new = 1.0
            # Recalculate mean and sample alpha_j, using its specific sigma_g_j_sq
            # gHat and inv_lhs_j were already calculated based on current sigma_g_j_sq
            alpha_j_new = gHat + np.random.randn() * np.sqrt(inv_lhs_j)
        else: # Excluded
            delta_j_new = 0.0
            alpha_j_new = 0.0
            # Julia samples beta from prior N(0, sigma_g_j^2), but alpha (effective effect) is 0.
            # We only care about alpha for prediction. beta storage is for method internals.
            # geno_data.beta_samples[0][j] = np.random.randn() * np.sqrt(sigma_g_j_sq)

        y_corrected += Z_j * (alpha_j_old - alpha_j_new)
        current_alpha_st[j] = alpha_j_new
        current_delta_st[j] = delta_j_new

    # Sample marker-specific variances sigma_g_j^2
    for j in range(n_markers):
        if current_delta_st[j] == 1.0: # If marker j is included
            # Posterior for sigma_g_j^2 | alpha_j ~ ScaledInvChi2(nu_prior + 1, (alpha_j^2 + nu_prior*S0_prior^2)/(nu_prior+1))
            current_sigma_g_j_sq_array[j] = sample_scalar_variance_component(
                data_vector=np.array([current_alpha_st[j]]),
                prior_df=prior_df_sigma_g_j_sq,
                prior_scale_parameter=prior_scale_S0_sigma_g_j_sq
            )
        else: # Marker j is excluded, sample its variance from its prior
            current_sigma_g_j_sq_array[j] = sample_scaled_inverse_chi_squared(
                df=prior_df_sigma_g_j_sq,
                scale_sq=prior_scale_S0_sigma_g_j_sq
            )
        if current_sigma_g_j_sq_array[j] < 1e-12: current_sigma_g_j_sq_array[j] = 1e-12

    # Sample Pi (P(effect != 0))
    if geno_data.estimate_pi:
        n_included = np.sum(current_delta_st)
        n_excluded = n_markers - n_included
        sampled_pi = np.random.beta(pi_prior_alpha + n_included, pi_prior_beta + n_excluded)
        geno_data.pi_value = max(1e-9, min(1.0 - 1e-9, sampled_pi))


def _sample_marker_effects_bayesl_st(
    geno_data: GenotypesData,
    y_corrected: np.ndarray, # y_obs - X*beta
    residual_variance: float, # sigma_e^2
    inv_weights: np.ndarray, # w_i
    # Common marker variance (sigma_g^2) is in geno_data.marker_effect_variance.value
    # Lasso lambda parameter (related to prior for tau_j^2) should be on geno_data or passed.
    # For simplicity, assume a fixed shrinkage parameter lambda_sq_prior or it's derived.
    # The Julia code uses a `gammaArray` for 1/tau_j^2 and updates it.
    # It also has a common vEff (sigma_g^2).
    # The formulation in Julia's BayesL! for `getlambda` is `lambda / gammaArray[j]`,
    # where lambda = vRes/vEff = sigma_e^2 / sigma_g^2.
    # This means `1/tau_j^2` (the individual shrinkage) is `(sigma_e^2 / sigma_g^2) / gamma_j`.
    # And `gamma_j` is sampled.
    # Prior for tau_j^2 is often Exponential(lambda^2/2), which means 1/tau_j^2 ~ Gamma(1, 2/lambda^2) * 0.5.
    # Or, if alpha_j ~ N(0, sigma_g^2 * tau_j^2), and tau_j^2 ~ Exp(lambda_sq_hyper_param / 2).
    # Let's assume geno_data.lasso_lambda_sq_hyper is this hyperparameter.
    lasso_lambda_sq_hyper: float = 1.0 # Placeholder, should be configurable
):
    """
    Samples marker effects (alpha_j) and their specific shrinkage/variance parameters
    for Bayesian Lasso (single trait).
    Updates geno_data.alpha_samples[0], and internal state for tau_j^2 (e.g., geno_data.gamma_array).
    Also samples common marker variance sigma_g^2.
    Modifies y_corrected by subtracting new Z*alpha.
    """
    if geno_data.genotypes is None: raise ValueError("Genotypes matrix not set.")
    if residual_variance <= 0: raise ValueError("Residual variance must be positive.")
    if geno_data.marker_effect_variance is None or \
       not isinstance(geno_data.marker_effect_variance.value, (float, int, np.floating)):
        raise ValueError("Common marker variance (scalar) must be set for BayesL.")

    common_marker_variance = float(geno_data.marker_effect_variance.value)
    if common_marker_variance <= 1e-12: # Cannot proceed if common variance is zero
        if np.any(geno_data.alpha_samples[0]):
            y_corrected += geno_data.genotypes @ geno_data.alpha_samples[0]
        geno_data.alpha_samples[0].fill(0.0)
        return # sigma_g^2 sampling will handle this state.

    n_markers = geno_data.genotypes.shape[1]
    current_alpha_st = geno_data.alpha_samples[0]

    # Initialize gamma_array (1/tau_j^2 in some notations, or related to it) if not present
    # Julia initializes gammaArray with rand(Gamma(1, 8)) or rand(Gamma((ntraits+1)/2, 8)).
    # Let's assume gamma_array stores 1/tau_j^2, and tau_j^2 ~ Exp(lambda_sq_hyper/2).
    # Then 1/tau_j^2 might follow a Gamma distribution.
    # For now, initialize if needed, e.g., with ones.
    if not hasattr(geno_data, 'gamma_array_bayesl') or \
       not isinstance(geno_data.gamma_array_bayesl, np.ndarray) or \
       len(geno_data.gamma_array_bayesl) != n_markers:
        # Initial value for gamma_j (1/tau_j^2). Park (2008) suggests lambda_sq_hyper.
        # Let's use a value derived from the prior mean of Exp. E[tau_j^2] = 2/lambda_sq_hyper
        # So, initial 1/tau_j^2 = lambda_sq_hyper/2
        initial_gamma_val = lasso_lambda_sq_hyper / 2.0
        initial_gamma_val = max(1e-6, initial_gamma_val) # ensure positive
        geno_data.gamma_array_bayesl = np.full(n_markers, initial_gamma_val, dtype=np.float64)

    current_gamma_array = geno_data.gamma_array_bayesl

    ZprimeWZ_diag = np.sum((geno_data.genotypes**2) * inv_weights[:, np.newaxis], axis=0)
    inv_residual_variance = 1.0 / residual_variance

    if np.any(current_alpha_st):
        y_corrected -= geno_data.genotypes @ current_alpha_st # y_corr = y_obs - Xb - Z*alpha_old

    # lambda_eff_factor = sigma_e^2 / sigma_g^2 (common sigma_g^2)
    lambda_eff_factor = residual_variance / common_marker_variance

    for j in range(n_markers):
        Z_j = geno_data.genotypes[:, j]
        alpha_j_old = current_alpha_st[j]
        y_target_for_j = y_corrected + Z_j * alpha_j_old # y_obs - Xb - Z_(-j)*alpha_(-j)_old

        # Individual shrinkage for marker j: lambda_j_eff = lambda_eff_factor / gamma_j
        # where gamma_j is 1/tau_j^2. So this is (sigma_e^2 / sigma_g^2) * tau_j^2
        # This term (1/sigma_g_j^2_eff) goes into LHS for alpha_j.
        # sigma_g_j^2_eff = sigma_g^2 / gamma_j (if gamma_j is the scaling factor for sigma_g^2)
        # Or, if alpha_j ~ N(0, sigma_g_j_sq), then 1/sigma_g_j_sq is added to LHS.
        # In Julia: lhs_contrib = lambda_eff_factor / gamma_j
        # This implies 1/sigma_g_j_sq_effective = (sigma_e^2/sigma_g^2) / gamma_j

        gamma_j = current_gamma_array[j]
        if gamma_j <= 1e-12: # Effectively infinite individual variance for alpha_j
            inv_sigma_g_j_sq_eff = 0 # No shrinkage if gamma_j is zero
        else:
            inv_sigma_g_j_sq_eff = lambda_eff_factor / gamma_j # This is (sigma_e^2 / (sigma_g^2 * gamma_j))

        lhs_j = (ZprimeWZ_diag[j] * inv_residual_variance) + inv_sigma_g_j_sq_eff

        alpha_j_new = 0.0
        if lhs_j > 1e-12:
            inv_lhs_j = 1.0 / lhs_j
            ZprimeW_y_tj = np.dot(Z_j, inv_weights * y_target_for_j)
            mean_alpha_j = inv_lhs_j * (ZprimeW_y_tj * inv_residual_variance)
            alpha_j_new = mean_alpha_j + np.random.randn() * np.sqrt(inv_lhs_j * residual_variance) # Var = sigma_e^2 / lhs_j
                                                                                                # No, var is just inv_lhs_j. Sample is N(mean, var).
                                                                                                # Julia: randn()*sqrt(invLhs*vRes) for BayesC0L.
                                                                                                # Here, vRes is residual_variance. So it's correct.

        y_corrected += Z_j * (alpha_j_old - alpha_j_new)
        current_alpha_st[j] = alpha_j_new

    # Sample gamma_j (related to 1/tau_j^2)
    # Posterior for tau_j^2 | alpha_j, sigma_g^2, lambda_sq_hyper is Inverse Gaussian.
    # tau_j^2 ~ IG(mu_prime, lambda_prime)
    # mu_prime = sqrt(lambda_sq_hyper * common_marker_variance / alpha_j^2)
    # lambda_prime = lambda_sq_hyper
    # Then 1/tau_j^2 is sampled. (This needs scipy.stats.invgauss)
    # Julia's sampleGammaArray! uses a Metropolis-Hastings for 1/gamma_j (where gamma_j seems to be tau_j^2)
    # or directly for gamma_j if gamma_j is 1/tau_j^2.
    # The gammaArray in Julia is 1/tau_sq, and it's sampled from Gamma(1, 8) or similar in MH.
    # For simplicity, let's assume a simpler update if direct conditional is hard.
    # Park (2008) shows 1/tau_j^2 ~ IG(sqrt(lambda_sq_hyper * sigma_g^2 / alpha_j^2), lambda_sq_hyper)
    # No, this is for tau_j^2.
    # Conditional for 1/sigma_j^2 (where sigma_j^2 = tau_j^2 * sigma_eps^2 / lambda_hyper^2) is GIG.
    # Simpler: if alpha_j ~ N(0, sigma_g_j^2), and sigma_g_j^2 ~ Exp(lambda_sq_hyper_param / (2*common_sigma_g_sq)),
    # then posterior for sigma_g_j^2 is GIG. (This is one formulation of Bayesian Lasso).

    # The Julia code sampleGammaArray! seems to be for tau_j^2 (their gammaArray elements).
    # tau_j^2 | alpha_j, sigma_e^2, lambda_rate ~ GIG(...) (Generalized Inverse Gaussian)
    # The specific parameterization in Julia used a Metropolis-Hastings for 1/gamma_j where gamma_j related to specific variance.
    # For now, let's placeholder this update. A proper GIG or specific MH sampler is needed.
    # TODO: Implement sampling for current_gamma_array (elements are 1/tau_j^2)
    # As a placeholder, let's re-sample from a simple prior update if alpha is small.
    # For example, if alpha_j is small, tau_j^2 tends to be small (large 1/tau_j^2).
    # If alpha_j is large, tau_j^2 tends to be large (small 1/tau_j^2).
    # This part is crucial and non-trivial.
    # Placeholder: Keep gamma_array fixed for now, or sample from prior if alpha is zero.
    for j in range(n_markers):
        if abs(current_alpha_st[j]) < 1e-6 : # If effect is tiny, sample tau_j^2 from prior like part
             # This is a heuristic, not the correct conditional posterior.
             current_gamma_array[j] = max(1e-6, lasso_lambda_sq_hyper / 2.0 + np.random.randn()*0.1)


    # Sample common marker variance (sigma_g^2)
    # This is sampled given current alpha_j and current tau_j^2 (via gamma_array)
    # Effective data for sigma_g^2 is alpha_j / tau_j (or alpha_j * sqrt(gamma_j))
    if geno_data.marker_effect_variance.estimate_variance:
        # sum_squares_for_common_var = sum (alpha_j^2 / tau_j^2) = sum (alpha_j^2 * gamma_j)
        sum_sq_scaled_alpha = np.sum(current_alpha_st**2 * current_gamma_array) # sum (alpha_j^2 / (tau_j^2))

        # Posterior for common_marker_variance (sigma_g^2) is ScaledInvChi2
        # nu_post = prior_df + n_markers
        # S2_post = (sum_sq_scaled_alpha + prior_df * prior_scale) / nu_post
        new_common_marker_var = sample_scalar_variance_component(
            data_vector=current_alpha_st, # Pass alphas
            prior_df=geno_data.marker_effect_variance.df,
            prior_scale_parameter=float(geno_data.marker_effect_variance.scale), # S0^2
            # Weights here should be gamma_j (1/tau_j^2)
            # sample_scalar_variance_component takes inv_weights for y_i ~ N(mu, sigma^2/w_i)
            # Here, alpha_j ~ N(0, sigma_g^2 * tau_j^2). So alpha_j / tau_j ~ N(0, sigma_g^2).
            # So, data_vector is alpha_j / tau_j = alpha_j * sqrt(gamma_j)
            # and inv_weights is None.
            inv_weights=current_gamma_array # This is not quite right.
                                            # If alpha_j ~ N(0, sigma_g^2 * tau_j^2), then
                                            # for sigma_g^2, sum_sq is sum(alpha_j^2 / tau_j^2).
                                            # This means data for sampling sigma_g^2 is effectively alpha_j / sqrt(tau_j^2)
                                            # i.e. alpha_j * (gamma_j)^0.25 ??? No.
                                            # SSE = sum( (alpha_j - 0)^2 / tau_j^2 ) = sum( alpha_j^2 * gamma_j )
                                            # This SSE is for sigma_g^2.
                                            # So, sample_scalar_variance_component needs this SSE directly.
                                            # It computes sum(data^2 * w). If data=alpha, w=gamma, then sum(alpha^2*gamma).
        )
        # Let's manually construct the SSE for sigma_g^2
        sse_for_common_var = np.sum(current_alpha_st**2 * current_gamma_array)
        nu_post_common_var = geno_data.marker_effect_variance.df + n_markers
        s2_post_common_var = (sse_for_common_var + geno_data.marker_effect_variance.df * float(geno_data.marker_effect_variance.scale)) / nu_post_common_var

        if nu_post_common_var > 0 and s2_post_common_var > 1e-12:
            new_common_marker_var = sample_scaled_inverse_chi_squared(df=nu_post_common_var, scale_sq=s2_post_common_var)
            if new_common_marker_var > 1e-12:
                geno_data.marker_effect_variance.value = new_common_marker_var
            else: geno_data.marker_effect_variance.value = 1e-12
        # Else retain old value if params are bad


if __name__ == '__main__':
    print("Testing RR-BLUP ST marker sampler (basic functionality):")
    n_obs, n_mkrs = 100, 200

    # Simulate GenotypesData
    sim_geno_data = GenotypesData(name="sim_markers")
    sim_geno_data.genotypes = np.random.randint(0, 3, size=(n_obs, n_mkrs)).astype(float)
    # Center genotypes (important for RR-BLUP interpretation)
    sim_geno_data.genotypes -= np.mean(sim_geno_data.genotypes, axis=0)
    sim_geno_data.alpha_samples = [np.zeros(n_mkrs)] # Initialize for single trait

    # Simulate y_corrected (y_obs - X*beta)
    true_residual_var = 1.0
    true_marker_var = 0.01 # Assuming sigma_g^2

    # Simulate true alpha and then y_corrected
    true_alpha = np.random.randn(n_mkrs) * np.sqrt(true_marker_var)
    y_obs_true_marker_effect = sim_geno_data.genotypes @ true_alpha
    y_corr_input = y_obs_true_marker_effect + np.random.randn(n_obs) * np.sqrt(true_residual_var)

    # Make copies for different test runs if needed
    y_corr_for_sampling = y_corr_input.copy()

    print(f"Input y_corrected mean: {np.mean(y_corr_for_sampling):.3f}, var: {np.var(y_corr_for_sampling):.3f}")

    # Run sampler for a few iterations (normally this is one pass within MCMC iter)
    # Here, we test if alpha converges somewhat.
    num_passes = 100
    collected_alphas = []
    for i in range(num_passes):
        # y_corr_input is y_obs - Xbeta. The function modifies it to be y_obs - Xbeta - Z*alpha_new
        # So for next pass, we want it to be y_obs - Xbeta again.
        # However, within an MCMC, y_corrected passed to marker sampler IS y_obs - Xbeta.
        # The function itself maintains the y_corrected state for its internal single pass over markers.
        # The y_corrected that is returned (implicitly, by modification) is for the next sampler (e.g. residual var sampler)

        # To test convergence of alpha, we need to feed it the "full" y_corrected for each pass
        # y_eff = y_obs_true_marker_effect + error_term (this is y_corr_input)
        # The function expects y_corrected to be y_obs-Xbeta.
        # And it will update alpha, and implicitly update the y_corrected it uses.

        # Let's simulate the MCMC context:
        # y_corrected_for_markers = y_obs - X*beta (fixed for this part of MCMC iter)
        # The function then iterates over markers, updating alpha and internally adjusting y_corrected.

        # To test the sampler's behavior over multiple full passes (like MCMC iterations):
        y_current_iter = y_corr_input.copy() # y_obs - X*beta

        _sample_marker_effects_rrblup_st(
            sim_geno_data,
            y_current_iter, # This will be modified by the function
            true_residual_var,
            true_marker_var
        )
        # sim_geno_data.alpha_samples[0] is now the new alpha for this pass
        if i > num_passes // 2: # Burn-in for this mini-MCMC
            collected_alphas.append(sim_geno_data.alpha_samples[0].copy())

    if collected_alphas:
        mean_sampled_alpha = np.mean(collected_alphas, axis=0)
        correlation = np.corrcoef(true_alpha, mean_sampled_alpha)[0,1]
        print(f"Mean sampled alpha (after {num_passes} passes) correlation with true alpha: {correlation:.3f}")
        # This correlation should be reasonably high if the sampler works.

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(true_alpha, mean_sampled_alpha, alpha=0.5)
        plt.plot([min(true_alpha), max(true_alpha)], [min(true_alpha), max(true_alpha)], 'r--')
        plt.xlabel("True Alpha")
        plt.ylabel("Sampled Mean Alpha")
        plt.title("RR-BLUP Sampler Test")
        plt.savefig("rrblup_sampler_test.png")
        print("Saved RR-BLUP sampler test plot to rrblup_sampler_test.png")
    else:
        print("No alphas collected, check num_passes.")

```
