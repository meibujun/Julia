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


# Placeholder for multi-trait RR-BLUP
# def _sample_marker_effects_rrblup_mt(...):
#     pass

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
