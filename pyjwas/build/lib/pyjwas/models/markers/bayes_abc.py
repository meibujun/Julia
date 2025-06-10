import numpy as np
from typing import List, Optional, Union

from pyjwas.core.definitions import Genotypes, DefaultFloat, Variance # Added Variance
# Assuming GibbsMats results are passed appropriately.
# For xArray[j] (marker j genotype vector), xRinvArray[j] (X[:,j]' * R_inv_scalar), xpRinvx[j] (X[:,j]' * R_inv_scalar * X[:,j])

def bayes_abc_marker_sampler_st(
    x_array: List[np.ndarray],
    x_rinv_array: Optional[List[np.ndarray]], # Made Optional as it's not used in current Python interpretation
    xp_rinvx_array: List[DefaultFloat],
    y_corr: np.ndarray,
    alpha_effects: np.ndarray,
    beta_effects: np.ndarray,
    delta_indicators: np.ndarray,
    residual_variance: DefaultFloat,
    marker_effect_variances: np.ndarray,
    pi_prob_non_zero: DefaultFloat
) -> None:
    """
    Core single-trait Gibbs sampler for marker effects (BayesA/B/C logic).
    Modifies y_corr, alpha_effects, beta_effects, delta_indicators in place.
    Corresponds to Julia's core BayesABC! function.
    """
    log_prior_delta1: float
    log_prior_delta0: float

    if not (0 < pi_prob_non_zero < 1):
        if pi_prob_non_zero == 1.0:
            log_prior_delta1 = 0.0
            log_prior_delta0 = -np.inf
        elif pi_prob_non_zero == 0.0:
            log_prior_delta1 = -np.inf
            log_prior_delta0 = 0.0
        else:
            raise ValueError("pi_prob_non_zero must be in (0,1) for variable selection, or exactly 0 or 1.")
    else:
        log_prior_delta1 = np.log(pi_prob_non_zero)
        log_prior_delta0 = np.log(1.0 - pi_prob_non_zero)

    inv_residual_variance = 1.0 / residual_variance
    marker_effect_variances_safe = np.maximum(marker_effect_variances, 1e-12)
    inv_marker_effect_variances = 1.0 / marker_effect_variances_safe
    log_marker_effect_variances = np.log(marker_effect_variances_safe)

    n_markers = len(alpha_effects)

    for j in range(n_markers):
        marker_genotypes_j = x_array[j]

        y_star_j = y_corr + marker_genotypes_j * alpha_effects[j]

        d_j = np.dot(marker_genotypes_j, y_star_j) * inv_residual_variance
        C_j = xp_rinvx_array[j] * inv_residual_variance + inv_marker_effect_variances[j]

        if C_j == 0:
            C_j = 1e-12

        inv_C_j = 1.0 / C_j
        posterior_mean_beta_j = d_j * inv_C_j

        log_marginal_lik_ratio_part = -0.5 * (np.log(C_j) + log_marker_effect_variances[j] - posterior_mean_beta_j * d_j)
        current_log_post_odds_delta1_vs_delta0 = log_marginal_lik_ratio_part + (log_prior_delta1 - log_prior_delta0)

        prob_delta1_posterior = 1.0 / (1.0 + np.exp(-current_log_post_odds_delta1_vs_delta0))

        old_alpha_j = alpha_effects[j]

        if np.random.rand() < prob_delta1_posterior:
            delta_indicators[j] = 1.0
            beta_effects[j] = np.random.normal(loc=posterior_mean_beta_j, scale=np.sqrt(inv_C_j))
            alpha_effects[j] = beta_effects[j]
        else:
            delta_indicators[j] = 0.0
            beta_effects[j] = np.random.normal(loc=0.0, scale=np.sqrt(marker_effect_variances_safe[j]))
            alpha_effects[j] = 0.0

        y_corr -= marker_genotypes_j * (alpha_effects[j] - old_alpha_j)

def bayes_abc_st_wrapper(
    genotype_set: Genotypes,
    y_corr: np.ndarray,
    residual_variance: DefaultFloat
):
    """
    Wrapper for single-trait BayesABC marker sampler.
    Uses precomputed components if available in genotype_set (Mi.mArray, etc.).
    """
    if genotype_set.n_traits != 1:
        raise ValueError("This BayesABC wrapper is for single-trait Genotypes objects.")

    if genotype_set.alpha is None or not genotype_set.alpha or genotype_set.alpha[0] is None:
        raise ValueError("genotype_set.alpha[0] (marker effects) must be initialized.")
    if genotype_set.beta is None or not genotype_set.beta or genotype_set.beta[0] is None:
        raise ValueError("genotype_set.beta[0] (marker effects) must be initialized.")
    if genotype_set.delta is None or not genotype_set.delta or genotype_set.delta[0] is None:
        raise ValueError("genotype_set.delta[0] (marker effects) must be initialized.")


    if genotype_set.genotypes is None:
        raise ValueError("Genotype matrix is None in genotype_set.")

    x_array_list = [genotype_set.genotypes[:,j].astype(DefaultFloat) for j in range(genotype_set.n_markers)]
    xp_rinvx_scalar_list = [np.sum(x_j**2) for x_j in x_array_list]

    marker_variances: np.ndarray
    if genotype_set.method == "BayesB":
        if not isinstance(genotype_set.G.val, np.ndarray) or len(genotype_set.G.val) != genotype_set.n_markers:
            raise ValueError("For BayesB, Genotypes.G.val must be an array of locus-specific variances.")
        marker_variances = genotype_set.G.val
    else:
        if not isinstance(genotype_set.G.val, (float, np.floating, np.ndarray)):
            raise ValueError(f"For {genotype_set.method}, Genotypes.G.val (sigma_g^2) must be a scalar float. Got: {genotype_set.G.val}")
        scalar_marker_variance = float(np.array(genotype_set.G.val).item())
        marker_variances = np.full(genotype_set.n_markers, scalar_marker_variance, dtype=DefaultFloat)

    if genotype_set.pi_val is None or not isinstance(genotype_set.pi_val, (float, np.floating)): # type: ignore
         raise ValueError("Genotype_set.pi_val (probability of non-zero effect) must be a scalar float.")

    bayes_abc_marker_sampler_st(
        x_array=x_array_list,
        x_rinv_array=None,
        xp_rinvx_array=xp_rinvx_scalar_list,
        y_corr=y_corr,
        alpha_effects=genotype_set.alpha[0],
        beta_effects=genotype_set.beta[0],
        delta_indicators=genotype_set.delta[0],
        residual_variance=residual_variance,
        marker_effect_variances=marker_variances,
        pi_prob_non_zero=float(genotype_set.pi_val) # type: ignore
    )

if __name__ == '__main__':
    print("--- BayesABC Marker Sampler Examples (Single Trait) ---")

    n_obs_test, n_markers_test = 20, 5

    Mi_test = Genotypes(name="test_chip", n_traits=1, n_markers=n_markers_test, method="BayesC")
    Mi_test.genotypes = np.random.randint(0, 3, size=(n_obs_test, n_markers_test)).astype(DefaultFloat)
    Mi_test.genotypes -= np.mean(Mi_test.genotypes, axis=0)

    Mi_test.alpha = [np.zeros(n_markers_test, dtype=DefaultFloat)]
    Mi_test.beta = [np.zeros(n_markers_test, dtype=DefaultFloat)]
    Mi_test.delta = [np.ones(n_markers_test, dtype=DefaultFloat)]

    # Initialize Variance object for G correctly
    Mi_test.G = Variance(val=0.01, df=4.0, scale=0.01*(4.0-1-1)) # Common marker effect variance sigma_g^2
    Mi_test.pi_val = 0.1

    y_corr_test = np.random.randn(n_obs_test).astype(DefaultFloat)
    vare_test = 1.0

    print(f"Initial alpha: {Mi_test.alpha[0][:3]}...")
    print(f"Initial delta: {Mi_test.delta[0][:3]}...")
    print(f"Initial y_corr sum: {np.sum(y_corr_test):.3f}")

    try:
        bayes_abc_st_wrapper(Mi_test, y_corr_test, vare_test)
        print("bayes_abc_st_wrapper executed.")
        print(f"Updated alpha: {Mi_test.alpha[0][:3]}...")
        print(f"Updated delta: {Mi_test.delta[0][:3]}...")
        print(f"Updated y_corr sum: {np.sum(y_corr_test):.3f}")

        Mi_test_BayesB = Genotypes(name="test_chip_B", n_traits=1, n_markers=n_markers_test, method="BayesB")
        Mi_test_BayesB.genotypes = Mi_test.genotypes.copy()
        Mi_test_BayesB.alpha = [np.zeros(n_markers_test, dtype=DefaultFloat)]
        Mi_test_BayesB.beta = [np.zeros(n_markers_test, dtype=DefaultFloat)]
        Mi_test_BayesB.delta = [np.ones(n_markers_test, dtype=DefaultFloat)]
        # Initialize G.val as an array for BayesB
        Mi_test_BayesB.G = Variance(val=np.full(n_markers_test, 0.01, dtype=DefaultFloat), df=4.0, scale=0.01*(4.0-1-1))
        Mi_test_BayesB.pi_val = 0.1

        y_corr_test_B = np.random.randn(n_obs_test).astype(DefaultFloat)
        print("\nTesting BayesB...")
        bayes_abc_st_wrapper(Mi_test_BayesB, y_corr_test_B, vare_test)
        print("BayesB wrapper executed.")
        print(f"Updated alpha (BayesB): {Mi_test_BayesB.alpha[0][:3]}...")
        print(f"Updated delta (BayesB): {Mi_test_BayesB.delta[0][:3]}...")

    except Exception as e:
        print(f"Error in BayesABC example: {e}")
        import traceback
        traceback.print_exc()
