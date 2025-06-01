import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Union, Tuple
# Assuming model_components.py and genotype_handler.py are accessible
# from ..core.model_components import MME_py, VarianceComponent, RandomEffectComponent, GenotypesComponent # etc.
# For now, define a placeholder MME_py if not fully fleshed out from previous steps
# This will be refined as we integrate.

# Ensure MME_py is defined or imported. For now, using a placeholder.
# This should ideally come from model_components.py
class MME_py:
    def __init__(self):
        self.n_models = 0
        self.X_effects_matrix: Optional[np.ndarray] = None # Design matrix for fixed and non-marker randoms
        self.y_observed: Optional[np.ndarray] = None
        self.solution_vector: Optional[np.ndarray] = None
        self.y_corrected: Optional[np.ndarray] = None

        self.residual_variance_prior: Any = None # Should be VarianceComponent
        self.current_residual_variance: Union[float, np.ndarray, None] = None

        self.random_effect_components: List[Any] = []
        self.current_random_effect_vc_estimates: List[Union[float, np.ndarray, None]] = []

        self.genotype_components: List[Any] = []
        self.current_genotype_vc_estimates: List[Union[float, np.ndarray, None]] = [] # For primary var of each GC

        self.inv_weights: Optional[np.ndarray] = None
        self.obs_id: List[str] = []
        self.mcmc_settings: Dict[str, Any] = {} # Stores chain_length, burn_in, etc.
        self.H_inverse: Optional[Any] = None # For SSGBLUP

        # For storing results
        self.posterior_samples: Dict[str, List[Any]] = {}
        self.posterior_means: Dict[str, Any] = {}

    def get_mcmc_setting(self, key: str, default: Any = None) -> Any:
        return self.mcmc_settings.get(key, default)

    def is_single_step_gblup(self) -> bool: # Example helper
        return self.get_mcmc_setting("single_step_analysis", False) and \
               any(hasattr(gc, 'method') and gc.method == "GBLUP" for gc in self.genotype_components)

    def get_polygenic_effect_component_for_ssgblup(self) -> Optional[Any]:
        for rec in self.random_effect_components:
            if hasattr(rec, 'random_type') and rec.random_type == "A": # or "H" after update
                return rec
        return None


def _construct_mme_lhs_rhs_py(mme: MME_py, y_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the LHS and RHS of the Mixed Model Equations for location parameters.
    LHS = X' R_eff^{-1} X + sum(K_i^{-1} * lambda_i)
    RHS = X' R_eff^{-1} y_current
    y_current is the data vector (e.g., y_obs or y_corrected for marker effects).
    """
    if mme.X_effects_matrix is None:
        raise ValueError("mme.X_effects_matrix is not set.")

    n_total_obs = mme.X_effects_matrix.shape[0]

    # Effective inverse residual covariance matrix (R_eff_inv)
    if mme.n_models == 1:
        if mme.current_residual_variance is None or mme.current_residual_variance == 0:
            raise ValueError("Current residual variance (sigma_e^2) is not set or zero.")

        current_sigma_e2 = mme.current_residual_variance
        current_inv_weights = mme.inv_weights if mme.inv_weights is not None else np.ones(n_total_obs)
        if len(current_inv_weights) != n_total_obs:
            if len(current_inv_weights) == n_total_obs // mme.n_models and mme.n_models > 1:
                 current_inv_weights = np.repeat(current_inv_weights, mme.n_models)
            elif len(current_inv_weights) != n_total_obs :
                 raise ValueError(f"inv_weights length {len(current_inv_weights)} doesn't match total_obs {n_total_obs}")

        R_eff_inv_diag = (1.0 / current_sigma_e2) * current_inv_weights
        LHS = (mme.X_effects_matrix.T * R_eff_inv_diag) @ mme.X_effects_matrix
        RHS = (mme.X_effects_matrix.T * R_eff_inv_diag) @ y_current
    else:
        print("Warning: Multi-trait MME construction for location parameters is using a simplified diagonal R_eff_inv.")
        if mme.current_residual_variance is None or not isinstance(mme.current_residual_variance, np.ndarray):
            raise ValueError("Current multi-trait residual covariance matrix not set or not a matrix.")

        diag_Sigma_e_inv = 1.0 / np.diag(mme.current_residual_variance)
        n_obs_per_trait = mme.X_effects_matrix.shape[0] // mme.n_models

        R_eff_inv_diag_mt = np.zeros(n_total_obs)
        current_inv_weights_mt = mme.inv_weights if mme.inv_weights is not None else np.ones(n_obs_per_trait)

        for i_trait in range(mme.n_models):
            start, end = i_trait * n_obs_per_trait, (i_trait + 1) * n_obs_per_trait
            trait_weights = current_inv_weights_mt[start:end] if len(current_inv_weights_mt) == n_total_obs else current_inv_weights_mt
            R_eff_inv_diag_mt[start:end] = diag_Sigma_e_inv[i_trait] * trait_weights

        LHS = (mme.X_effects_matrix.T * R_eff_inv_diag_mt) @ mme.X_effects_matrix
        RHS = (mme.X_effects_matrix.T * R_eff_inv_diag_mt) @ y_current

    # --- This is the addVinv_py equivalent part ---
    # current_LHS starts as X_effects_matrix' R_eff_inv X_effects_matrix.
    # This X_effects_matrix should correspond to fixed effects.
    # Random effects (polygenic, other non-marker REs) will have their contributions added.

    if mme.n_models == 1 and mme.current_residual_variance is not None and mme.current_residual_variance > 1e-9:
        sigma_e2 = mme.current_residual_variance

        for idx, rec in enumerate(mme.random_effect_components):
            # This requires robust mapping of rec to its columns in LHS. Placeholder logic:
            if hasattr(rec, 'start_col_in_LHS') and hasattr(rec, 'num_cols_in_LHS'):
                start_col, num_cols = rec.start_col_in_LHS, rec.num_cols_in_LHS
                if start_col + num_cols > LHS.shape[0]: continue # Index out of bounds

                current_rec_variance = mme.current_random_effect_vc_estimates[idx]
                if current_rec_variance is None or current_rec_variance == 0: continue

                lambda_re_val = sigma_e2 / current_rec_variance

                if rec.random_type == "I":
                    for k_diag in range(num_cols):
                        LHS[start_col + k_diag, start_col + k_diag] += lambda_re_val
                elif rec.random_type in ["A", "H"] and rec.Vinv_obj is not None:
                    if rec.Vinv_obj.shape == (num_cols, num_cols):
                         LHS[start_col : start_col+num_cols, start_col : start_col+num_cols] += rec.Vinv_obj * lambda_re_val
                    # else: print warning about shape mismatch for K_inv
    return LHS, RHS


def sample_location_parameters_py(mme: MME_py, y_current_for_rhs: np.ndarray):
    """
    Samples fixed and random (non-marker) effects using Gibbs sampling (single iteration).
    Updates mme.solution_vector in place.
    y_current_for_rhs is the current data vector (y_obs or y_corrected for markers).
    """
    A, b = _construct_mme_lhs_rhs_py(mme, y_current_for_rhs)

    if mme.solution_vector is None or len(mme.solution_vector) != A.shape[0]:
        mme.solution_vector = np.zeros(A.shape[0])

    for i in range(len(mme.solution_vector)):
        if A[i, i] < 1e-9: continue

        off_diagonal_sum = A[i, :i] @ mme.solution_vector[:i] + \
                           A[i, i+1:] @ mme.solution_vector[i+1:]

        mean_i = (b[i] - off_diagonal_sum) / A[i, i]
        variance_i_scaled = 1.0 / A[i, i]

        actual_sampling_variance = variance_i_scaled
        if mme.n_models == 1 and mme.current_residual_variance is not None:
            actual_sampling_variance *= mme.current_residual_variance

        if actual_sampling_variance < 1e-9: actual_sampling_variance = 1e-9

        mme.solution_vector[i] = np.random.normal(mean_i, np.sqrt(actual_sampling_variance))

# --- Actual Sampling Functions ---

def sample_scaled_inv_chi_sq(df: float, scale_param: float, num_samples: int = 1) -> Union[float, np.ndarray]:
    """
    Samples from a scaled inverse chi-squared distribution.
    Assumes scale_param is the sum of squares (S_0 or posterior SS).
    Sample = Posterior_Sum_of_Squares / ChiSq_sample(Posterior_DF).
    """
    if df <= 1e-9:
        return 1e-9 if scale_param < 1e-9 else scale_param / (df if df > 1e-9 else 1.0)

    chi2_sample = np.random.chisquare(df, size=num_samples)
    chi2_sample[chi2_sample < 1e-9] = 1e-9

    if num_samples == 1:
        return scale_param / chi2_sample[0]
    else:
        return scale_param / chi2_sample


def sample_residual_variance_py(mme: MME_py):
    """Samples residual variance(s) (sigma_e^2 or Sigma_e). Updates mme.current_residual_variance."""
    if mme.y_corrected is None: return

    sse = np.dot(mme.y_corrected, mme.y_corrected)
    n_total_obs = len(mme.y_corrected)

    prior = mme.residual_variance_prior
    prior_sum_of_squares = prior.df * prior.value if prior.value is not None and prior.df > 0 else 0.0
    if prior.value is None and prior.scale is not None: prior_sum_of_squares = prior.scale

    posterior_df = prior.df + n_total_obs
    posterior_scale_sum_squares = prior_sum_of_squares + sse

    if mme.n_models == 1:
        new_res_var = sample_scaled_inv_chi_sq(posterior_df, posterior_scale_sum_squares)
        mme.current_residual_variance = new_res_var if new_res_var > 1e-9 else 1e-9
    else:
        print("Warning: Multi-trait residual variance sampling is simplified (assumes diagonal).")
        if isinstance(mme.current_residual_variance, np.ndarray) and mme.residual_variance_prior.constraint:
            n_obs_per_trait = n_total_obs // mme.n_models
            new_res_var_diag = np.zeros(mme.n_models)
            for i in range(mme.n_models):
                y_corr_trait_i = mme.y_corrected[i*n_obs_per_trait:(i+1)*n_obs_per_trait]
                sse_trait_i = np.dot(y_corr_trait_i, y_corr_trait_i)

                prior_df_trait = prior.df / mme.n_models
                prior_val_trait = np.diag(prior.value)[i] if isinstance(prior.value, np.ndarray) else (prior.value if prior.value else 0.0)
                prior_ss_trait = prior_df_trait * prior_val_trait

                post_df_trait_i = prior_df_trait + n_obs_per_trait
                post_scale_ss_trait_i = prior_ss_trait + sse_trait_i

                sampled_var_trait_i = sample_scaled_inv_chi_sq(post_df_trait_i, post_scale_ss_trait_i)
                new_res_var_diag[i] = sampled_var_trait_i if sampled_var_trait_i > 1e-9 else 1e-9
            mme.current_residual_variance = np.diag(new_res_var_diag)
        else:
            new_res_val_scalar = sample_scaled_inv_chi_sq(posterior_df, posterior_scale_sum_squares)
            mme.current_residual_variance = new_res_val_scalar if new_res_val_scalar > 1e-9 else 1e-9


def sample_other_random_effect_variances_py(mme: MME_py):
    """Samples variances for non-marker random effects. Updates mme.current_random_effect_vc_estimates."""
    if mme.solution_vector is None: return

    for i, rec in enumerate(mme.random_effect_components):
        if not rec.variance_prior.estimate_variance: continue

        # This requires mme.effect_slices_in_mme to be correctly populated by the MME builder
        # and mme.solution_vector to contain the current estimates for all effects.

        sum_sq_u = 0.0
        num_levels_effect = 0

        # Try to find the slice for this random effect component.
        # The component 'rec' itself might store its mapping if MME builder sets it.
        # Or we search by a conventional name in mme.effect_slices_in_mme.
        # This is still an area needing robust definition by the MME builder.

        # Conceptual: find effect_slice for 'rec'
        # For example, if rec.term_array = ["y1:animal"], its key in effect_slices_in_mme might be "y1:animal".
        # This assumes each RandomEffectComponent corresponds to one main entry in effect_slices_in_mme
        # which defines its block in solution_vector.

        # For this example, let's assume rec has attributes like:
        # rec.slice_in_mme = slice(start, end) # Set by MME builder
        # rec.Vinv_obj = K_inv (e.g. A_inv, H_inv, or identity for IID if not part of X_fixed)

        effect_slice = None
        if hasattr(rec, 'slice_in_mme') and rec.slice_in_mme is not None:
            effect_slice = rec.slice_in_mme
        elif mme.effect_slices_in_mme: # Fallback: try to find by name (less robust)
            # This needs a consistent naming convention. Assume rec.term_array[0] or rec.name attribute.
            key_to_find = rec.term_array[0] if rec.term_array else (rec.name if hasattr(rec, 'name') else None)
            if key_to_find and key_to_find in mme.effect_slices_in_mme:
                effect_slice = mme.effect_slices_in_mme[key_to_find]

        if effect_slice:
            u_current = mme.solution_vector[effect_slice]
            num_levels_effect = len(u_current)

            if rec.Vinv_obj is not None: # e.g., A_inv, H_inv
                # Ensure Vinv_obj is conformable with u_current
                if rec.Vinv_obj.shape == (num_levels_effect, num_levels_effect):
                    # sum_sq_u = u_current.T @ rec.Vinv_obj @ u_current # If Vinv_obj is sparse, use sparse methods
                    if hasattr(rec.Vinv_obj, 'dot'): # Scipy sparse matrix
                        sum_sq_u = u_current.T @ rec.Vinv_obj.dot(u_current)
                    else: # Numpy array
                        sum_sq_u = u_current.T @ rec.Vinv_obj @ u_current
                else:
                    print(f"Warning: Vinv shape mismatch for RE {rec.term_array}. Using u'u.")
                    sum_sq_u = np.dot(u_current, u_current) # Fallback to IID like for sum of squares
            else: # IID effect (Vinv is Identity)
                sum_sq_u = np.dot(u_current, u_current)
        else:
            # Fallback to placeholder if effect slice not found (indicates MME builder issue)
            print(f"Warning: Could not map random effect {rec.term_array} to solution vector for SSE calculation. Using placeholder SSE.")
            num_levels_effect = rec.variance_prior.df # Use prior df to avoid div by zero if levels=0
            u_dummy_solutions = np.random.normal(0, np.sqrt(rec.variance_prior.value if rec.variance_prior.value is not None and rec.variance_prior.value > 0 else 1.0), int(num_levels_effect) if num_levels_effect > 0 else 10)
            sum_sq_u = np.dot(u_dummy_solutions, u_dummy_solutions)


        prior = rec.variance_prior
        posterior_df = prior.df + num_levels_effect

        prior_sum_of_squares = prior.df * prior.value if prior.value is not None and prior.df > 0 else 0.0
        if prior.value is None and prior.scale is not None: prior_sum_of_squares = prior.scale

        posterior_scale_sum_squares = prior_sum_of_squares + sum_sq_u

        new_variance_sample = sample_scaled_inv_chi_sq(posterior_df, posterior_scale_sum_squares)
        new_variance_sample = new_variance_sample if new_variance_sample > 1e-9 else 1e-9

        mme.current_random_effect_vc_estimates[i] = new_variance_sample

        if hasattr(rec, 'variance_prior_old') and hasattr(rec, 'variance_prior_new'):
            rec.variance_prior_old.value = rec.variance_prior_new.value
            rec.variance_prior_new.value = new_variance_sample


def sample_marker_variance_gblup_py(gc: Any, mme: MME_py, trait_idx: int):
    """Samples genetic variance (sigma_a^2) for GBLUP. Updates corresponding mme.current_genotype_vc_estimates."""
    if gc.alpha is None or gc.D_eigenvalues is None: return

    alpha_trait = gc.alpha[trait_idx]
    if alpha_trait is None: return

    valid_D_mask = gc.D_eigenvalues > 1e-9
    if not np.any(valid_D_mask): return

    sse_g = np.sum(alpha_trait[valid_D_mask]**2 / gc.D_eigenvalues[valid_D_mask])

    prior = gc.total_genetic_variance_prior
    if prior.value is None: prior.value = 0.1

    n_effects = len(alpha_trait)
    posterior_df = prior.df + n_effects
    prior_sum_of_squares = prior.df * prior.value
    posterior_scale_sum_squares = prior_sum_of_squares + sse_g

    new_sigma_a2 = sample_scaled_inv_chi_sq(posterior_df, posterior_scale_sum_squares)
    new_sigma_a2 = new_sigma_a2 if new_sigma_a2 > 1e-9 else 1e-9

    gc_idx = mme.genotype_components.index(gc)
    mme.current_genotype_vc_estimates[gc_idx] = new_sigma_a2
    gc.total_genetic_variance_prior.value = new_sigma_a2
    gc.marker_variance_prior.value = new_sigma_a2


def sample_marker_variance_bayesc0_py(gc: Any, mme: MME_py, trait_idx: int):
    """Samples common marker effect variance (sigma_g^2) for BayesC0. Updates mme.current_genotype_vc_estimates."""
    if gc.alpha is None: return
    alpha_trait = gc.alpha[trait_idx]
    if alpha_trait is None: return

    sse_markers = np.dot(alpha_trait, alpha_trait)

    prior = gc.marker_variance_prior
    if prior.value is None: prior.value = 0.01

    n_markers_in_model = gc.n_markers
    posterior_df = prior.df + n_markers_in_model
    prior_sum_of_squares = prior.df * prior.value
    posterior_scale_sum_squares = prior_sum_of_squares + sse_markers

    new_sigma_g2 = sample_scaled_inv_chi_sq(posterior_df, posterior_scale_sum_squares)
    new_sigma_g2 = new_sigma_g2 if new_sigma_g2 > 1e-9 else 1e-9

    gc_idx = mme.genotype_components.index(gc)
    mme.current_genotype_vc_estimates[gc_idx] = new_sigma_g2
    gc.marker_variance_prior.value = new_sigma_g2

# --- Placeholder for marker effects samplers (complex, depend on method) ---
def sample_marker_effects_gblup_py(gc: Any, mme: MME_py, y_corr_trait: np.ndarray, trait_idx: int):
    pass

def sample_marker_effects_bayesc0_py(gc: Any, mme: MME_py, y_corr_trait: np.ndarray, trait_idx: int):
    # For BayesC0: all markers included (delta_j=1). Effects share common variance sigma_g^2.
    # This function updates gc.alpha[trait_idx] and y_corr_trait IN PLACE.

    alpha_current_trait = gc.alpha[trait_idx]
    X_markers = gc.genotype_matrix # Should be (n_obs_for_this_trait x n_markers)

    # Ensure X_markers is aligned with y_corr_trait if y_corr_trait is a view of stacked y_corrected
    # This alignment is complex for MT if X_markers is common for all individuals.
    # For ST, X_markers rows should match y_corr_trait.
    # n_obs_trait = len(y_corr_trait)
    # if X_markers.shape[0] != n_obs_trait:
    #     print(f"Warning: X_markers row mismatch in sample_marker_effects_bayesc0_py. X_rows: {X_markers.shape[0]}, y_len: {n_obs_trait}")
    #     # This might require slicing X_markers if it's a global matrix for all obs_ids
    #     # X_markers_trait = X_markers[trait_specific_indices, :] # Conceptual
    # else:
    #     X_markers_trait = X_markers

    # Simplified: assume X_markers is correctly dimensioned for y_corr_trait (ST case)
    X_markers_trait = X_markers

    sigma_e2 = mme.current_residual_variance # Assuming ST for simplicity here
    gc_idx = mme.genotype_components.index(gc)
    sigma_g2 = mme.current_genotype_vc_estimates[gc_idx]

    if sigma_e2 is None or sigma_g2 is None or sigma_e2 < 1e-9 or sigma_g2 < 1e-9:
        # print("Warning: Variances (residual or marker) not available or too small for BayesC0 effect sampling.")
        return

    # inv_weights_diag = mme.inv_weights (assuming ST, and inv_weights match y_corr_trait)
    # R_inv_diag_eff = inv_weights_diag / sigma_e2
    # For now, assume inv_weights are 1 for simplicity of XjTXj_eff

    for j in range(gc.n_markers):
        Xj = X_markers_trait[:, j]

        # Add back current effect of marker j to y_corr_trait
        y_corr_trait += Xj * alpha_current_trait[j] # y_corr_marker_j = y_corr_without_any_effect_j + Xj * old_alpha_j
                                                  # Here, y_corr_trait is y_corr_without_effect_j (after previous iter)
                                                  # So, effectively y_corr_trait becomes y_corr_without_effect_k_neq_j

        # Calculate Xj' R_inv Xj and Xj' R_inv y_corr_marker_j
        # If R_inv is diagonal (1/sigma_e2 * inv_weights)
        # XjTXj_eff = np.sum(Xj * Xj * R_inv_diag_eff) # Weighted sum of squares of Xj
        # XjTy_eff = np.sum(Xj * y_corr_trait * R_inv_diag_eff) # Weighted sum of Xj * y_corr

        # Simplified: inv_weights = 1
        XjTXj_eff = np.dot(Xj, Xj) / sigma_e2
        XjTy_eff = np.dot(Xj, y_corr_trait) / sigma_e2

        lhs_j = XjTXj_eff + (1.0 / sigma_g2)
        if lhs_j < 1e-9: lhs_j = 1e-9 # Avoid division by zero

        rhs_j = XjTy_eff
        # In Julia: rhs = (dot(xRinv,yCorr) + xpRinvx[j]*α[j])*invVarRes
        # Here, xpRinvx[j]*α[j]*invVarRes is part of XjTy_eff if y_corr_trait was y_obs - sum(Xk*ak) + Xj*aj_old
        # The current y_corr_trait = y_obs - sum(Xk*ak_current_iter_except_j) - Xj*aj_prev_iter
        # So Xj'Rinv * (y_corr_trait) = Xj'Rinv * (y_obs - sum(Xk*ak_no_j))

        mean_j = rhs_j / lhs_j
        var_j = 1.0 / lhs_j # This is conditional variance scaled by sigma_e2, actual is var_j * sigma_e2

        alpha_new_j = np.random.normal(mean_j, np.sqrt(var_j)) # Samples from N(mean, var)

        # Update y_corr_trait by subtracting new effect and adding back old (which was already part of it)
        # y_corr_trait effectively has old_alpha_j contribution. We need to adjust for (new - old).
        y_corr_trait -= Xj * alpha_new_j # y_corr_trait now y_obs - sum(Xk*ak_new_iter_incl_j)
        alpha_current_trait[j] = alpha_new_j


def sample_pi_value_py(gc: Any, mme: MME_py, trait_idx: int):
    pass


def run_mcmc_py(mme: MME_py, df_phenotypes: pd.DataFrame, mcmc_settings: Dict):
