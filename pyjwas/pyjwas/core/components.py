from typing import List, Dict, Any, Union, Optional, Tuple # Added Tuple
import numpy as np
import pandas as pd # Only for type hint if Pedigree is a DataFrame, not used directly here yet
from scipy.sparse import csc_matrix, eye as sparse_eye, issparse,lil_matrix # For sparse identity & construction

from pyjwas.core.definitions import MME, ModelTerm, Variance, RandomEffect, Genotypes, DefaultFloat, DefaultInt, MCMCInfo # Added DefaultInt, MCMCInfo
from pyjwas.core.math_utils import check_positive_definite # Assuming this is in math_utils

# Placeholder for Pedigree object/module.
# For now, we'll assume it has methods like get_A_inverse() and get_ids()
# This will be properly integrated when the pedigree module is translated.
class PedigreePlaceholder:
    def __init__(self, name: str = "dummy_ped"):
        self.name = name
        self._ids: List[str] = [] # Ensure _ids is typed

    def get_A_inverse(self) -> Optional[csc_matrix]:
        print(f"Warning: Pedigree {self.name} using placeholder A-inverse (None).")
        return None # Needs actual implementation

    def get_ids(self) -> List[str]:
        #print(f"Warning: Pedigree {self.name} using placeholder IDs ({self._ids[:3]}...).")
        return self._ids

    def set_ids(self, ids: List[str]): # Helper for testing
        self._ids = ids


def set_random_effect_pedigree(
    mme: MME,
    random_term_str: str,
    pedigree: PedigreePlaceholder, # Replace with actual Pedigree type later
    G_prior_mean: Union[DefaultFloat, np.ndarray, bool] = False,
    df_prior: DefaultFloat = 4.0,
    estimate_variance: bool = True,
    estimate_scale: bool = False, # Julia: not supported for random terms
    constraint: bool = False # For multi-trait covariance constraint
) -> None:
    """
    Sets variables as random polygenic effects using pedigree information.
    Corresponds to the first set_random function in random_effects.jl.
    """
    if mme.ped is not None: # In Julia, mme.ped was 0 before assignment
        raise ValueError("Pedigree information is already added. Polygenic effects can only be set once.")

    mme.ped = pedigree # Store the pedigree object (actual or placeholder)

    # Call the general set_random_effect function, passing pedigree as Vinv_source
    set_random_effect_general(
        mme=mme,
        random_term_str=random_term_str,
        G_prior_mean=G_prior_mean,
        Vinv_matrix_or_source=pedigree, # Pass pedigree object itself
        level_names_for_Vinv=[], # Will be derived from pedigree if Vinv_matrix_or_source is Pedigree
        df_prior=df_prior,
        estimate_variance=estimate_variance,
        estimate_scale=estimate_scale,
        constraint=constraint
    )

def set_random_effect_general(
    mme: MME,
    random_term_str: str,
    G_prior_mean: Union[DefaultFloat, np.ndarray, bool] = False,
    Vinv_matrix_or_source: Optional[Union[csc_matrix, PedigreePlaceholder, np.ndarray]] = None, # Can be sparse matrix, dense, or Pedigree
    level_names_for_Vinv: Optional[List[str]] = None, # Required if Vinv_matrix is provided directly
    df_prior: DefaultFloat = 4.0,
    estimate_variance: bool = True,
    estimate_scale: bool = False, # Julia: not supported
    constraint: bool = False # For multi-trait covariance constraint
) -> None:
    """
    Sets variables as general random effects (i.i.d., pedigree-based, or with custom Vinv).
    Corresponds to the second set_random function in random_effects.jl.
    """
    # --- Pre-checks ---
    G_matrix: Optional[np.ndarray] = None
    if G_prior_mean is not False:
        G_matrix_temp = np.array(G_prior_mean, dtype=DefaultFloat)
        if G_matrix_temp.ndim == 0: # scalar G
            G_matrix = G_matrix_temp.reshape((1, 1))
        else:
            G_matrix = G_matrix_temp
        if not check_positive_definite(G_matrix): # Assumes math_utils.check_positive_definite
            raise ValueError("The covariance matrix G_prior_mean is not positive definite.")
    # else G_matrix remains None

    if level_names_for_Vinv is None:
        level_names_for_Vinv_processed: List[str] = []
    else:
        level_names_for_Vinv_processed = [str(name) for name in level_names_for_Vinv] # Ensure string type


    if estimate_scale:
        raise NotImplementedError("estimate_scale for random effect variance is not currently supported.")
    if constraint:
        pass


    # --- Identify and qualify model terms for this random effect ---
    term_names_from_random_str = [s.strip() for s in random_term_str.split(' ') if s.strip()]
    qualified_model_term_ids: List[str] = []

    for term_base_name in term_names_from_random_str:
        found_in_any_model = False
        for model_idx, model_eq_str in enumerate(mme.model_vec):
            for mt_obj in mme.model_terms:
                term_formula = "*".join(mt_obj.factors)
                if mt_obj.i_trait == mme.lhs_vec[model_idx] and term_formula == term_base_name:
                    qualified_model_term_ids.append(mt_obj.trm_str)
                    found_in_any_model = True

        if not found_in_any_model and term_base_name != "ϵ":
            is_interaction_part = any(term_base_name in mt.factors for mt in mme.model_terms)
            if not is_interaction_part:
                 print(f"Warning: Random effect term '{term_base_name}' was not directly found as a simple term in any model equation's RHS. It might be an interaction component or intended for 'ϵ'.")


    if not qualified_model_term_ids and "ϵ" not in term_names_from_random_str:
        raise ValueError(f"No model terms in MME matched the random effect specification: '{random_term_str}'.")

    if G_matrix is not None and len(term_names_from_random_str) != G_matrix.shape[0]:
        raise ValueError(f"Dimension mismatch: G_prior_mean implies {G_matrix.shape[0]} random effects, but '{random_term_str}' specifies {len(term_names_from_random_str)} base effects (which then expand to {len(qualified_model_term_ids)} qualified terms across traits).")

    # --- Determine random_type and Vinv ---
    actual_Vinv: Optional[csc_matrix] = None
    random_effect_type: str = "I"
    final_level_names: List[str] = list(level_names_for_Vinv_processed)

    if isinstance(Vinv_matrix_or_source, PedigreePlaceholder):
        random_effect_type = "A"
        ped_obj = Vinv_matrix_or_source
        actual_Vinv = ped_obj.get_A_inverse()
        final_level_names = ped_obj.get_ids()
        if not final_level_names:
            raise ValueError("Pedigree effect specified, but no IDs found in pedigree object.")

    elif Vinv_matrix_or_source is not None:
        random_effect_type = "V"
        if isinstance(Vinv_matrix_or_source, np.ndarray):
            actual_Vinv = csc_matrix(Vinv_matrix_or_source)
        elif issparse(Vinv_matrix_or_source):
            actual_Vinv = Vinv_matrix_or_source # type: ignore
        else:
            raise TypeError("Vinv_matrix_or_source must be a NumPy array, SciPy sparse matrix, or Pedigree object.")

        if not final_level_names:
            raise ValueError("level_names_for_Vinv must be provided when Vinv_matrix is directly supplied.")
        if actual_Vinv is not None and (actual_Vinv.shape[0] != len(final_level_names) or actual_Vinv.shape[1] != len(final_level_names)):
            raise ValueError(f"Shape mismatch for Vinv_matrix ({actual_Vinv.shape}) and level_names_for_Vinv ({len(final_level_names)}).")
        if len(set(final_level_names)) != len(final_level_names): # check for duplicates
            raise ValueError("level_names_for_Vinv contains duplicate values.")
    else:
        random_effect_type = "I"
        actual_Vinv = None

    for mt_id in qualified_model_term_ids:
        if mt_id in mme.model_term_dict:
            mme.model_term_dict[mt_id].random_type = random_effect_type
            if final_level_names:
                mme.model_term_dict[mt_id].names = final_level_names
                mme.model_term_dict[mt_id].n_levels = len(final_level_names)
        else:
            print(f"Warning: Model term ID '{mt_id}' (qualified for random effect) not found in mme.model_term_dict. This should not happen.")

    num_effects_in_G = G_matrix.shape[0] if G_matrix is not None else 1
    variance_prior_df = DefaultFloat(df_prior)

    prior_scale_matrix_for_G: Union[np.ndarray, bool] = False
    if G_matrix is not None:
        if (variance_prior_df - num_effects_in_G - 1) > 0:
            prior_scale_matrix_for_G = G_matrix * (variance_prior_df - num_effects_in_G - 1)
        else:
            print(f"Warning: Degrees of freedom {variance_prior_df} too low for {num_effects_in_G} effects to use G_prior_mean directly for prior scale. Using G_prior_mean as scale or small default.")
            prior_scale_matrix_for_G = G_matrix

    Gi_val_initial: Union[np.ndarray, bool]
    if G_matrix is not None:
        try:
            Gi_val_initial = np.linalg.inv(G_matrix.astype(DefaultFloat))
        except np.linalg.LinAlgError:
            print(f"Warning: G_prior_mean matrix for {random_term_str} is singular. Using pseudo-inverse or default.")
            Gi_val_initial = np.linalg.pinv(G_matrix.astype(DefaultFloat))
    else:
        Gi_val_initial = False

    variance_G = Variance(
        val=Gi_val_initial,
        df=variance_prior_df,
        scale=prior_scale_matrix_for_G,
        estimate_variance=estimate_variance,
        estimate_scale=False,
        constraint=constraint
    )

    GiOld_G = Variance(**vars(variance_G))
    GiNew_G = Variance(**vars(variance_G))

    random_effect = RandomEffect(
        term_array=qualified_model_term_ids,
        Gi=variance_G,
        GiOld=GiOld_G,
        GiNew=GiNew_G,
        Vinv=actual_Vinv,
        names=final_level_names,
        random_type=random_effect_type
    )
    mme.rnd_trm_vec.append(random_effect)

    if random_effect_type == "A" and mme.ped is not None:
        is_first_primary_ped_effect = not any(rt.random_type == "A" for rt in mme.rnd_trm_vec[:-1])
        if is_first_primary_ped_effect:
            mme.ped_trm_vec = qualified_model_term_ids
            print(f"Polygenic effect '{random_term_str}' using pedigree '{mme.ped.name}' registered.")

# --- MME Contributions from Random Effects ---
def add_vinv_to_lhs(mme: MME) -> None:
    """
    Adds contributions from random effects (Vinv * G_inv_scaled) to the MME LHS.
    Corresponds to Julia's addVinv.
    This function MODIFIES mme.mme_lhs in place.
    It assumes mme.mme_lhs has been initialized (e.g., as X'R_invX).
    """
    if mme.mme_lhs is None:
        raise ValueError("mme.mme_lhs must be initialized before adding random effect contributions.")

    for random_effect in mme.rnd_trm_vec:
        term_id_list = random_effect.term_array # List of ModelTerm unique IDs, e.g., ["y1:animal", "y2:animal"]

        first_term_in_effect = mme.model_term_dict.get(term_id_list[0])
        if not first_term_in_effect:
            print(f"Warning: Could not find model term {term_id_list[0]} in mme.model_term_dict. Skipping random effect {random_effect.term_array}.")
            continue

        num_levels_for_Vinv = first_term_in_effect.n_levels
        if num_levels_for_Vinv == 0 and random_effect.random_type != "I":
             if random_effect.Vinv is not None:
                 num_levels_for_Vinv = random_effect.Vinv.shape[0]
             else:
                 print(f"Error: Cannot determine size of Identity matrix for IID effect {random_effect.term_array} with 0 levels.")
                 continue

        Vi: Optional[csc_matrix]
        if random_effect.Vinv is not None:
            Vi = random_effect.Vinv
            if Vi.shape[0] != num_levels_for_Vinv or Vi.shape[1] != num_levels_for_Vinv:
                 print(f"Warning: Shape of provided Vinv ({Vi.shape}) for {random_effect.term_array} " +
                       f"does not match n_levels of first term ({num_levels_for_Vinv}). Using Vinv's shape.")
                 num_levels_for_Vinv = Vi.shape[0]
        else:
            if num_levels_for_Vinv > 0:
                 precision_dtype = DefaultFloat if (mme.mcmc_info is None or mme.mcmc_info.double_precision) else np.float32
                 Vi = sparse_eye(num_levels_for_Vinv, dtype=precision_dtype, format="csc")
            else:
                print(f"Error: Cannot create Identity V_inv for IID effect {random_effect.term_array} with 0 levels.")
                continue

        if Vi is None:
            print(f"Error: Vi (V_inverse) could not be determined for random effect {random_effect.term_array}. Skipping.")
            continue

        for i, term_i_id in enumerate(term_id_list):
            model_term_i = mme.model_term_dict.get(term_i_id)
            if not model_term_i:
                print(f"Warning: Model term {term_i_id} not found. Skipping its contribution to LHS.")
                continue

            start_pos_i = model_term_i.start_pos
            end_pos_i = start_pos_i + num_levels_for_Vinv -1

            for j, term_j_id in enumerate(term_id_list):
                model_term_j = mme.model_term_dict.get(term_j_id)
                if not model_term_j:
                    print(f"Warning: Model term {term_j_id} not found. Skipping its contribution to LHS.")
                    continue

                start_pos_j = model_term_j.start_pos
                end_pos_j = start_pos_j + num_levels_for_Vinv -1

                G_inv_ij_scalar: DefaultFloat
                if mme.n_models > 1:
                    if not isinstance(random_effect.Gi.val, np.ndarray) or random_effect.Gi.val.ndim != 2:
                        if isinstance(random_effect.Gi.val, (float, np.floating)): # type: ignore
                            G_inv_ij_scalar = DefaultFloat(random_effect.Gi.val) if i == j else 0.0
                        else:
                            print(f"Error: Gi.val for random effect {random_effect.term_array} is not a valid matrix for multi-trait. Gi.val: {random_effect.Gi.val}")
                            continue
                    else:
                        G_inv_ij_scalar = DefaultFloat(random_effect.Gi.val[i, j])
                else:
                    if mme.R.val is False or mme.R_old is None :
                        raise ValueError("Residual variances (mme.R.val, mme.R_old) not set for single-trait lambda MME LHS update.")
                    if random_effect.GiNew.val is False or random_effect.GiOld.val is False:
                        raise ValueError(f"GiNew/GiOld .val not set for random effect {random_effect.term_array} in single-trait lambda MME.")

                    gi_new_val_arr = np.atleast_2d(np.array(random_effect.GiNew.val, dtype=DefaultFloat))
                    gi_old_val_arr = np.atleast_2d(np.array(random_effect.GiOld.val, dtype=DefaultFloat))

                    current_R_val_scalar = DefaultFloat(mme.R.val)
                    old_R_val_scalar = DefaultFloat(mme.R_old)

                    G_inv_new_ij = gi_new_val_arr[i,j]
                    G_inv_old_ij = gi_old_val_arr[i,j]

                    G_inv_ij_scalar = (G_inv_new_ij * current_R_val_scalar) - (G_inv_old_ij * old_R_val_scalar)

                mme_lhs_contribution: csc_matrix = Vi * G_inv_ij_scalar

                if not issparse(mme.mme_lhs) or mme.mme_lhs.format != 'csc': # type: ignore
                    try:
                        mme.mme_lhs[start_pos_i:end_pos_i+1, start_pos_j:end_pos_j+1] += mme_lhs_contribution.toarray() if issparse(mme_lhs_contribution) else mme_lhs_contribution # type: ignore
                    except Exception as e_assign:
                        print(f"Error assigning to mme.mme_lhs block ({start_pos_i}:{end_pos_i+1}, {start_pos_j}:{end_pos_j+1}). Error: {e_assign}")
                        print(f"LHS shape: {mme.mme_lhs.shape}, Block shape: {mme_lhs_contribution.shape if mme_lhs_contribution is not None else 'None'}")

                else:
                    try:
                        current_block = mme.mme_lhs[start_pos_i:end_pos_i+1, start_pos_j:end_pos_j+1]
                        if current_block.shape != mme_lhs_contribution.shape:
                             raise ValueError(f"LHS block shape {current_block.shape} != contribution shape {mme_lhs_contribution.shape}")
                        mme.mme_lhs[start_pos_i:end_pos_i+1, start_pos_j:end_pos_j+1] = current_block + mme_lhs_contribution
                    except Exception as e_assign_sparse:
                        print(f"Error assigning to sparse mme.mme_lhs block ({start_pos_i}:{end_pos_i+1}, {start_pos_j}:{end_pos_j+1}). Error: {e_assign_sparse}")
                        print(f"LHS shape: {mme.mme_lhs.shape}, Block shape: {mme_lhs_contribution.shape if mme_lhs_contribution is not None else 'None'}")

# --- Residual Covariance Matrix (R_inv) Construction ---
def _get_R_inv_for_pattern(
    full_R_matrix: np.ndarray,
    observed_trait_pattern: np.ndarray, # Boolean array indicating observed traits
    cache_Ri_dict: Dict[Tuple[bool, ...], np.ndarray]
) -> np.ndarray:
    """
    Calculates or retrieves from cache the inverse of the residual covariance
    matrix specific to the observed traits.
    Corresponds to Julia's getRi.

    Args:
        full_R_matrix: The full (n_traits x n_traits) residual covariance matrix (R0 in Julia).
        observed_trait_pattern: Boolean array of length n_traits, True for observed.
        cache_Ri_dict: Dictionary to cache results for patterns.

    Returns:
        An (n_traits x n_traits) matrix where the block corresponding to observed
        traits contains inv(R_sub), and other elements are zero.
    """
    pattern_tuple = tuple(observed_trait_pattern)
    if pattern_tuple in cache_Ri_dict:
        return cache_Ri_dict[pattern_tuple]

    n_traits = full_R_matrix.shape[0]
    R_inv_for_pattern = np.zeros((n_traits, n_traits), dtype=full_R_matrix.dtype)

    if np.any(observed_trait_pattern):
        obs_indices = np.where(observed_trait_pattern)[0]
        R_sub = full_R_matrix[np.ix_(obs_indices, obs_indices)]

        if R_sub.size > 0:
            try:
                R_sub_inv = np.linalg.inv(R_sub)
            except np.linalg.LinAlgError:
                print(f"Warning: Submatrix of R for pattern {pattern_tuple} is singular. Using pseudo-inverse.")
                R_sub_inv = np.linalg.pinv(R_sub)

            for i_local, i_global in enumerate(obs_indices):
                for j_local, j_global in enumerate(obs_indices):
                    R_inv_for_pattern[i_global, j_global] = R_sub_inv[i_local, j_local]
        else:
             pass

    cache_Ri_dict[pattern_tuple] = R_inv_for_pattern
    return R_inv_for_pattern

def make_R_inv_sparse(
    mme: MME,
    pheno_df: pd.DataFrame,
    obs_inv_weights: Optional[np.ndarray] = None
) -> csc_matrix:
    """
    Constructs the sparse block-diagonal inverse residual covariance matrix (R_inv)
    for the entire dataset, accounting for missing data patterns.
    Corresponds to Julia's mkRi.

    Args:
        mme: The MME object, containing mme.R.val (full R0 matrix) and mme.lhs_vec (trait names).
        pheno_df: DataFrame containing phenotype data (to determine missing patterns).
        obs_inv_weights: Optional array of observation-specific inverse weights (for heterogeneous variance).
                         If None, assumes weights are 1.0. Length must be n_observations.

    Returns:
        A sparse CSC matrix representing R_inv for all observations and traits.
        Shape: (n_obs * n_traits) x (n_obs * n_traits).
    """
    if mme.R.val is False or not isinstance(mme.R.val, np.ndarray):
        raise ValueError("mme.R.val (full residual covariance matrix R0) must be set and be a NumPy array.")

    R0_matrix = np.atleast_2d(mme.R.val.astype(DefaultFloat))

    n_traits = mme.n_models
    if R0_matrix.shape != (n_traits, n_traits):
        raise ValueError(f"Shape of mme.R.val ({R0_matrix.shape}) must match (n_traits, n_traits) which is ({n_traits},{n_traits}).")

    for trait_name in mme.lhs_vec:
        if trait_name not in pheno_df.columns:
            raise ValueError(f"Trait column '{trait_name}' from mme.lhs_vec not found in phenotype DataFrame.")

    observed_patterns_matrix = ~pheno_df[mme.lhs_vec].isnull().values

    n_obs = observed_patterns_matrix.shape[0]

    if obs_inv_weights is None:
        obs_inv_weights = np.ones(n_obs, dtype=DefaultFloat)
    elif len(obs_inv_weights) != n_obs:
        raise ValueError(f"Length of obs_inv_weights ({len(obs_inv_weights)}) must match number of observations ({n_obs}).")

    Ri_cache_dict: Dict[Tuple[bool, ...], np.ndarray] = {}
    if hasattr(mme, 'res_var') and mme.res_var is not None and hasattr(mme.res_var, 'RiDict') and isinstance(mme.res_var.RiDict, dict) : # type: ignore
         Ri_cache_dict = mme.res_var.RiDict # type: ignore
    elif hasattr(mme, 'res_var') and mme.res_var is not None: # mme.res_var exists but not as expected
         print("Warning: mme.res_var found but not in expected ResVar format with RiDict. Using local cache for R_inv patterns.")

    max_nnz = n_obs * n_traits * n_traits
    row_indices = np.zeros(max_nnz, dtype=DefaultInt)
    col_indices = np.zeros(max_nnz, dtype=DefaultInt)
    data_values = np.zeros(max_nnz, dtype=DefaultFloat)

    current_pos = 0
    for i_obs in range(n_obs):
        obs_pattern = observed_patterns_matrix[i_obs, :]

        R_inv_pattern_specific = _get_R_inv_for_pattern(R0_matrix, obs_pattern, Ri_cache_dict)
        R_inv_pattern_specific *= obs_inv_weights[i_obs]

        for i_trait_local in range(n_traits):
            global_row_idx = i_trait_local * n_obs + i_obs

            for j_trait_local in range(n_traits):
                global_col_idx = j_trait_local * n_obs + i_obs

                val = R_inv_pattern_specific[i_trait_local, j_trait_local]
                if val != 0:
                    if current_pos >= max_nnz:
                        raise IndexError("Exceeded pre-allocated space for R_inv_sparse COO arrays.")
                    row_indices[current_pos] = global_row_idx
                    col_indices[current_pos] = global_col_idx
                    data_values[current_pos] = val
                    current_pos += 1

    row_indices = row_indices[:current_pos]
    col_indices = col_indices[:current_pos]
    data_values = data_values[:current_pos]

    total_dims = n_obs * n_traits
    R_inv_final_sparse = csc_matrix((data_values, (row_indices, col_indices)), shape=(total_dims, total_dims))
    R_inv_final_sparse.eliminate_zeros()

    return R_inv_final_sparse

# --- Sampling Missing Residuals ---
def sample_missing_residuals(mme: MME, current_residuals_flat: np.ndarray) -> np.ndarray:
    """
    Imputes missing residuals based on observed residuals and the model's
    residual covariance structure (mme.R.val).
    Corresponds to Julia's sampleMissingResiduals.

    Args:
        mme: The MME object, containing mme.R.val (full R0),
             and mme.missing_pattern (n_obs x n_traits boolean matrix from pheno_df.isnull()).
             It also uses mme.res_var.RiDict (or similar cache) implicitly if _get_R_inv_for_pattern
             was used to populate it, though this function re-partitions R directly.
        current_residuals_flat: A flat NumPy array of residuals (n_obs * n_traits),
                                where values for missing phenotypes are placeholders (e.g., 0 or previous sample).

    Returns:
        A flat NumPy array (n_obs * n_traits) with missing residuals imputed.
    """
    if mme.R.val is False or not isinstance(mme.R.val, np.ndarray):
        raise ValueError("mme.R.val (full residual covariance matrix R0) must be set and be a NumPy array.")
    R0_matrix = np.atleast_2d(mme.R.val.astype(DefaultFloat))

    if mme.missing_pattern is None or not isinstance(mme.missing_pattern, np.ndarray):
        raise ValueError("mme.missing_pattern (boolean matrix of observed phenotypes) is not set.")

    observed_patterns_matrix = mme.missing_pattern # True for observed, False for missing
    n_obs, n_traits = observed_patterns_matrix.shape

    if current_residuals_flat.shape[0] != n_obs * n_traits:
        raise ValueError(f"Shape of current_residuals_flat ({current_residuals_flat.shape}) does not match n_obs*n_traits ({n_obs*n_traits}).")

    residuals_by_obs = current_residuals_flat.copy().reshape((n_obs, n_traits))
    partition_cache: Dict[Any, Any] = {} # More general type for cache content

    for i_obs in range(n_obs):
        obs_pattern = observed_patterns_matrix[i_obs, :]

        if np.all(obs_pattern) or not np.any(obs_pattern):
            continue

        missing_pattern = ~obs_pattern
        pattern_key = (tuple(obs_pattern), tuple(missing_pattern))

        R_oo: Optional[np.ndarray] = None
        R_mo: Optional[np.ndarray] = None
        L_cholesky_cond_var: Optional[np.ndarray] = None

        if pattern_key in partition_cache:
            cached_data = partition_cache[pattern_key]
            if cached_data[0] is not None:
                 R_oo, R_om, R_mo, R_mm, L_cholesky_cond_var = cached_data
            else:
                continue
        else:
            obs_indices = np.where(obs_pattern)[0]
            miss_indices = np.where(missing_pattern)[0]

            R_oo = R0_matrix[np.ix_(obs_indices, obs_indices)]
            R_om = R0_matrix[np.ix_(obs_indices, miss_indices)]
            R_mo = R0_matrix[np.ix_(miss_indices, obs_indices)]
            R_mm = R0_matrix[np.ix_(miss_indices, miss_indices)]

            if R_oo.size == 0:
                continue

            try:
                R_oo_inv = np.linalg.inv(R_oo)
            except np.linalg.LinAlgError:
                print(f"Warning: R_oo submatrix for pattern {obs_pattern} is singular at obs {i_obs}. Using pseudo-inverse.")
                R_oo_inv = np.linalg.pinv(R_oo)

            conditional_variance = R_mm - R_mo @ R_oo_inv @ R_om
            conditional_variance = (conditional_variance + conditional_variance.T) / 2.0
            try:
                L_cholesky_cond_var = np.linalg.cholesky(conditional_variance)
            except np.linalg.LinAlgError:
                print(f"Warning: Conditional variance for missing residuals (obs {i_obs}, pattern {missing_pattern}) not PSD. Adding jitter.")
                jitter = np.eye(conditional_variance.shape[0]) * 1e-9
                try:
                    L_cholesky_cond_var = np.linalg.cholesky(conditional_variance + jitter)
                except np.linalg.LinAlgError:
                    print(f"Error: Cholesky decomposition failed for conditional variance even with jitter at obs {i_obs}. Skipping imputation for this obs.")
                    partition_cache[pattern_key] = (None, None, None, None, None)
                    continue
            partition_cache[pattern_key] = (R_oo, R_om, R_mo, R_mm, L_cholesky_cond_var)

        if L_cholesky_cond_var is None :
            continue

        residuals_observed_values = residuals_by_obs[i_obs, obs_pattern]

        if R_oo is not None and R_oo.size > 0:
             try:
                R_oo_inv_current = np.linalg.inv(R_oo)
             except np.linalg.LinAlgError:
                R_oo_inv_current = np.linalg.pinv(R_oo)
        else:
            continue

        if R_mo is None: continue

        conditional_mean_adjustment = R_mo @ R_oo_inv_current @ residuals_observed_values

        num_missing_for_obs = np.sum(missing_pattern)
        random_std_normal_sample = np.random.randn(num_missing_for_obs)
        sampled_deviation = L_cholesky_cond_var @ random_std_normal_sample
        imputed_missing_residuals = conditional_mean_adjustment + sampled_deviation
        residuals_by_obs[i_obs, missing_pattern] = imputed_missing_residuals

    return residuals_by_obs.ravel()

# --- Variance Component Sampling Utilities ---
def sample_variance_scalar(
    data_vector: np.ndarray,
    prior_df: DefaultFloat,
    prior_scale: DefaultFloat
) -> DefaultFloat:
    """
    Samples a scalar variance from its conditional posterior (Scaled Inverse Chi-squared).
    """
    n = len(data_vector)
    if n == 0:
        if prior_df <= 0 or prior_scale <=0:
             print("Warning: Sampling variance for empty data with non-positive prior_df/prior_scale. Returning prior_scale or small value.")
             return max(DefaultFloat(prior_scale), DefaultFloat(1e-9))
        if prior_df > 0:
            return DefaultFloat((prior_df * prior_scale) / np.random.chisquare(prior_df))
        else:
            return max(DefaultFloat(prior_scale), DefaultFloat(1e-9))

    posterior_df = prior_df + n
    sum_sq_data = np.dot(data_vector, data_vector)
    posterior_sum_of_squares = sum_sq_data + prior_df * prior_scale

    if posterior_df <= 0:
        print(f"Warning: Posterior DF ({posterior_df}) is non-positive. Returning small variance.")
        return DefaultFloat(1e-9)
    if posterior_sum_of_squares <= 0:
        print(f"Warning: Posterior Sum of Squares ({posterior_sum_of_squares}) is non-positive. Returning small variance.")
        return DefaultFloat(1e-9)

    chi_sq_sample = np.random.chisquare(posterior_df)
    if chi_sq_sample == 0:
        print("Warning: Chi-squared sample is zero. Returning large variance.")
        return DefaultFloat(1e9)

    return DefaultFloat(posterior_sum_of_squares / chi_sq_sample)

def sample_variance_scalar_weighted(
    data_vector: np.ndarray,
    prior_df: DefaultFloat,
    prior_scale: DefaultFloat,
    inv_weights: Optional[np.ndarray] = None
) -> DefaultFloat:
    """
    Samples a scalar variance, incorporating inverse weights.
    """
    if inv_weights is not None:
        if len(data_vector) != len(inv_weights):
            raise ValueError("Length of data_vector and inv_weights must match.")
        weighted_data_vector = data_vector * np.sqrt(inv_weights)
        return sample_variance_scalar(weighted_data_vector, prior_df, prior_scale)
    else:
        return sample_variance_scalar(data_vector, prior_df, prior_scale)

def sample_variance_matrix(
    data_array_list: List[np.ndarray],
    n_observations: int,
    prior_df: DefaultFloat,
    prior_scale_matrix: np.ndarray,
    inv_weights: Optional[np.ndarray] = None,
    constraint_diagonal: bool = False
) -> np.ndarray:
    """
    Samples a variance-covariance matrix from its conditional posterior.
    """
    n_traits = len(data_array_list)
    if n_observations == 0 and n_traits == 0 :
        if prior_scale_matrix.ndim == 2 and prior_scale_matrix.shape[0] == prior_scale_matrix.shape[1]:
            n_traits = prior_scale_matrix.shape[0]
            if n_traits == 0: raise ValueError("Cannot determine n_traits for zero-observation sampling.")
        else:
            raise ValueError("data_array_list is empty and prior_scale_matrix is not informative for n_traits.")

    if n_traits == 0 and n_observations > 0 :
         raise ValueError("data_array_list cannot be empty if n_observations > 0.")

    if prior_scale_matrix.shape != (n_traits, n_traits):
        if not (n_observations == 0 and n_traits == prior_scale_matrix.shape[0]):
             raise ValueError(f"Shape of prior_scale_matrix {prior_scale_matrix.shape} must be ({n_traits},{n_traits}).")

    SSCP_matrix = np.zeros((n_traits, n_traits), dtype=DefaultFloat)

    if n_observations == 0:
        if constraint_diagonal:
            sampled_variances = np.zeros(n_traits, dtype=DefaultFloat)
            for i in range(n_traits):
                if prior_df <= 0 or prior_scale_matrix[i,i] <= 0:
                    print(f"Warning: Sampling variance for trait {i} with empty data and non-positive prior. Returning small value.")
                    sampled_variances[i] = 1e-9
                    continue
                sampled_variances[i] = (prior_df * prior_scale_matrix[i,i]) / np.random.chisquare(prior_df)
            return np.diag(sampled_variances)
        else:
            try:
                from scipy.stats import invwishart
                if prior_df <= n_traits -1 :
                     print(f"Warning: Prior DF {prior_df} too low for InvWishart with {n_traits} traits. Returning prior_scale_matrix.")
                     return prior_scale_matrix
                return invwishart.rvs(df=prior_df, scale=prior_scale_matrix)
            except ImportError:
                print("Warning: scipy.stats.invwishart not available. Cannot sample from prior for matrix. Returning prior_scale_matrix.")
                return prior_scale_matrix

    data_matrix_list_cols = []
    for i_trait, arr in enumerate(data_array_list):
        if len(arr) != n_observations:
            raise ValueError(f"Trait {i_trait} data length {len(arr)} != n_observations {n_observations}")
        data_matrix_list_cols.append(arr.reshape(-1,1))

    if not data_matrix_list_cols:
        return np.diag(np.full(n_traits, 1e-9, dtype=DefaultFloat))

    data_matrix = np.hstack(data_matrix_list_cols)

    if inv_weights is not None:
        if len(inv_weights) != n_observations:
            raise ValueError("Length of inv_weights must match n_observations.")
        for i in range(n_traits):
            for j in range(i, n_traits):
                val = np.dot(data_matrix[:, i] * inv_weights, data_matrix[:, j])
                SSCP_matrix[i, j] = val
                if i != j:
                    SSCP_matrix[j, i] = val
    else:
        SSCP_matrix = data_matrix.T @ data_matrix

    posterior_df = prior_df + n_observations
    posterior_scale_matrix = prior_scale_matrix + SSCP_matrix
    posterior_scale_matrix = (posterior_scale_matrix + posterior_scale_matrix.T) / 2.0

    if constraint_diagonal:
        sampled_variances = np.zeros(n_traits, dtype=DefaultFloat)
        for i in range(n_traits):
            post_df_scalar = prior_df + n_observations
            post_sum_sq_scalar = SSCP_matrix[i,i] + prior_df * prior_scale_matrix[i,i]

            if post_df_scalar <= 0 or post_sum_sq_scalar <= 0:
                print(f"Warning: Non-positive posterior df/scale for diagonal variance {i}. Returning small value.")
                sampled_variances[i] = 1e-9
                continue

            chi_sq_sample_scalar = np.random.chisquare(post_df_scalar)
            if chi_sq_sample_scalar == 0:
                sampled_variances[i] = 1e9
            else:
                sampled_variances[i] = post_sum_sq_scalar / chi_sq_sample_scalar
        return np.diag(sampled_variances)
    else:
        try:
            from scipy.stats import invwishart
            if posterior_df <= n_traits - 1:
                print(f"Warning: Posterior DF {posterior_df} too low for InvWishart with {n_traits} traits. Returning posterior_scale_matrix scaled.")
                return posterior_scale_matrix / max(posterior_df, 1.0)
            return invwishart.rvs(df=posterior_df, scale=posterior_scale_matrix)
        except ImportError:
            print("Warning: scipy.stats.invwishart not available. Cannot sample matrix. Returning posterior_scale_matrix.")
            return posterior_scale_matrix

# --- Main Variance Component Samplers ---
def sample_general_random_effect_vcs(mme: MME, current_solution_vector: np.ndarray) -> None:
    """
    Samples variance components for general random effects (e.g., polygenic, other user-defined).
    Corresponds to Julia's sampleVCs. Updates mme.rnd_trm_vec[i].Gi (and GiNew, GiOld) in place.

    Args:
        mme: The MME object.
        current_solution_vector: The full current solution vector (effects) from MME.
    """
    if mme.rnd_trm_vec is None:
        return

    for random_effect in mme.rnd_trm_vec:
        term_id_list = random_effect.term_array
        if not term_id_list:
            continue

        first_term_obj = mme.model_term_dict.get(term_id_list[0])
        if not first_term_obj:
            print(f"Warning: First term {term_id_list[0]} for random effect not found in model_term_dict. Skipping VC sampling for this RE.")
            continue

        num_levels_re = first_term_obj.n_levels
        if random_effect.Vinv is not None:
            num_levels_re = random_effect.Vinv.shape[0]
        elif num_levels_re == 0:
             print(f"Warning: IID Random effect {random_effect.term_array} has 0 levels. Cannot sample VC. Ensure terms are in pheno data.")
             continue

        Vi: Optional[csc_matrix]
        if random_effect.Vinv is not None:
            Vi = random_effect.Vinv
        else:
            precision = DefaultFloat if (mme.mcmc_info is None or mme.mcmc_info.double_precision) else np.float32
            Vi = sparse_eye(num_levels_re, dtype=precision, format="csc")

        if Vi is None :
            print(f"Error: Vi could not be determined for random effect {random_effect.term_array}. Skipping VC sampling.")
            continue

        n_effect_terms_in_G = len(term_id_list)
        S_matrix = np.zeros((n_effect_terms_in_G, n_effect_terms_in_G), dtype=DefaultFloat)

        for i, term_i_id in enumerate(term_id_list):
            model_term_i = mme.model_term_dict.get(term_i_id)
            if not model_term_i or model_term_i.start_pos is None or model_term_i.n_levels == 0:
                print(f"Warning: ModelTerm {term_i_id} not properly initialized for VC sampling. Skipping.")
                continue # Or S_matrix entry remains 0 for this pair

            if model_term_i.n_levels != num_levels_re:
                print(f"Warning: Level mismatch for {term_i_id} ({model_term_i.n_levels}) vs V_inv ({num_levels_re}). Using V_inv's dim.")

            sol_i = current_solution_vector[model_term_i.start_pos : model_term_i.start_pos + num_levels_re]

            for j_idx in range(i, n_effect_terms_in_G):
                model_term_j = mme.model_term_dict.get(term_id_list[j_idx])
                if not model_term_j or model_term_j.start_pos is None or model_term_j.n_levels == 0:
                    print(f"Warning: ModelTerm {term_id_list[j_idx]} not properly initialized for VC sampling. Skipping.")
                    continue
                if model_term_j.n_levels != num_levels_re:
                     print(f"Warning: Level mismatch for {term_id_list[j_idx]} ({model_term_j.n_levels}) vs V_inv ({num_levels_re}). Using V_inv's dim.")

                sol_j = current_solution_vector[model_term_j.start_pos : model_term_j.start_pos + num_levels_re]

                val_S_ij = sol_i.T @ (Vi @ sol_j)
                S_matrix[i, j_idx] = val_S_ij
                if i != j_idx:
                    S_matrix[j_idx, i] = val_S_ij

        prior_df_for_G = random_effect.Gi.df # This is nu_0 for G
        prior_scale_matrix_for_G = random_effect.Gi.scale # This is S_0 for G

        if prior_df_for_G is False or prior_scale_matrix_for_G is False:
            print(f"Warning: Prior df or scale for G of random effect {random_effect.term_array} is False. Skipping VC sampling.")
            continue

        posterior_df_G = DefaultFloat(prior_df_for_G) + num_levels_re
        posterior_scale_matrix_G = np.array(prior_scale_matrix_for_G, dtype=DefaultFloat) + S_matrix
        posterior_scale_matrix_G = (posterior_scale_matrix_G + posterior_scale_matrix_G.T) / 2.0

        new_G_matrix: np.ndarray
        try:
            from scipy.stats import invwishart
            if posterior_df_G <= n_effect_terms_in_G - 1:
                print(f"Warning: Posterior DF {posterior_df_G} too low for InvWishart for G of {random_effect.term_array}. Using posterior scale matrix scaled.")
                new_G_matrix = posterior_scale_matrix_G / max(posterior_df_G, 1.0)
            else:
                new_G_matrix = invwishart.rvs(df=posterior_df_G, scale=posterior_scale_matrix_G)
        except ImportError:
            print(f"Warning: scipy.stats.invwishart not available. Using mean of posterior for G of {random_effect.term_array}.")
            new_G_matrix = posterior_scale_matrix_G / max(posterior_df_G - n_effect_terms_in_G - 1, 1.0)
        except np.linalg.LinAlgError as e_wish:
             print(f"Error sampling InvWishart for G of {random_effect.term_array}: {e_wish}. Using scaled posterior_scale_matrix.")
             new_G_matrix = posterior_scale_matrix_G / max(posterior_df_G, 1.0)

        if not (mme.mcmc_info is None or mme.mcmc_info.double_precision):
            new_G_matrix = new_G_matrix.astype(np.float32)

        new_G_matrix_2d = np.atleast_2d(new_G_matrix)

        try:
            new_G_inv_matrix = np.linalg.inv(new_G_matrix_2d)
        except np.linalg.LinAlgError:
            print(f"Warning: Sampled G matrix for {random_effect.term_array} is singular. Using pseudo-inverse.")
            new_G_inv_matrix = np.linalg.pinv(new_G_matrix_2d)

        random_effect.GiOld.val = np.array(random_effect.GiNew.val, copy=True) if random_effect.GiNew.val is not False else False
        random_effect.GiNew.val = new_G_inv_matrix
        random_effect.Gi.val = new_G_inv_matrix
        pass


def sample_marker_effect_vcs(genotype_set: Genotypes, n_traits_overall: int) -> None:
    """
    Samples variance components for marker effects within a Genotypes object.
    Corresponds to Julia's sample_marker_effect_variance. Updates Genotypes.G.val in place.
    """
    Mi = genotype_set
    inv_weights_markers: Optional[np.ndarray] = None
    active_mask_for_weighting: Optional[np.ndarray] = None # For BayesC weights

    if Mi.method == "BayesL":
        if Mi.gamma_array is None or np.any(Mi.gamma_array == 0):
            print(f"Warning: gamma_array for BayesL in {Mi.name} is None or contains zeros. Using no weights (incorrect for BayesL).")
        else:
            inv_weights_markers = 1.0 / Mi.gamma_array
    elif Mi.method == "GBLUP":
        if Mi.D is not None and not np.any(Mi.D == 0):
            inv_weights_markers = 1.0 / Mi.D

    # Determine n_traits specific to this genotype set
    n_traits_for_Mi_G = Mi.n_traits

    if n_traits_for_Mi_G == 1:
        if Mi.G.df is False or Mi.G.scale is False:
             print(f"Warning: Prior df or scale for marker effect variance G of {Mi.name} is False. Skipping VC sampling.")
             return

        if Mi.method in ["BayesC", "BayesL", "RR-BLUP", "GBLUP"]:
            if Mi.alpha is None or not Mi.alpha or Mi.alpha[0] is None:
                print(f"Warning: Marker effects Mi.alpha for {Mi.name} are not available. Skipping G sampling.")
                return
            current_marker_effects = Mi.alpha[0]

            n_active_loci: int
            effects_for_var_calc: np.ndarray
            if Mi.method == "BayesC":
                if Mi.delta is None or not Mi.delta or Mi.delta[0] is None:
                     print(f"Warning: Marker inclusion indicators Mi.delta for {Mi.name} (BayesC) not available. Using all markers.")
                     n_active_loci = Mi.n_markers
                     effects_for_var_calc = current_marker_effects
                     active_mask_for_weighting = np.ones(Mi.n_markers, dtype=bool) # All active for weights
                else:
                    active_mask_for_weighting = Mi.delta[0].astype(bool)
                    n_active_loci = int(np.sum(active_mask_for_weighting))
                    effects_for_var_calc = current_marker_effects[active_mask_for_weighting] if n_active_loci > 0 else np.array([], dtype=DefaultFloat)
            else:
                n_active_loci = Mi.n_markers
                effects_for_var_calc = current_marker_effects
                active_mask_for_weighting = np.ones(Mi.n_markers, dtype=bool) # All active for weights

            current_inv_weights = None
            if inv_weights_markers is not None:
                current_inv_weights = inv_weights_markers[active_mask_for_weighting] if (Mi.method == "BayesC" and n_active_loci > 0 and n_active_loci < Mi.n_markers) else inv_weights_markers


            Mi.G.val = sample_variance_scalar_weighted(
                effects_for_var_calc,
                DefaultFloat(Mi.G.df),
                DefaultFloat(Mi.G.scale),
                inv_weights=current_inv_weights
            )
            if Mi.method == "BayesL" and Mi.gamma_array is not None and Mi.G.val is not False and Mi.G.val > 0:
                print(f"Placeholder: BayesL gamma_array update needed for {Mi.name}")
                pass
        elif Mi.method == "BayesB":
            if Mi.beta is None or not Mi.beta or Mi.beta[0] is None:
                print(f"Warning: Marker effects Mi.beta for {Mi.name} (BayesB) not available. Skipping G sampling.")
                return
            if Mi.G.val is False or not isinstance(Mi.G.val, np.ndarray) or len(Mi.G.val) != Mi.n_markers:
                Mi.G.val = np.zeros(Mi.n_markers, dtype=DefaultFloat)
            for j in range(Mi.n_markers):
                marker_j_effect = np.array([Mi.beta[0][j]], dtype=DefaultFloat)
                Mi.G.val[j] = sample_variance_scalar(marker_j_effect, DefaultFloat(Mi.G.df), DefaultFloat(Mi.G.scale))
        else:
            print(f"Warning: Unknown marker method {Mi.method} for single-trait G sampling in {Mi.name}.")

    else: # Multi-trait context for this marker set (n_traits_for_Mi_G > 1)
        if Mi.G.df is False or Mi.G.scale is False or not isinstance(Mi.G.scale, np.ndarray):
             print(f"Warning: Prior df or scale matrix for G of {Mi.name} is False or not matrix. Skipping VC sampling.")
             return

        if Mi.method in ["RR-BLUP", "BayesC", "BayesL", "GBLUP"]:
            if Mi.alpha is None or len(Mi.alpha) != n_traits_for_Mi_G:
                print(f"Warning: Marker effects Mi.alpha for {Mi.name} (multi-trait) wrong trait number. Skipping G sampling.")
                return

            n_markers_for_sample = Mi.n_markers
            effects_list_for_var_calc: List[np.ndarray] = Mi.alpha
            active_mask_mt_for_weights: Optional[np.ndarray] = None


            if Mi.method == "BayesC":
                if Mi.delta is not None and Mi.delta[0] is not None:
                    active_mask_mt_for_weights = Mi.delta[0].astype(bool)
                    n_markers_for_sample = int(np.sum(active_mask_mt_for_weights))
                    if n_markers_for_sample == 0:
                        print(f"Warning: No active markers for BayesC multi-trait in {Mi.name}. Sampling G from prior.")
                    else:
                        effects_list_for_var_calc = [alpha_t[active_mask_mt_for_weights] for alpha_t in Mi.alpha]
                # else use all markers if delta is not informative

            current_inv_weights_mt = None
            if inv_weights_markers is not None:
                if active_mask_mt_for_weights is not None and n_markers_for_sample < Mi.n_markers : # BayesC and some inactive
                    current_inv_weights_mt = inv_weights_markers[active_mask_mt_for_weights]
                else: # Other methods or all active BayesC
                    current_inv_weights_mt = inv_weights_markers


            if n_markers_for_sample == 0 and Mi.method == "BayesC":
                 Mi.G.val = sample_variance_matrix(
                    [], 0,
                    DefaultFloat(Mi.G.df),
                    np.array(Mi.G.scale, dtype=DefaultFloat),
                    constraint_diagonal=Mi.G.constraint
                )
            else:
                Mi.G.val = sample_variance_matrix(
                    effects_list_for_var_calc,
                    n_markers_for_sample,
                    DefaultFloat(Mi.G.df),
                    np.array(Mi.G.scale, dtype=DefaultFloat),
                    inv_weights=current_inv_weights_mt,
                    constraint_diagonal=Mi.G.constraint
                )

            if Mi.method == "BayesL" and Mi.gamma_array is not None and Mi.G.val is not False:
                print(f"Placeholder: BayesL multi-trait gamma_array update needed for {Mi.name}")
                pass

        elif Mi.method == "BayesB":
            if Mi.beta is None or len(Mi.beta) != n_traits_for_Mi_G:
                print(f"Warning: Marker effects Mi.beta for {Mi.name} (BayesB multi-trait) not available. Skipping G sampling.")
                return

            if Mi.G.val is False or not isinstance(Mi.G.val, list) or len(Mi.G.val) != Mi.n_markers:
                Mi.G.val = [np.zeros((n_traits_for_Mi_G, n_traits_for_Mi_G), dtype=DefaultFloat) for _ in range(Mi.n_markers)]

            for j in range(Mi.n_markers):
                effects_vector_marker_j = np.array([Mi.beta[t][j] for t in range(n_traits_for_Mi_G)], dtype=DefaultFloat)
                data_list_for_marker_j = [np.array([effects_vector_marker_j[t]], dtype=DefaultFloat) for t in range(n_traits_for_Mi_G)]
                Mi.G.val[j] = sample_variance_matrix(
                    data_list_for_marker_j, 1,
                    DefaultFloat(Mi.G.df), np.array(Mi.G.scale, dtype=DefaultFloat),
                    constraint_diagonal=Mi.G.constraint
                )
        else:
            print(f"Warning: Unknown marker method {Mi.method} for multi-trait G sampling in {Mi.name}.")


# --- Update the __main__ block to test main VC samplers ---
if __name__ == '__main__':
    print("--- Components.py Examples ---")
    # ... (condensed previous tests for brevity, ensure MME, ModelTerm etc. are defined if needed)
    # For this test, we'll focus only on the new VC samplers.
    # print("    (Previous example outputs omitted for this test focus)")


    # --- Test Main VC Samplers ---
    print("\nTesting Main VC Samplers...")

    # 1. sample_general_random_effect_vcs
    mme_for_svc = MME(n_models=1, model_vec=["y = intercept + animal"], model_terms=[], model_term_dict={}, lhs_vec=["y"])
    mme_for_svc.mcmc_info = MCMCInfo(double_precision=True)

    n_animals = 5
    mt_i_svc = ModelTerm(i_model=0,i_trait="y",trm_str="y:intercept",n_factors=1,factors=["intercept"],start_pos=0,n_levels=1,names=["i"])
    mt_a_svc = ModelTerm(i_model=0,i_trait="y",trm_str="y:animal",n_factors=1,factors=["animal"],start_pos=1,n_levels=n_animals,names=[f"an{k}" for k in range(n_animals)])
    mme_for_svc.model_terms = [mt_i_svc, mt_a_svc]; mme_for_svc.mme_pos = 1 + n_animals
    for mt_ in mme_for_svc.model_terms: mme_for_svc.model_term_dict[mt_.trm_str] = mt_

    Gi_var_obj = Variance(val=np.array([[1/1.0]]), df=4.0, scale=np.array([[1.0*(4-1-1)]]), estimate_variance=True)
    animal_re = RandomEffect(term_array=["y:animal"], Gi=Gi_var_obj, GiOld=Variance(**vars(Gi_var_obj)), GiNew=Variance(**vars(Gi_var_obj)), random_type="I")
    mme_for_svc.rnd_trm_vec.append(animal_re)

    dummy_solutions = np.random.randn(1 + n_animals).astype(DefaultFloat)
    dummy_solutions[1:] *= np.sqrt(1.0)

    print(f"Initial Gi.val for animal RE: {animal_re.Gi.val}")
    try:
        sample_general_random_effect_vcs(mme_for_svc, dummy_solutions)
        print("sample_general_random_effect_vcs executed.")
        print(f"Updated Gi.val for animal RE: {animal_re.Gi.val}")
    except Exception as e_svc:
        print(f"Error in sample_general_random_effect_vcs test: {e_svc}")
        import traceback; traceback.print_exc()

    # 2. sample_marker_effect_vcs
    n_markers_test = 20
    marker_effects_st = np.random.randn(n_markers_test).astype(DefaultFloat) * 0.1

    geno_set_st = Genotypes(
        name="chip_st",
        n_traits=1,
        n_markers=n_markers_test,
        method="BayesC",
        alpha=[marker_effects_st],
        delta=[(np.random.rand(n_markers_test) < 0.1).astype(int)],
        G=Variance(val=1.0/0.01, df=4.0, scale=0.01 * (4.0-1-1) )
    )
    print(f"Initial marker G.val (sigma_g_inv) for {geno_set_st.name}: {geno_set_st.G.val}")
    try:
        sample_marker_effect_vcs(geno_set_st, n_traits_overall=1)
        print("sample_marker_effect_vcs executed for single-trait BayesC.")
        print(f"Updated marker G.val for {geno_set_st.name}: {geno_set_st.G.val}")
    except Exception as e_smkvc:
        print(f"Error in sample_marker_effect_vcs (ST BayesC) test: {e_smkvc}")
        import traceback; traceback.print_exc()

    n_traits_mt_markers = 2
    marker_effects_mt = [np.random.randn(n_markers_test).astype(DefaultFloat) * 0.1 for _ in range(n_traits_mt_markers)]
    G_marker_prior_mean = np.array([[0.01, 0.002],[0.002, 0.008]], dtype=DefaultFloat)
    G_marker_prior_df = 5.0
    G_marker_prior_scale = G_marker_prior_mean * (G_marker_prior_df - n_traits_mt_markers -1)

    geno_set_mt = Genotypes(
        name="chip_mt",
        n_traits=n_traits_mt_markers,
        n_markers=n_markers_test,
        method="RR-BLUP",
        alpha=marker_effects_mt,
        G=Variance(val=np.linalg.inv(G_marker_prior_mean), df=G_marker_prior_df, scale=G_marker_prior_scale, constraint=False)
    )
    print(f"Initial marker G.val (G_marker_inv) for {geno_set_mt.name}:\n{geno_set_mt.G.val}")
    try:
        sample_marker_effect_vcs(geno_set_mt, n_traits_overall=n_traits_mt_markers)
        print("sample_marker_effect_vcs executed for multi-trait RR-BLUP.")
        print(f"Updated marker G.val for {geno_set_mt.name}:\n{geno_set_mt.G.val}")
    except Exception as e_smkvc_mt:
        print(f"Error in sample_marker_effect_vcs (MT RR-BLUP) test: {e_smkvc_mt}")
        import traceback; traceback.print_exc()
