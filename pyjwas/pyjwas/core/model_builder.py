from typing import List, Dict, Any, Union, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, hstack, vstack, diags,issparse
from pyjwas.core.definitions import (
    MME, ModelTerm, Variance, Genotypes, MCMCInfo, DefaultFloat, DefaultInt
) # ResVar, RandomEffect might be needed by helpers later
from pyjwas.core.utils import get_effect_names

# Helper function to check for positive definiteness (basic version)
def is_positive_definite(matrix: np.ndarray) -> bool:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False # Not a square matrix
    if matrix.shape == (1,1): # scalar case
        return matrix[0,0] > 0
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def mk_dict(data_vector: List[Any]) -> Tuple[Dict[str, int], List[str]]:
    """
    Get column index in the incidence matrix for each level of a factor.
    Corresponds to Julia's mkDict.

    Args:
        data_vector: A list of values (e.g., ["a1","a4","a1","a2"]).

    Returns:
        A tuple containing:
            - A dictionary mapping each unique level name to its 1-based index.
            - A list of unique level names in order of appearance.
    """
    # Python typically uses 0-based indexing, but Julia's example implies 1-based for dict values.
    # For internal consistency with sparse matrix construction, 0-based is often easier.
    # Let's stick to 0-based for dict values and adjust if direct Julia porting is needed.
    unique_levels = sorted(list(set(map(str, data_vector)))) # Ensure consistent ordering
    level_map = {level: i for i, level in enumerate(unique_levels)}
    return level_map, unique_levels

def build_model(
    model_equations: str,
    R_value: Union[DefaultFloat, np.ndarray, bool] = False,
    df: DefaultFloat = 4.0,
    estimate_variance: bool = True,
    estimate_scale: bool = False, # Julia version had this, seems not fully supported there either
    constraint: bool = False, # For multi-trait, constraint=true means no residual covariance
    # NNBayes parameters (to be fully implemented later)
    num_hidden_nodes: Optional[int] = None,
    nonlinear_function: Optional[Union[str, Callable]] = None, # type: ignore
    latent_traits: Union[bool, List[str]] = False,
    user_sigma2_yobs: Union[bool, DefaultFloat] = False,
    user_sigma2_weights_NN: Union[bool, DefaultFloat] = False,
    # Censored/Categorical trait parameters
    censored_trait_names: Optional[List[str]] = None,
    categorical_trait_names: Optional[List[str]] = None,
    # Globally defined Genotypes objects (passed explicitly in Python)
    global_genotypes: Optional[Dict[str, Genotypes]] = None
) -> MME:
    """
    Builds an MME object from model equations.
    Corresponds to Julia's build_model.
    """
    if censored_trait_names is None:
        censored_trait_names = []
    if categorical_trait_names is None:
        categorical_trait_names = []
    if global_genotypes is None:
        global_genotypes = {}

    if R_value is not False:
        # Convert R_value to numpy array if it's a scalar or list for consistent checking
        r_matrix = np.array(R_value, dtype=DefaultFloat)
        if r_matrix.ndim == 0: # scalar
            r_matrix = r_matrix.reshape((1,1))
        if not is_positive_definite(r_matrix):
            raise ValueError("The residual covariance matrix R is not positive definite.")

    if not isinstance(model_equations, str) or not model_equations:
        raise ValueError("Model equations must be a non-empty string. See documentation for examples.")

    if estimate_scale:
        # Current implementation matches Julia's note: not fully supported.
        print("Warning: estimate_scale for residual variance is not fully supported.")


    # --- Bayesian Neural Network (NNBayes) initial handling (stubbed) ---
    is_nnbayes_partial = False
    original_model_equations = model_equations
    if nonlinear_function is not False and num_hidden_nodes is not None:
        print("Bayesian Neural Network parameters provided. Full NN logic will be implemented later.")
        # Placeholder for nnbayes_check_print_parameter
        # num_hidden_nodes, is_fully_connected, is_activation_fcn = nnbayes_check_print_parameter(...)
        is_fully_connected = True # Placeholder
        is_activation_fcn = isinstance(nonlinear_function, str) # Placeholder

        # Placeholder for nnbayes_model_equation
        # model_equations = nnbayes_model_equation(original_model_equations, num_hidden_nodes, is_fully_connected)
        print(f"Warning: NNBayes model equation rewriting is stubbed. Using original: {model_equations}")

        is_nnbayes_partial = not is_fully_connected # Simplified
    # --- End NNBayes stub ---

    model_eq_list = [eq.strip() for eq in model_equations.splitlines() if eq.strip()] # Handles ';' or newline
    if not model_eq_list:
        model_eq_list = [eq.strip() for eq in model_equations.split(';') if eq.strip()]

    n_models = len(model_eq_list)

    if R_value is not False:
        r_matrix_for_check = np.array(R_value, dtype=DefaultFloat)
        if r_matrix_for_check.ndim == 0:
             r_matrix_for_check = r_matrix_for_check.reshape((1,1))
        if r_matrix_for_check.shape[0] != n_models:
            raise ValueError(f"The residual covariance matrix R is not {n_models}x{n_models}.")

    all_lhs_vec: List[str] = []
    all_model_terms: List[ModelTerm] = []
    model_term_dict: Dict[str, ModelTerm] = {}

    for model_idx, model_eq_str in enumerate(model_eq_list):
        if "=" not in model_eq_str:
            raise ValueError(f"Model equation '{model_eq_str}' is missing '='.")
        lhs_str, rhs_str = [part.strip() for part in model_eq_str.split("=", 1)]

        if not lhs_str:
            raise ValueError(f"LHS cannot be empty in '{model_eq_str}'.")
        all_lhs_vec.append(lhs_str)

        rhs_terms_str = [term.strip() for term in rhs_str.split("+") if term.strip()]

        for term_formula_str in rhs_terms_str: # e.g., "A" or "A*B"
            factors_in_term = [factor.strip() for factor in term_formula_str.split("*") if factor.strip()]
            full_term_id = f"{lhs_str}:{term_formula_str}" # Unique ID like "y1:A" or "y1:A*B"

            mt = ModelTerm(
                i_model=model_idx,      # 0-indexed model
                i_trait=lhs_str,
                trm_str=full_term_id,   # This is "y1:A", Julia's trmStr was "A" then prefixed. Let's make it unique from start.
                n_factors=len(factors_in_term),
                factors=factors_in_term # List of factor names like ["A", "B"]
            )
            all_model_terms.append(mt)
            model_term_dict[full_term_id] = mt

    # Process Genotypes objects
    mme_genotypes_list: List[Genotypes] = []
    # Iterms to remove from model_terms if they are genotype terms
    genotype_term_ids_to_remove = []

    for term_id, term_obj in model_term_dict.items():
        # A term like "y1:geno_A" where "geno_A" is a factor.
        # We check if "geno_A" (the last factor) corresponds to a Genotypes object.
        potential_geno_name = term_obj.factors[-1]
        if potential_geno_name in global_genotypes:
            term_obj.random_type = "genotypes"
            geno_data_template = global_genotypes[potential_geno_name]

            # Create a copy or specific instance for this model/trait context if needed
            # For now, let's assume one Genotypes object can serve multiple model terms if its name matches

            # Check if this genotype (by name) is already added to mme_genotypes_list
            existing_geno_obj = next((g for g in mme_genotypes_list if g.name == geno_data_template.name), None)

            if not existing_geno_obj:
                # If this is the first time we see this named Genotype object in the model equations
                current_geno_obj = Genotypes(**vars(geno_data_template)) # Make a copy
                current_geno_obj.name = potential_geno_name # Ensure name is set from key

                current_geno_obj.n_traits = 1 if is_nnbayes_partial else n_models
                current_geno_obj.trait_names = [term_obj.i_trait] if is_nnbayes_partial else all_lhs_vec

                if n_models != 1 and current_geno_obj.G.df is not False: # type: ignore
                    current_geno_obj.G.df += n_models # type: ignore

                # Check R_value compatibility
                r_val_for_check = R_value
                if r_val_for_check is not False:
                    r_val_np = np.array(r_val_for_check)
                    if r_val_np.ndim == 0: r_val_np = r_val_np.reshape((1,1))

                    g_val_np = np.array(current_geno_obj.G.val) if current_geno_obj.G.val is not False else None
                    gv_val_np = np.array(current_geno_obj.genetic_variance.val) if current_geno_obj.genetic_variance.val is not False else None

                    if not is_nnbayes_partial:
                        if (g_val_np is not None and g_val_np.ndim > 0 and g_val_np.shape[0] != r_val_np.shape[0]) or \
                           (gv_val_np is not None and gv_val_np.ndim > 0 and gv_val_np.shape[0] != r_val_np.shape[0]):
                             raise ValueError(f"Genomic covariance matrix for '{current_geno_obj.name}' is not compatible with R matrix size ({r_val_np.shape[0]}x{r_val_np.shape[0]}).")
                mme_genotypes_list.append(current_geno_obj)

            genotype_term_ids_to_remove.append(term_id)

    # Remove genotype terms from the main model_terms list and dict as they are handled in MME.M
    all_model_terms = [mt for mt in all_model_terms if mt.trm_str not in genotype_term_ids_to_remove]
    for term_id in genotype_term_ids_to_remove:
        if term_id in model_term_dict:
            del model_term_dict[term_id]

    # Set scale and df for residual variance
    # Note: Julia code has different logic for df_R (df + nModels for MT).
    # Inverse Wishart E[Sigma] = S / (nu - p - 1). If S is scale_R, nu is df_R, p is n_models.
    # Prior mean R_value = scale_R / (df_R - n_models -1)
    # So scale_R = R_value * (df_R - n_models -1)
    # Julia: scale_R = R*(df - 2)/df for ST, R*(df-1) for MT. This seems to be S_0 from Gelman BDA for InvWish.
    # Let's follow Julia's direct calculation for now.

    current_R_val = np.array(R_value, dtype=DefaultFloat) if R_value is not False else False

    if n_models == 1:
        scale_R = (current_R_val * (df - 2) / df) if R_value is not False else False
        df_R = df
    else: # multi-trait
        scale_R = (current_R_val * (df - (n_models-1) )) if R_value is not False else False # This is a guess to make it more standard, Julia had df-1
        df_R = df + n_models -1 # Julia had df + nModels. Standard Inv-Wishart is df_prior + p - 1 for posterior. Let's use Julia's for now.
        df_R = df + n_models # Reverting to closer to Julia's for now.

    residual_variance = Variance(
        val=current_R_val,
        df=DefaultFloat(df_R),
        scale=scale_R if scale_R is not False else False,
        estimate_variance=estimate_variance,
        estimate_scale=estimate_scale, # Julia: estimate_scale for residual is not supported
        constraint=constraint
    )

    mme = MME(
        n_models=n_models,
        model_vec=model_eq_list,
        model_terms=all_model_terms,
        model_term_dict=model_term_dict,
        lhs_vec=all_lhs_vec,
        R=residual_variance,
        M=mme_genotypes_list,
        mcmc_info=MCMCInfo() # Default MCMCInfo
    )

    # NNBayes specific MME modifications (stubbed)
    if nonlinear_function is not False and num_hidden_nodes is not None:
        # mme.is_fully_connected = is_fully_connected
        # mme.is_activation_fcn = is_activation_fcn
        # mme.nonlinear_function = ...
        # mme.latent_traits = latent_traits
        # mme.sigma2_yobs = ...
        # mme.sigma2_weights_NN = ...
        # mme.fixed_sigma2_NN = ...
        pass

    # Setup traits_type
    mme.traits_type = ["continuous"] * n_models
    for i, trait_name in enumerate(all_lhs_vec):
        if trait_name in censored_trait_names:
            mme.traits_type[i] = "censored"
        elif trait_name in categorical_trait_names:
            mme.traits_type[i] = "categorical"

    return mme

def set_covariate(mme: MME, *covariate_names: str) -> None:
    """
    Sets variables as covariates in the MME object.
    Covariates are continuous variables. Default is factor (categorical).
    Args:
        mme: The MME object to modify.
        *covariate_names: Strings of covariate names, can be space-separated.
    """
    parsed_covs = []
    for item in covariate_names:
        parsed_covs.extend(item.split())

    # Add to mme.cov_vec if not already present
    for cov_name in parsed_covs:
        if cov_name not in mme.cov_vec:
            mme.cov_vec.append(cov_name)

def get_data_for_term(term: ModelTerm, df: pd.DataFrame, mme: MME) -> None:
    """
    Populates term.data (string representation) and term.val (numerical values)
    from the DataFrame based on the term's factors and whether they are covariates.
    Modifies the term object in-place.
    Corresponds to Julia's getData.
    """
    n_obs = df.shape[0]

    if not term.factors: # Should not happen with proper parsing
        term.data = [""] * n_obs
        term.val = np.zeros(n_obs, dtype=DefaultFloat)
        return

    # Handle intercept separately for clarity
    if term.factors == ["intercept"]:
        term.data = ["intercept"] * n_obs
        term.val = np.ones(n_obs, dtype=DefaultFloat)
        return

    # Initial factor
    current_factor_name = term.factors[0]
    if current_factor_name in mme.cov_vec:
        if not pd.api.types.is_numeric_dtype(df[current_factor_name]):
            raise TypeError(f"Covariate '{current_factor_name}' must have a numeric data type in DataFrame.")
        term.data = [current_factor_name] * n_obs
        term.val = df[current_factor_name].values.astype(DefaultFloat)
    else: # Categorical factor
        term.data = df[current_factor_name].astype(str).tolist()
        term.val = np.ones(n_obs, dtype=DefaultFloat)

    # Subsequent factors (for interactions)
    for i in range(1, term.n_factors):
        factor_name = term.factors[i]
        if factor_name in mme.cov_vec:
            if not pd.api.types.is_numeric_dtype(df[factor_name]):
                raise TypeError(f"Covariate '{factor_name}' must have a numeric data type in DataFrame.")
            term.data = [f"{d}*{factor_name}" for d in term.data]
            term.val = term.val * df[factor_name].values.astype(DefaultFloat)
        else: # Categorical factor
            term.data = [f"{d}*{f_val}" for d, f_val in zip(term.data, df[factor_name].astype(str))]
            # term.val remains multiplied by 1.0 (already handled)

    # Replace NaNs in term.val with 0.0 (as in Julia's coalesce.(val, 0.0))
    term.val = np.nan_to_num(term.val, nan=0.0)

    # Ensure correct float type based on MCMCInfo (if available and set)
    if mme.mcmc_info and not mme.mcmc_info.double_precision:
        term.val = term.val.astype(np.float32)
    else:
        term.val = term.val.astype(DefaultFloat)


def _get_factor_levels_from_data_str(data_str: str) -> List[str]:
    """Helper to split interaction strings like 'A1*B2' into ['A1', 'B2']"""
    return [s.strip() for s in data_str.split('*')]


def build_term_incidence_matrix(term: ModelTerm, df_shape_0: int, mme: MME) -> None:
    """
    Constructs the sparse incidence matrix (term.X) for a given ModelTerm.
    Modifies term.X, term.n_levels, term.names, term.start_pos in-place.
    Updates mme.mme_pos.
    Corresponds to Julia's getX.
    """
    n_obs_total = df_shape_0 # Total observations in the current trait's context

    # Row indices: observations for the current model (trait)
    # Julia: xi = (term.i_model-1)*n_obs .+ collect(1:n_obs)
    # Python: 0-indexed rows for this specific term's contribution to the trait's block
    row_indices = np.arange(n_obs_total)

    # Values for sparse matrix
    values = term.val

    # Column indices depend on levels
    # Clean data strings: replace "factor*missing" or "missing*factor" with "missing"
    # This ensures "missing" is treated as a single level for indexing
    term_data_cleaned = []
    for data_str in term.data:
        if "missing" in _get_factor_levels_from_data_str(data_str):
            term_data_cleaned.append("missing")
        else:
            term_data_cleaned.append(data_str)

    col_indices = np.zeros(n_obs_total, dtype=DefaultInt)

    if term.random_type == "I" or term.random_type == "fixed": # Fixed effects or i.i.d. random
        # Get unique levels from the cleaned data (excluding "missing" for dict creation, then add it)
        valid_data_levels = [d for d in term_data_cleaned if d != "missing"]
        level_map, term.names = mk_dict(valid_data_levels if valid_data_levels else ["_placeholder_if_all_missing_"]) # Ensure mk_dict gets a list
        if not valid_data_levels and "missing" in term_data_cleaned : # handles case where only "missing" is present
             term.names = ["missing_placeholder_level"] # give it a name so n_levels is 1
             level_map={"missing_placeholder_level":0}


        term.n_levels = len(term.names)

        current_level_idx = 0
        final_level_map = {}
        for level_name in term.names:
            final_level_map[level_name] = current_level_idx
            current_level_idx +=1

        # Add "missing" to map, pointing to a valid index (e.g., first level) but its values will be zeroed out
        # Julia used dict["missing"] = 1 (1-based). For 0-based, it could be 0.
        # This column will effectively not contribute for "missing" data due to value zeroing.
        missing_col_idx_for_missing_data = 0 # Arbitrary, as value is 0
        if not term.names: # If all data was missing, and no placeholder created
            term.n_levels = 1 # Create a dummy level for missing
            term.names = ["_all_missing_dummy_level_"]
            final_level_map[term.names[0]] = 0
            missing_col_idx_for_missing_data = 0


        for i, data_str in enumerate(term_data_cleaned):
            if data_str == "missing":
                col_indices[i] = missing_col_idx_for_missing_data
                values[i] = 0.0 # Crucial: zero out value for missing data
            else:
                col_indices[i] = final_level_map[data_str]

    elif term.random_type == "V" or term.random_type == "A": # Pedigree or other specific random effects
        # Assumes term.names (e.g., all animal IDs from pedigree) and term.n_levels are pre-populated
        # by set_random() or similar logic before calling get_mme_components.
        if not term.names or term.n_levels == 0:
            raise ValueError(f"Term '{term.trm_str}' is random type '{term.random_type}' but has no pre-defined levels/names.")

        level_map = {name: i for i, name in enumerate(term.names)}

        # Extract the relevant part of the interaction string that matches a level in term.names
        # e.g. if term.data[i] is "animal123*age_group2" and "animal123" is in term.names
        for i, full_data_str in enumerate(term_data_cleaned):
            if full_data_str == "missing":
                col_indices[i] = 0 # Arbitrary, value is zero
                values[i] = 0.0
            else:
                # For interactions like "animal*age_class", "animal" is the one with levels in term.names
                found_level = False
                for factor_level_part in _get_factor_levels_from_data_str(full_data_str):
                    if factor_level_part in level_map:
                        col_indices[i] = level_map[factor_level_part]
                        found_level = True
                        break
                if not found_level:
                    # This observation's specific level for the random effect is not in the predefined list.
                    # This could be an error, or imply it should be treated as missing for this random effect.
                    # For now, treat as missing (value becomes 0).
                    # print(f"Warning: Level part from '{full_data_str}' not found in term.names for '{term.trm_str}'. Treating as missing for this term.")
                    col_indices[i] = 0 # Arbitrary
                    values[i] = 0.0
    else:
        raise ValueError(f"Unknown term random_type: {term.random_type} for term {term.trm_str}")

    # Create the sparse incidence matrix for this term's contribution to one trait block
    # The number of columns is term.n_levels.
    # The number of rows is n_obs_total (observations for the current trait).
    # This X is for a single trait. It will be correctly placed within the multi-trait X later.
    if term.n_levels == 0 : # if a factor has no levels (e.g. all missing and no placeholder)
        # Create an empty sparse matrix of the correct shape (n_obs_total x 0)
        # This might need to be (n_obs_total x 1) with all zeros if downstream expects some columns.
        # For now, (n_obs_total x 0) is more accurate if there are truly no levels.
        # However, Julia adds a dummy column for missing. Let's ensure n_levels is at least 1.
         term.n_levels = 1 # Ensure at least one column, even if it's all zeros
         # This case should ideally be handled by placeholder logic in level_map creation.


    term.X = csc_matrix((values, (row_indices, col_indices)), shape=(n_obs_total, term.n_levels))
    term.X.eliminate_zeros() # Good practice

    term.start_pos = mme.mme_pos # 0-indexed start column in the full X matrix (across all terms for one trait)
    mme.mme_pos += term.n_levels


def get_mme_components(mme: MME, df: pd.DataFrame) -> None:
    """
    Constructs the main components of the MME (X, y_sparse, mme_lhs, mme_rhs).
    Corresponds to Julia's getMME.
    """
    if mme.mme_lhs is not None: # Check if already built
        print("MME components seem to be already built. Re-building.")
        # raise ValueError("MME components already built. Call build_model() for a new model.")

    n_total_obs = df.shape[0] # Total rows in the dataframe

    # Heterogeneous residuals weights
    if mme.mcmc_info.heterogeneous_residuals:
        if "weights" not in df.columns:
            raise ValueError("DataFrame must contain 'weights' column for heterogeneous_residuals.")
        inv_weights_flat = 1.0 / df["weights"].values.astype(DefaultFloat)
    else:
        inv_weights_flat = np.ones(n_total_obs, dtype=DefaultFloat)

    if not mme.mcmc_info.double_precision:
        inv_weights_flat = inv_weights_flat.astype(np.float32)
    mme.inv_weights = inv_weights_flat # Store the flat (n_obs x 1) array

    # Prepare incidence matrices (X) for each term, for each trait block
    # This needs to be done carefully for multi-trait models.
    # The X matrix in MME will be a block diagonal matrix if traits don't share fixed effects structure,
    # or a more complex concatenation if they do.
    # JWAS.jl appears to build X for all traits stacked, then uses Ri.

    # First, ensure all model terms have their data and raw X computed for a single trait context
    mme.mme_pos = 0 # Reset position counter for the current trait's X block
    all_term_X_blocks_for_trait = [] # List to hold individual term X matrices

    for term in mme.model_terms:
        if term.X is None: # If not already populated (e.g. by a previous call or direct setup)
            get_data_for_term(term, df, mme) # Populates term.data and term.val
            build_term_incidence_matrix(term, n_total_obs, mme) # Populates term.X, term.n_levels, term.names
        all_term_X_blocks_for_trait.append(term.X)

    # Concatenate X matrices for all terms horizontally for a single trait structure
    if not all_term_X_blocks_for_trait:
         # This means only an intercept or no fixed/covariate effects were defined beyond genotypes
        # Create a dummy X of shape (n_total_obs, 0) or handle intercept explicitly if it's the only term.
        # If model_terms is empty (e.g. y = G), then X_single_trait should be empty or handle intercept.
        # For now, assume if all_model_terms is empty, X is effectively zero columns for fixed effects.
        X_single_trait_structure = csc_matrix((n_total_obs, 0), dtype=DefaultFloat)
    else:
        X_single_trait_structure = hstack(all_term_X_blocks_for_trait, format="csc")


    # Construct the full X and y_sparse for all traits
    # y is stacked: [y_trait1; y_trait2; ...]
    # X is block-diagonal if effects are trait-specific, or more complex if shared.
    # JWAS approach: X is (n_obs * n_traits) x (sum of all effect levels)
    # Each term.X is n_obs x n_levels_for_term.
    # The final X needs to map observations for each trait to the correct columns.

    list_X_trait_blocks = []
    list_y_trait_blocks = []

    # The mme.model_terms are already associated with a specific i_model (trait index)
    # We need to build the full X matrix considering this.
    # The logic in Julia's getX: term.X = sparse(xi,xj,xv,nObs*nModels,trm.nLevels)
    # This implies each term.X is already global. Let's adjust build_term_incidence_matrix
    # No, Julia getX: xi = (trm.iModel-1)*nObs .+ collect(1:nObs)
    # This means row indices are already global.
    # So, build_term_incidence_matrix should take total_obs_all_traits = n_total_obs * mme.n_models
    # And use term.i_model to offset row_indices.

    # Let's refine: each term.X is specific to its trait block first, then combine.
    # This is more modular. The Julia code seems to make term.X global from the start.

    # Re-think: The current term.X is n_obs x n_levels_for_term.
    # The final X matrix for MME:
    # It has n_total_obs * mme.n_models rows.
    # It has mme.mme_pos columns (sum of n_levels for all unique terms across all traits).
    # This structure matches Julia's approach.

    # Let's assume mme.model_terms contains ALL terms for ALL traits, and mme.mme_pos is the total number of columns.
    # We need to reconstruct the global X and y.

    # Global X construction
    # Each term in mme.model_terms already has term.i_model (trait index) and term.start_pos
    # The term.X is n_total_obs x term.n_levels

    # This part needs careful alignment with how Julia constructs the *final* X matrix.
    # If term.X is already (n_obs_all_traits x n_levels_for_that_term_instance), then hstack is fine.
    # Julia's getX: `trm.X = sparse(xi,xj,xv,nObs*nModels,trm.nLevels)` where xi are global row indices.
    # This means individual `term.X` are already shaped for the global MME.
    # So, the python `build_term_incidence_matrix` needs to be adjusted to create such global term.X.

    # Let's adjust build_term_incidence_matrix and get_data_for_term to work on global MME scope.
    # This is a significant change. For now, let's assume the simplified X_single_trait_structure
    # and adapt for multi-trait MME LHS/RHS construction.

    # Simplified X for now (assuming effects are not shared across traits in X columns for this step)
    # This will need to be expanded for true multi-trait shared effects.
    if mme.n_models > 1:
        # This is a placeholder. True multi-trait X is more complex.
        # For now, assume X_single_trait_structure is for the first trait, or needs replication.
        # This part is complex and needs to correctly map effects for each trait.
        # JWAS's X is (N_obs * N_traits) x (Total_levels_across_all_terms_and_traits)
        # Each term in model_terms defines its columns.
        # The global X is essentially a hstack of these, where each original term.X
        # needs to be built with global row indices.

        # Let's defer full X construction to a dedicated function and assume X_single_trait_structure for now for LHS/RHS logic
        mme.X = X_single_trait_structure # Placeholder - this is not the full multi-trait X
        print("Warning: Full multi-trait X matrix construction is simplified. Assuming effects apply to first trait or need specific handling.")
    else:
        mme.X = X_single_trait_structure

    # Construct y_sparse (N_obs * N_traits x 1)
    y_stacked_list = []
    for i_model in range(mme.n_models):
        trait_name = mme.lhs_vec[i_model]
        y_trait = df[trait_name].fillna(0.0).values.astype(DefaultFloat) # Fill NA with 0
        if not mme.mcmc_info.double_precision:
            y_trait = y_trait.astype(np.float32)
        y_stacked_list.append(y_trait)

    y_stacked = np.concatenate(y_stacked_list)
    mme.y_sparse = csc_matrix(y_stacked.reshape(-1, 1)) # Make it a column vector

    # MME LHS and RHS
    # This requires the full X and proper R_inv (Ri in Julia)

    if mme.n_models == 1:
        # R_inv for single trait is just 1/sigma_e^2, applied via inv_weights
        # R_inv_diag = mme.inv_weights / mme.R.val # If R.val is sigma_e^2
        # For now, assume inv_weights already incorporates 1/R if R is scalar.
        # More typically, R_inv is block diagonal with inv_weights on diagonal of each block.
        R_inv_sparse = diags(mme.inv_weights, format="csc")
        mme.mme_lhs = mme.X.T @ R_inv_sparse @ mme.X
        mme.mme_rhs = mme.X.T @ R_inv_sparse @ mme.y_sparse
    else: # Multi-trait
        if mme.R.constraint or mme.R.val is False or not isinstance(mme.R.val, np.ndarray) or mme.R.val.shape[0] != mme.n_models:
            # Diagonal R_inv: each trait independent given its inv_weight
            # inv_weights_all_traits = np.concatenate([mme.inv_weights / mme.R.val[i,i] for i in range(mme.n_models)]) # Needs R.val to be matrix
            # This simplification assumes R is diagonal.
            # If R.val is just a scalar True/False, this is problematic.
            # Let's assume R.val holds the (n_traits x n_traits) matrix if not constrained.

            # If constrained or R not properly matrix, treat as block diagonal with 1/var_i for each trait
            # This is a simplification. Julia's mkRi is more sophisticated.
            if isinstance(mme.R.val, np.ndarray) and mme.R.val.ndim ==2 and not mme.R.constraint:
                 # Full R matrix provided
                try:
                    R_inv_block = np.linalg.inv(mme.R.val)
                except np.linalg.LinAlgError:
                    raise ValueError("Residual covariance matrix R is singular.")

                # Construct sparse block diagonal Ri from R_inv_block, element-wise multiplied by inv_weights
                # This is complex. For now, approximate or use simpler version.
                # This section needs the mkRi equivalent.
                print("Warning: Full multi-trait Ri construction (from R matrix and inv_weights) is complex and currently simplified.")
                # Simplified: Assume R is diagonal if constraint is true, or use simple diagonal scaling.
                full_R_inv_diag_vals = []
                for i in range(mme.n_models):
                    # This assumes R.val is diagonal, or we take diagonal part if constrained.
                    # If R.val is False, this part is ill-defined.
                    r_ii = mme.R.val[i,i] if isinstance(mme.R.val, np.ndarray) and mme.R.val.ndim==2 else (mme.R.val if isinstance(mme.R.val, float) else 1.0)
                    if r_ii == 0: r_ii = 1e-9 # Avoid division by zero, though should come from positive definite R
                    full_R_inv_diag_vals.extend(mme.inv_weights / r_ii)
                Ri_sparse = diags(np.array(full_R_inv_diag_vals), format="csc")

            else: # R.constraint is true or R.val is not a proper matrix
                full_R_inv_diag_vals = []
                for i in range(mme.n_models):
                    # Assuming R.val might be scalar if not a matrix, or default to 1.0 for structure
                    # This part is very approximate without mkRi
                    r_val_diag_i = 1.0 # Default if R.val is not informative for diagonal element
                    if isinstance(mme.R.val, np.ndarray) and mme.R.val.ndim == 0: # scalar R for multi-trait (unusual)
                        r_val_diag_i = mme.R.val
                    elif isinstance(mme.R.val, np.ndarray) and mme.R.val.ndim == 2: # matrix R but constraint=true
                        r_val_diag_i = mme.R.val[i,i]
                    elif isinstance(mme.R.val, float):
                         r_val_diag_i = mme.R.val

                    if r_val_diag_i == 0: r_val_diag_i = 1e-9
                    full_R_inv_diag_vals.extend(mme.inv_weights / r_val_diag_i) # inv_weights is per obs, R_val per trait
                Ri_sparse = diags(np.array(full_R_inv_diag_vals), format="csc")

        else: # Full R matrix, non-constrained
            # This is where the mkRi logic from Julia is essential.
            # mkRi creates a sparse block diagonal matrix where each block is an inverse of a sub-matrix of R,
            # corresponding to the observed traits for that individual.
            # This is too complex to implement fully here without more context/effort.
            # Fallback to simplified diagonal version for now.
            print("Warning: Full multi-trait Ri for non-constrained R with missing data patterns is not yet implemented. Using simplified diagonal R_inv.")
            full_R_inv_diag_vals = []
            for i in range(mme.n_models):
                 full_R_inv_diag_vals.extend(mme.inv_weights / mme.R.val[i,i])
            Ri_sparse = diags(np.array(full_R_inv_diag_vals), format="csc")

        # This X needs to be the globally structured X.
        # Using the simplified mme.X for now.
        # If mme.X is (N_obs x TotalFixedLevels), and Ri is (N_obs*N_traits x N_obs*N_traits)
        # this multiplication is not directly compatible.
        # This indicates a mismatch in how X is being formed vs. how Ri is expected.
        # Julia's X is (N_obs*N_traits x TotalFixedLevels). Ri is (N_obs*N_traits x N_obs*N_traits).
        # The current python mme.X is (N_obs x TotalFixedLevels). This needs to be fixed.

        # For now, this will likely error or give wrong dimensions.
        # This highlights the need to ensure mme.X is correctly formed for multi-trait.
        # Assuming mme.X was correctly formed globally (as Julia implies individual term.X are global):
        # mme.mme_lhs = mme.X.T @ Ri_sparse @ mme.X
        # mme.mme_rhs = mme.X.T @ Ri_sparse @ mme.y_sparse
        # If mme.X is still (N_obs x P), and we need to make it (N_obs*N_traits x P*N_traits) or similar:
        # This requires Kronecker products or careful stacking.
        # Example: X_global = scipy.sparse.krons(scipy.sparse.eye(mme.n_models), mme.X) if effects repeat per trait
        # This is a large architectural decision for multi-trait.

        # Fallback to erroring out until X construction is fixed for multi-trait.
        raise NotImplementedError("Multi-trait MME LHS/RHS construction requires X to be globally structured for all traits. This is not fully implemented with the current X construction.")


    # Add contributions from random effects (G_inv part) to mme.mme_lhs
    # This would call an equivalent of Julia's addVinv(mme)
    # For now, this is a placeholder.
    # add_random_effects_to_lhs(mme) # This function would modify mme.mme_lhs

    if issparse(mme.mme_lhs): mme.mme_lhs.eliminate_zeros()
    if issparse(mme.mme_rhs): mme.mme_rhs.eliminate_zeros()

    # Check for zero diagonals in LHS (as in Julia)
    # if isinstance(mme.mme_lhs, np.ndarray) or issparse(mme.mme_lhs):
    #     diag_lhs = mme.mme_lhs.diagonal()
    #     if np.any(diag_lhs == 0):
    #         zero_diag_indices = np.where(diag_lhs == 0)[0]
    #         # effect_names = get_effect_names(mme) # Needs implementation
    #         # problematic_effects = [effect_names[i] for i in zero_diag_indices]
    #         # print(f"Warning: Zero diagonal elements in MME LHS for effects: {problematic_effects}")
    #         # This usually indicates no data for certain levels. Julia errors here.
    #         # For now, let's just warn.
    #         print(f"Warning: Zero diagonal elements found in MME LHS at indices: {zero_diag_indices}.")


if __name__ == '__main__':
    # Example Usage
    print("--- Model Builder Examples ---")

    # Dummy Genotypes object for testing build_model
    geno_A_data = Genotypes(name="genoA", n_markers=100, method="BayesC")
    global_geno_objects = {"genoA": geno_A_data}

    # Single-trait model
    st_model_eq = "yield = intercept + parity + age + genoA"
    try:
        mme_st = build_model(st_model_eq, R_value=100.0, global_genotypes=global_geno_objects)
        set_covariate(mme_st, "age")
        print(f"Single-trait MME built for: {mme_st.model_vec}")
        print(f"Covariates: {mme_st.cov_vec}")
        print(f"LHS: {mme_st.lhs_vec}")
        print(f"Model Terms ({len(mme_st.model_terms)}):")
        for mt in mme_st.model_terms:
            print(f"  {mt.trm_str}, Factors: {mt.factors}, RandomType: {mt.random_type}")
        print(f"Genotype Sets ({len(mme_st.M)}):")
        for g in mme_st.M:
            print(f"  Name: {g.name}, Method: {g.method}, Traits: {g.trait_names}")

    except Exception as e:
        print(f"Error in ST model: {e}")

    # Multi-trait model
    mt_model_eq = """
    milk_yield = intercept + herd + lactation_stage + age + genoA
    fat_percent = intercept + herd + age + genoA
    """
    R_mt = np.array([[10.0, 1.5], [1.5, 0.5]])
    try:
        mme_mt = build_model(mt_model_eq, R_value=R_mt, global_genotypes=global_geno_objects)
        set_covariate(mme_mt, "age")
        print(f"Multi-trait MME built for: {mme_mt.model_vec}")
        print(f"Covariates: {mme_mt.cov_vec}")
        print(f"LHS: {mme_mt.lhs_vec}")
        print(f"Model Terms ({len(mme_mt.model_terms)}):")
        for mt_term in mme_mt.model_terms:
             print(f"  {mt_term.trm_str}, Factors: {mt_term.factors}, Trait: {mt_term.i_trait}")
        print(f"Genotype Sets ({len(mme_mt.M)}):")
        for g_mt in mme_mt.M:
            print(f"  Name: {g_mt.name}, Method: {g_mt.method}, Traits: {g_mt.trait_names}")


        # Example data for get_mme_components
        example_data_mt = pd.DataFrame({
            'milk_yield': np.random.rand(10) * 100,
            'fat_percent': np.random.rand(10) * 4 + 1,
            'herd': np.random.choice(['H1', 'H2', 'H3'], 10),
            'lactation_stage': np.random.choice(['L1', 'L2'], 10),
            'age': np.random.rand(10) * 5 + 2,
            'genoA_id': [f"id_{i}" for i in range(10)] # Assuming genoA links via animal ID
        })
        # This will error due to NotImplementedError for multi-trait LHS/RHS
        # get_mme_components(mme_mt, example_data_mt)
        # print("get_mme_components called for multi-trait (simplified).")

    except Exception as e:
        print(f"Error in MT model: {e}")

    # Test get_data_for_term and build_term_incidence_matrix
    if 'mme_st' in locals():
        try:
            dummy_df_st = pd.DataFrame({
                'yield': [10,12,11,13,14],
                'parity': ['1','2','1','2','1'],
                'age': [2.1, 3.0, 2.2, 3.1, 2.3],
                'genoA': ['id1','id2','id3','id4','id5'] # This is just a placeholder column name
            })
            mme_st.mcmc_info = MCMCInfo() # Ensure MCMCInfo is present

            # Example: Process 'intercept' term
            intercept_term = next(t for t in mme_st.model_terms if "intercept" in t.factors)
            get_data_for_term(intercept_term, dummy_df_st, mme_st)
            build_term_incidence_matrix(intercept_term, dummy_df_st.shape[0], mme_st)
            print(f"Intercept term data: {intercept_term.data}")
            print(f"Intercept term val: {intercept_term.val}")
            print(f"Intercept term X shape: {intercept_term.X.shape}, NLevels: {intercept_term.n_levels}")

            # Example: Process 'age' (covariate) term
            age_term = next(t for t in mme_st.model_terms if "age" in t.factors)
            get_data_for_term(age_term, dummy_df_st, mme_st)
            build_term_incidence_matrix(age_term, dummy_df_st.shape[0], mme_st)
            print(f"Age term X shape: {age_term.X.shape}, NLevels: {age_term.n_levels}")

            # Example: Process 'parity' (factor) term
            parity_term = next(t for t in mme_st.model_terms if "parity" in t.factors)
            get_data_for_term(parity_term, dummy_df_st, mme_st)
            build_term_incidence_matrix(parity_term, dummy_df_st.shape[0], mme_st)
            print(f"Parity term X shape: {parity_term.X.shape}, NLevels: {parity_term.n_levels}, Names: {parity_term.names}")
            print(f"Parity term X (dense):\
{parity_term.X.toarray()}")

            # Call get_mme_components for single trait
            get_mme_components(mme_st, dummy_df_st)
            print("get_mme_components called for single-trait.")
            print(f"MME LHS shape: {mme_st.mme_lhs.shape}")
            print(f"MME RHS shape: {mme_st.mme_rhs.shape}")
            print(f"Effect names: {get_effect_names(mme_st)}")


        except Exception as e:
            print(f"Error in ST term processing example: {e}")
