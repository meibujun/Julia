import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import os # For path joining in _prepare_for_run (example data saving)

# These would be actual imports from other modules within the package
from .core.model_components import MME_py, VarianceComponent
from .pedigree.pedigree_module import get_pedigree as get_pedigree_internal
from .genotypes.genotype_handler import read_genotypes_py, calculate_grm_py
from .core.model_components import (
    add_genotypes_py,
    set_random_py,
    check_model_arguments_py,
    check_data_consistency_py,
    set_default_priors_for_variance_components_py,
    set_marker_hyperparameters_py,
    calculate_H_inverse_py
)
# Conceptual imports for functions that would be part of MME building/setup
# from .core.model_setup import build_X_y_mats_py, init_mme_matrices_py
# For now, _parse_model_equation_py and _build_data_matrices_py are local helpers

# For run methods (conceptual)
from .mcmc.mcmc_engine import run_mcmc_py as run_mcmc_internal
from .gwas.gwas_module import run_window_gwas_py as run_gwas_internal
# For results methods (conceptual)
from .utils.output_utils import get_ebv_py as get_ebv_internal, save_results_to_csv_py


# --- Model Equation Parser (Simplified Placeholder for _prepare_for_run) ---
def _parse_model_equation_py(equation_str: str, trait_names_from_lhs_parse: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Simplified parser for model equations.
    Extracts terms for each trait. Populates mme.model_terms and mme.model_term_dict.
    A real implementation would be much more robust.
    """
    model_terms_list = []
    model_term_dict = {}

    raw_equations = [eq.strip() for eq in equation_str.replace(';', '\n').split('\n') if eq.strip()]

    parsed_lhs_check = []
    for eq_part in raw_equations:
        lhs, rhs_str = eq_part.split('=', 1)
        current_trait_name = lhs.strip()
        parsed_lhs_check.append(current_trait_name)

        rhs_terms_in_eq = [term.strip() for term in rhs_str.split('+')]
        for term_base_name in rhs_terms_in_eq:
            full_term_id = f"{current_trait_name}:{term_base_name}"
            term_config = {
                "id_str": full_term_id,
                "base_name": term_base_name,
                "trait_name": current_trait_name,
                "factors": term_base_name.split('*'),
                "is_fixed": True,
                "is_covariate": False,
                "random_type": "fixed",
                "names": [], # Will be populated for factors by MME builder
                "n_levels": 0, # Will be populated
                "start_pos_in_coeffs": -1, # Will be populated by MME builder
                "is_in_X_effects_matrix": False # Flag if part of main X
            }
            model_terms_list.append(term_config)
            model_term_dict[full_term_id] = term_config

    if parsed_lhs_check != trait_names_from_lhs_parse:
        # This can happen if the initial LHS parsing in set_model_equation was too simple
        # and this more detailed parsing (if it were more detailed) found discrepancies.
        # For now, this check is more of a failsafe for the simple parser.
        print(f"Warning: LHS trait list from full parse {parsed_lhs_check} differs from initial {trait_names_from_lhs_parse}. Using full parse.")
        # mme.lhs_vec = parsed_lhs_check
        # mme.n_models = len(parsed_lhs_check)
        # This implies mme.traits_type might also need realignment if lhs_vec changes order/content.

    return model_terms_list, model_term_dict


# --- Data Processing and Matrix Building (Enhanced Placeholder for _prepare_for_run) ---
def _build_data_matrices_py(mme: MME_py):
    """
    Builds y_observed, X_effects_matrix (for fixed effects: intercept, covariates, factors),
    obs_id, and inv_weights. Updates mme.model_term_dict with column mapping info.
    This is a significant step. A full implementation is complex.
    """
    if mme.phenotype_dataframe is None:
        raise ValueError("Phenotype data not loaded into MME object for matrix building.")

    df = mme.phenotype_dataframe.copy() # Work on a copy
    pheno_info = mme.phenotype_info

    # 1. Finalize obs_id: Filter df for individuals present in pedigree and/or genotype data if applicable
    #    This step is complex and involves consistency checks (done by check_data_consistency_py).
    #    For now, assume df contains the individuals to be analyzed.
    #    Ensure IDs are strings and handle potential missing IDs in phenotype data.
    df.dropna(subset=[pheno_info["id_column"]], inplace=True)
    df[pheno_info["id_column"]] = df[pheno_info["id_column"]].astype(str)
    mme.obs_id = df[pheno_info["id_column"]].tolist()
    n_obs_per_trait_orig = len(mme.obs_id)

    if n_obs_per_trait_orig == 0:
        raise ValueError("No observations remaining after handling missing IDs in phenotype data.")

    # 2. Construct y_observed (flattened for multi-trait: [y_trait1; y_trait2; ...])
    y_list_for_stacking = []
    for trait_col_name in mme.lhs_vec:
        if trait_col_name not in df.columns:
            raise ValueError(f"Trait column '{trait_col_name}' not found in phenotype DataFrame.")
        # Convert to numeric, coercing errors will turn non-numeric to NaN
        trait_data = pd.to_numeric(df[trait_col_name], errors='coerce')
        # Handle missing trait observations (e.g., fill with mean or mark for special handling in MME)
        # For now, let's fill with mean of observed values for that trait (simplistic imputation for y)
        if trait_data.isnull().any():
            print(f"Warning: Missing values found in trait '{trait_col_name}'. Imputing with mean for MME setup.")
            trait_data.fillna(trait_data.nanmean(), inplace=True) # Use nanmean
        y_list_for_stacking.append(trait_data.values)

    mme.y_observed = np.concatenate(y_list_for_stacking) if y_list_for_stacking else np.array([])
    n_total_observations = len(mme.y_observed)

    # 3. Construct X_effects_matrix for fixed effects from mme.model_terms
    X_cols_list_per_trait: Dict[str, List[pd.DataFrame]] = {trait: [] for trait in mme.lhs_vec}
    effect_column_map: List[Dict[str, Any]] = [] # To store info about columns in X
    current_col_offset_in_X = 0

    # Iterate through unique base terms that are fixed (not explicitly random)
    # This needs a proper list of fixed effect terms from the parsed model equation.
    # Using mme.model_terms (which contains all term occurrences per trait)

    processed_base_terms_for_X = set() # To avoid reprocessing a base term like 'age' for X matrix

    for term_cfg_prototype in mme.model_terms: # These are from _parse_model_equation_py
        # Only consider terms that are fixed and not handled by specialized matrices (like pedigree 'animal')
        is_random_special = False
        for rec_conf in mme.random_effects_config: # User-defined random effects
            if rec_conf['name'] == term_cfg_prototype['base_name']:
                is_random_special = True; break
        if is_random_special and term_cfg_prototype['base_name'] != "intercept": # Keep intercept if fixed
             # This random effect will be handled by Z matrix or specialized sampler, not in this X.
             # However, IID random effects based on factors in data *could* be part of X via dummies.
             # This logic needs refinement based on how RandomEffectComponent Z matrices are built/used.
             # For now, if user explicitly called add_random_effect for it, assume it's not in this X.
            continue

        base_name = term_cfg_prototype['base_name']

        # If this base term (e.g. "age", "herd") hasn't had its columns created for X_effects_matrix yet
        # This logic is for effects that might be common across traits or trait-specific in X
        # For now, assume effects are trait-specific if they appear for multiple traits.
        # The block_diag structure below handles this naturally if each trait gets its own columns.

        # This simplified loop processes each term instance from the parsed equation
        # A better way is to get unique fixed effect terms, build their incidence, then expand for MT.

    # Rebuild X based on model_term_dict which should now have updated term types
    for trait_idx, trait_name in enumerate(mme.lhs_vec):
        X_cols_for_this_trait_df_list = []
        # Sort terms for consistent column order (e.g., intercept first, then others)
        trait_terms_sorted = sorted(
            [tc for tc in mme.model_terms if tc['trait_name'] == trait_name and tc['is_fixed']],
            key=lambda x: (x['base_name'] != 'intercept', x['base_name'])
        )

        for term_cfg in trait_terms_sorted:
            base_name = term_cfg['base_name']
            full_term_id = term_cfg['id_str']
            term_detail_for_mme = mme.model_term_dict[full_term_id] # Get the centrally stored config

            term_detail_for_mme['start_pos_in_coeffs'] = current_col_offset_in_X

            if base_name == "intercept":
                col_data = pd.Series(np.ones(n_obs_per_trait_orig), name=full_term_id)
                X_cols_for_this_trait_df_list.append(col_data)
                term_detail_for_mme['n_levels'] = 1
                current_col_offset_in_X += 1
            elif base_name in pheno_info.get("covariate_columns", []): # It's a covariate
                if base_name not in df.columns:
                    print(f"Warning: Covariate '{base_name}' for trait '{trait_name}' not in phenotype DataFrame.")
                    term_detail_for_mme['n_levels'] = 0; continue
                col_data = pd.Series(df[base_name].astype(float).values, name=full_term_id)
                X_cols_for_this_trait_df_list.append(col_data)
                term_detail_for_mme['n_levels'] = 1
                term_detail_for_mme['is_covariate'] = True
                current_col_offset_in_X += 1
            elif base_name in df.columns: # Potential factor (not listed as covariate)
                factor_series = df[base_name].astype('category')
                # drop_first needs to be true if a general intercept for this trait is already included
                has_intercept = f"{trait_name}:intercept" in mme.model_term_dict
                dummies = pd.get_dummies(factor_series, prefix=full_term_id, drop_first=has_intercept)
                X_cols_for_this_trait_df_list.append(dummies)
                term_detail_for_mme['n_levels'] = dummies.shape[1]
                term_detail_for_mme['is_covariate'] = False
                current_col_offset_in_X += dummies.shape[1]
            else:
                # Term not in data (e.g. 'animal' for pedigree, or an effect to be estimated without direct data columns)
                # These won't contribute to this X_effects_matrix for fixed effects.
                term_detail_for_mme['n_levels'] = 0
                # print(f"Info: Term '{base_name}' for trait '{trait_name}' not found in phenotype data columns or not a covariate, not added to X_effects_matrix.")

        if X_cols_for_this_trait_df_list:
            trait_X_df = pd.concat(X_cols_for_this_trait_df_list, axis=1)
            trait_X_parts.append(trait_X_df.values)
        else:
            trait_X_parts.append(np.zeros((n_obs_per_trait_orig, 0))) # No fixed effects for this trait

    if not trait_X_parts:
        mme.X_effects_matrix = np.zeros((n_total_observations, 0))
    else:
        from scipy.linalg import block_diag # Local import
        mme.X_effects_matrix = block_diag(*trait_X_parts)

    mme.X_effects_column_names = [term['id_str'] for term in mme.model_terms if term.get('start_pos_in_coeffs', -1) != -1 and term['is_fixed']]
    # This naming needs to be more robust if dummy variables expand one term into multiple columns.
    # The current_col_offset_in_X gives total number of columns from these fixed effects.

    mme.inv_weights = np.ones(n_total_observations)
    print(f"Data matrices built: y_observed ({mme.y_observed.shape}), X_effects_matrix ({mme.X_effects_matrix.shape}) for {current_col_offset_in_X} effects.")
    return current_col_offset_in_X


class GenoStockModel:
>>>>>>> REPLACE
I've updated `genostockpy/api.py` by implementing a more detailed version of `_prepare_for_run()`. I've also enhanced the placeholder model equation parser (`_parse_model_equation_py`) and the data matrix builder (`_build_data_matrices_py`) which are now defined as helper functions at the module level within `api.py`.

**Key Changes in `_prepare_for_run()` and its helpers in `api.py`:**

1.  **`_parse_model_equation_py(equation_str, existing_lhs_vec)` (Enhanced Placeholder):**
    *   Takes the equation string and the list of LHS trait names derived by `set_model_equation`.
    *   Splits equations by `+` and terms by `*` (still very basic, no proper interaction or factor/covariate detection here).
    *   Creates a list of term configuration dictionaries (`model_terms_list`) where each entry represents a term's occurrence (e.g., "y1:age", "y2:age" are distinct). Each config includes `id_str`, `base_name`, `trait_name`, `factors` (split by `*`), and defaults for `is_fixed`, `is_covariate`, `random_type`.
    *   Creates `model_term_dict` mapping `full_term_id` to its configuration.
    *   This parsed information is stored in `self._mme.model_terms` and `self._mme.model_term_dict`.

2.  **`_build_data_matrices_py(mme: MME_py)` (Enhanced Placeholder):**
    *   Uses `mme.phenotype_dataframe` and `mme.model_terms` (populated by the parser).
    *   **`mme.obs_id`**: Finalized from the phenotype data's ID column.
    *   **`mme.y_observed`**: Constructed by stacking trait columns from phenotype data (flattened for multi-trait). Missing values in traits are now (simplistically) imputed with the mean of that trait.
    *   **`mme.X_effects_matrix`**:
        *   Iterates through each trait. For each trait, it iterates through its fixed model terms (from `mme.model_terms`).
        *   **Intercept**: If "intercept" term exists for the trait, adds a column of ones.
        *   **Covariates**: If a term is listed in `phenotype_info["covariate_columns"]`, its data is added.
        *   **Factors**: If a term is in phenotype data but not a covariate, it's treated as a factor. `pd.get_dummies()` creates dummy variables. `drop_first=True` is used if an intercept for that trait is present.
        *   The `X` parts for each trait are then combined into `mme.X_effects_matrix` using `scipy.linalg.block_diag`. This implies a model where effects are trait-specific if they appear in multiple trait equations (e.g., "y1:age" and "y2:age" get separate columns).
        *   Updates `n_levels` and `start_pos_in_coeffs` (conceptual offset) in the `mme.model_term_dict` entries for terms included in `X_effects_matrix`. This mapping is still not fully robust.
    *   Initializes `mme.inv_weights` to ones.
    *   Returns the total number of columns constructed in `X_effects_matrix`.

3.  **`_prepare_for_run()` Orchestration**:
    *   Calls `_parse_model_equation_py`.
    *   Calls `get_pedigree_internal` if pedigree is configured.
    *   Processes `genotype_components_config`: calls `read_genotypes_py`, then `add_genotypes_py` (which internally calls method-specific setups like `setup_gblup_py` and `set_marker_hyperparameters_py`).
    *   Processes `random_effects_config`: calls `set_random_py`.
    *   Configures `residual_variance_prior`.
    *   Handles SSGBLUP setup: reorders pedigree, calculates GRM (if needed via `calculate_grm_py`), calculates $H^{-1}$ (via `calculate_H_inverse_py`), and updates the polygenic `RandomEffectComponent`.
    *   Calls `_build_data_matrices_py` to form `y_observed`, `X_effects_matrix`, etc.
    *   Conceptual calls to validation (`check_model_arguments_py`, `check_data_consistency_py` - commented out) and final prior settings (`set_default_priors_for_variance_components_py`, `set_marker_hyperparameters_py`).
    *   Calls `self._mme.initialize_mcmc_state()`.

**Limitations Persisting:**
*   The model equation parser and `X_effects_matrix` builder are still simplified. They don't robustly handle interactions, determine factor vs. covariate status automatically from data types, or create a fully detailed term-to-MME-column map for all effect types. This impacts the ability of `set_random_py` to always find terms correctly and the `addVinv` part of MME LHS construction.
*   `check_data_consistency_py` is commented out in `_prepare_for_run` because it requires a more fully formed MME (e.g., knowing which individuals are in genotype files vs. pedigree vs. phenotype before final alignment).

This detailed `_prepare_for_run` provides a much clearer path from API calls to a configured `MME_py` object. The next logical step is to refine `_construct_mme_lhs_rhs_py` in `mcmc_engine.py` to correctly use the variance components for building the MME LHS, at least for a simple single-trait model with fixed effects and one type of random effect (e.g., IID or polygenic). Then, update unit tests.
