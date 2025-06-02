import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from .model_components import MME_py # Assuming MME_py and ParsedTerm are accessible
from .model_parser import ParsedTerm # If ParsedTerm is a separate class

def build_full_design_matrices_py(mme: MME_py):
    """
    Builds y_observed, X_effects_matrix (for fixed effects: intercept, covariates, factors),
    Z_marker_matrix (if applicable for direct MME inclusion of markers),
    obs_ids_list, inv_weights, and populates mme.effects_map.
    Updates mme object in place.
    """
    if mme.phenotype_dataframe is None:
        raise ValueError("Phenotype data not loaded into MME object for matrix building.")
    if not mme.model_terms: # Should be list of ParsedTerm objects from model_parser
        raise ValueError("Model terms not parsed. Call parser before building design matrices.")

    df = mme.phenotype_dataframe.copy() # Work on a copy
    pheno_info = mme.phenotype_info
    id_column = pheno_info["id_column"]

    # 1. Finalize obs_id and filter/sort DataFrame to this order
    df.dropna(subset=[id_column], inplace=True)
    df[id_column] = df[id_column].astype(str)
    mme.obs_id = df[id_column].unique().tolist() # Get unique IDs in their order of appearance
    # Ensure DataFrame is sorted/aligned according to these unique mme.obs_id
    df = df.set_index(id_column).loc[mme.obs_id].reset_index()

    n_obs_per_trait_orig = len(mme.obs_id)
    if n_obs_per_trait_orig == 0:
        raise ValueError("No observations remaining after handling missing IDs in phenotype data.")

    # 2. Construct y_observed (flattened for multi-trait: [y_trait1; y_trait2; ...])
    y_list_for_stacking = []
    for trait_name in mme.lhs_vec:
        if trait_name not in df.columns:
            raise ValueError(f"Trait column '{trait_name}' not found in phenotype DataFrame.")
        trait_data_series = pd.to_numeric(df[trait_name], errors='coerce')
        if trait_data_series.isnull().all():
            raise ValueError(f"Trait '{trait_name}' contains only missing values. Cannot proceed.")
        if trait_data_series.isnull().any():
            mean_val = np.nanmean(trait_data_series.values)
            print(f"Warning: Missing values in trait '{trait_name}'. Imputed with mean ({mean_val:.4f}).")
            trait_data_series.fillna(mean_val, inplace=True)
        y_list_for_stacking.append(trait_data_series.values)

    mme.y_observed = np.concatenate(y_list_for_stacking) if y_list_for_stacking else np.array([])
    n_total_observations_stacked = len(mme.y_observed)

    # 3. Construct X_effects_matrix for fixed effects & IID randoms modeled in X
    #    and Z_marker_matrix for markers modeled directly in MME.
    #    Populate mme.effects_map: {'effect_full_id': {'type': ..., 'start_col': ..., 'num_cols': ...}}

    mme.effects_map: Dict[str, Dict[str, Any]] = {}
    X_fixed_blocks_for_stacking: List[np.ndarray] = [] # Each element is trait's X block (n_obs_per_trait x n_fx_trait)

    current_col_offset_overall = 0 # Global column index for the combined MME system [X | Z_marker | Z_other_random]

    # --- Part A: Build X_effects_matrix (for fixed effects part of MME) ---
    # Iterate through traits to build block-diagonal X if effects are trait-specific
    for trait_idx, trait_name in enumerate(mme.lhs_vec):
        X_cols_for_this_trait_df_list: List[pd.DataFrame] = []

        # Get fixed terms for this trait, ensure consistent order (e.g., intercept first)
        trait_fixed_terms = sorted(
            [term for term in mme.model_terms if term.trait_name == trait_name and term.is_fixed and not term.is_random],
            key=lambda x: (x.term_type != 'intercept', x.base_name) # Sort intercept first
        )

        for term_cfg in trait_fixed_terms: # term_cfg is a ParsedTerm object
            full_term_id = term_cfg.full_id # e.g., "y1:age"
            base_name = term_cfg.base_name

            term_matrix_for_effect_in_trait = pd.DataFrame(index=df.index) # n_obs_per_trait_orig rows

            if term_cfg.term_type == "intercept":
                term_matrix_for_effect_in_trait[full_term_id] = 1.0
                term_cfg.n_levels = 1
                term_cfg.columns_in_X = [full_term_id]
            elif term_cfg.is_covariate:
                if base_name not in df.columns:
                    print(f"Warning: Covariate '{base_name}' for trait '{trait_name}' not in DataFrame. Skipping."); continue
                term_matrix_for_effect_in_trait[full_term_id] = df[base_name].astype(float)
                term_cfg.n_levels = 1
                term_cfg.columns_in_X = [full_term_id]
            elif term_cfg.term_type == "factor":
                if base_name not in df.columns:
                    print(f"Warning: Factor '{base_name}' for trait '{trait_name}' not in DataFrame. Skipping."); continue

                factor_series = df[base_name].astype('category')
                has_intercept_for_trait = any(tc.base_name == 'intercept' and tc.trait_name == trait_name for tc in trait_fixed_terms)
                dummies = pd.get_dummies(factor_series, prefix=full_term_id, drop_first=has_intercept_for_trait, dtype=float)
                if dummies.shape[1] == 0: term_cfg.n_levels = 0; continue
                term_matrix_for_effect_in_trait = pd.concat([term_matrix_for_effect_in_trait, dummies], axis=1)
                term_cfg.n_levels = dummies.shape[1]
                term_cfg.columns_in_X = dummies.columns.tolist()
            elif term_cfg.term_type == "interaction":
                # Basic interaction: product of columns (if all parts are covariates)
                # Or dummy interactions (if factors involved - more complex)
                # This is a placeholder for a complex step.
                print(f"Warning: Interaction term '{base_name}' processing is simplified/placeholder.")
                # Example: if term_cfg.factors = ["A", "B"] and both are covariates
                # term_matrix_for_effect_in_trait[full_term_id] = df[term_cfg.factors[0]] * df[term_cfg.factors[1]]
                # term_cfg.n_levels = 1
                # term_cfg.columns_in_X = [full_term_id]
                term_cfg.n_levels = 0 # Skip for now
            else:
                term_cfg.n_levels = 0; continue # Skip unknown fixed terms

            if term_cfg.n_levels > 0:
                X_cols_for_this_trait_df_list.append(term_matrix_for_effect_in_trait.loc[:, term_cfg.columns_in_X]) # Select only newly added columns
                # Update effects_map (global indexing)
                mme.effects_map[full_term_id] = {
                    'type': 'fixed_in_X', 'trait': trait_name, 'base_name': base_name,
                    'start_col': current_col_offset_overall + sum(b.shape[1] for b in X_fixed_blocks_per_trait_dfs[trait_name]), # Offset within this trait's block
                    'num_cols': term_cfg.n_levels, 'term_obj': term_cfg
                }

        if X_cols_for_this_trait_df_list:
            X_fixed_blocks_per_trait_dfs[trait_name] = pd.concat(X_cols_for_this_trait_df_list, axis=1).values
        else:
            X_fixed_blocks_per_trait_dfs[trait_name] = np.zeros((n_obs_per_trait_orig, 0))

    # Assemble the final X_effects_matrix using block_diag for traits
    final_X_fixed_trait_np_arrays = [X_fixed_blocks_per_trait_dfs[trait] for trait in mme.lhs_vec]

    if not any(block.shape[1] > 0 for block in final_X_fixed_trait_np_arrays):
        mme.X_effects_matrix = np.zeros((n_total_observations_stacked, 0))
    else:
        from scipy.linalg import block_diag
        mme.X_effects_matrix = block_diag(*final_X_fixed_trait_np_arrays)

    # Update global start_col for fixed effects in effects_map (done after all fixed blocks are sized)
    temp_offset = 0
    for trait_name in mme.lhs_vec: # Iterate in order of traits
        trait_fixed_terms_sorted = sorted( # Iterate in same order as X block construction
            [term for term in mme.model_terms if term.trait_name == trait_name and term.is_fixed and not term.is_random],
            key=lambda x: (x.term_type != 'intercept', x.base_name)
        )
        for term_cfg in trait_fixed_terms_sorted:
            if term_cfg.full_id in mme.effects_map and term_cfg.n_levels > 0:
                 mme.effects_map[term_cfg.full_id]['start_col'] = temp_offset
                 temp_offset += mme.effects_map[term_cfg.full_id]['num_cols']
    current_col_offset_overall = temp_offset


    # --- Part B: Build Z_marker_matrix for explicit marker effects (e.g. BayesC0) ---
    Z_marker_blocks_for_stacking: List[np.ndarray] = []
    for trait_idx, trait_name in enumerate(mme.lhs_vec):
        Z_cols_for_this_trait_list: List[np.ndarray] = []
        for gc in mme.genotype_components:
            if gc.method in ["BayesC0", "RR-BLUP"]: # If markers solved in MME
                if gc.genotype_matrix is None: continue

                # Align gc.genotype_matrix rows with mme.obs_id for this trait's observations
                # This is complex if gc.obs_ids != mme.obs_id or if MT structure is complex.
                # Assume for now gc.genotype_matrix is (n_obs_per_trait_orig x n_markers) and aligned.
                if list(gc.obs_ids) != mme.obs_id: # Basic check, needs robust alignment
                    print(f"Warning: Marker component {gc.name} obs_ids differ from main MME obs_ids. Alignment required for Z_marker_matrix.")
                    # aligned_marker_matrix = align_matrix_to_ids(gc.genotype_matrix, gc.obs_ids, mme.obs_id) # Conceptual
                    aligned_marker_matrix = None # Skip if not aligned for now
                else:
                    aligned_marker_matrix = gc.genotype_matrix

                if aligned_marker_matrix is not None:
                    Z_cols_for_this_trait_list.append(aligned_marker_matrix)
                    # Update effects_map for these marker effects
                    map_key_marker_block = f"markers_{gc.name}_{trait_name}"
                    mme.effects_map[map_key_marker_block] = {
                        'type': 'marker_block', 'trait': trait_name, 'geno_comp_name': gc.name,
                        'start_col': current_col_offset_overall, 'num_cols': gc.n_markers,
                        'term_obj': gc # Reference to GenotypesComponent
                    }
                    # Individual marker mapping (if solver needs it, or for output)
                    # for marker_idx, marker_id_str in enumerate(gc.marker_ids):
                    #     mme.effects_map[f"{trait_name}:{gc.name}:{marker_id_str}"] = {
                    #         'type': 'marker', 'start_col': current_col_offset_overall + marker_idx, 'num_cols': 1}
                    current_col_offset_overall += gc.n_markers

        if Z_cols_for_this_trait_list:
            Z_marker_blocks_for_stacking.append(np.hstack(Z_cols_for_this_trait_list))
        else:
            Z_marker_blocks_for_stacking.append(np.zeros((n_obs_per_trait_orig, 0)))

    if not Z_marker_blocks_for_stacking or all(block.shape[1] == 0 for block in Z_marker_blocks_for_stacking):
        mme.Z_marker_matrix = np.zeros((n_total_observations_stacked, 0))
    else:
        from scipy.linalg import block_diag
        mme.Z_marker_matrix = block_diag(*Z_marker_blocks_for_stacking)

    # --- Part C: Map other random effects (polygenic, etc., not in X or Z_marker) ---
    # These are handled by adding K_inv to LHS, their solutions are appended to main solution_vector.
    for rec in mme.random_effect_components:
        # Check if this REC is NOT an IID effect already built into X_effects_matrix
        is_in_X = False
        if rec.term_array: # Check if its terms were mapped into X_effects_matrix
            term_id_for_rec = rec.term_array[0] # Use first term as representative
            if term_id_for_rec in mme.effects_map and mme.effects_map[term_id_for_rec]['type'] == 'fixed_in_X':
                is_in_X = True

        if not is_in_X and rec.random_type in ["A", "H", "V"]:
            num_re_levels = 0
            if rec.Vinv_obj is not None: num_re_levels = rec.Vinv_obj.shape[0]
            elif rec.Vinv_names: num_re_levels = len(rec.Vinv_names)

            if num_re_levels > 0 :
                map_key = rec.term_array[0] if rec.term_array else f"random_effect_{rec.name or 'unknown'}"
                mme.effects_map[map_key] = {
                    'type': f'random_{rec.random_type}',
                    'trait': 'all_traits_combined', # This needs refinement for MT if effect is trait-specific
                    'start_col': current_col_offset_overall,
                    'num_cols': num_re_levels,
                    'base_name': rec.name if hasattr(rec,'name') and rec.name else (rec.term_array[0].split(":")[-1] if rec.term_array else "unknown_re"),
                    'term_obj': rec
                }
                current_col_offset_overall += num_re_levels

    mme.inv_weights = np.ones(n_total_observations_stacked)
    print(f"MME Builder: y({mme.y_observed.shape if mme.y_observed is not None else 'N/A'}), "
          f"X_fx({mme.X_effects_matrix.shape if mme.X_effects_matrix is not None else 'N/A'}), "
          f"Z_mk({mme.Z_marker_matrix.shape if mme.Z_marker_matrix is not None else 'N/A'}). "
          f"Total effects in map: {len(mme.effects_map)}, total columns for MME system: {current_col_offset_overall}")

    mme.num_total_effects_in_mme_system = current_col_offset_overall
    return mme.num_total_effects_in_mme_system

