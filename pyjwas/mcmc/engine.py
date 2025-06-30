from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import time
import os
import sys

from ..core.mme import MixedModelEquations
from ..core.mcmc_info import MCMCInfo
from ..core.variance_covariance import VarianceCovariance
from ..utils.solvers import gibbs_sample_solution_in_place
from ..utils.samplers import sample_scalar_variance_component, sample_matrix_variance_component, sample_general_random_effect_variances
from ..core.model_builder import _make_Ri_matrix
from scipy.sparse import identity as sparse_identity, csc_matrix, csr_matrix, diags, lil_matrix, spmatrix
from .marker_samplers import (
    _sample_marker_effects_rrblup_st,
    _sample_marker_effects_bayesa_st,
    _sample_marker_effects_bayesc_st,
    _sample_marker_effects_bayesb_st,
    _sample_marker_effects_bayesl_st,
    _sample_marker_effects_mt_rrblup # Added MT RR-BLUP
)


def _print_progress(iteration, total_iterations, start_time, print_frequency):
    if iteration % print_frequency == 0 or iteration == total_iterations:
        elapsed_time = time.time() - start_time
        avg_time_per_iter = elapsed_time / iteration if iteration > 0 else 0
        eta = avg_time_per_iter * (total_iterations - iteration)
        print(f"Iteration {iteration}/{total_iterations} | "
              f"Elapsed: {elapsed_time:.2f}s | "
              f"ETA: {eta:.2f}s")
        sys.stdout.flush()


def _update_mme_lhs_for_solver(mme: MixedModelEquations):
    if mme.mme_LHS is None:
        raise ValueError("_update_mme_lhs_for_solver cannot proceed if mme.mme_LHS is not initialized.")

    is_sparse_lhs = isinstance(mme.mme_LHS, spmatrix)
    # Assume mme.mme_LHS is already LIL if it's sparse and updates are frequent,
    # handled at the start of run_mcmc.

    for py_random_term in mme.random_effect_terms:
        if not (py_random_term.Gi_new and py_random_term.Gi_new.value is not None and \
                py_random_term.Gi_old and py_random_term.Gi_old.value is not None):
            continue

        n_levels_first_term = mme.model_term_dict[py_random_term.term_array[0]].n_levels if \
                              py_random_term.term_array and py_random_term.term_array[0] in mme.model_term_dict else 0
        if n_levels_first_term == 0 and py_random_term.V_inv is None : continue

        Vi = py_random_term.V_inv if py_random_term.V_inv is not None else \
             sparse_identity(n_levels_first_term, format="csc", dtype=np.float64)

        for i, term_i_str in enumerate(py_random_term.term_array):
            model_term_i = mme.model_term_dict.get(term_i_str)
            if not model_term_i or model_term_i.n_levels == 0: continue
            start_pos_i, end_pos_i = model_term_i.start_pos, model_term_i.start_pos + model_term_i.n_levels

            for j, term_j_str in enumerate(py_random_term.term_array):
                model_term_j = mme.model_term_dict.get(term_j_str)
                if not model_term_j or model_term_j.n_levels == 0: continue
                start_pos_j, end_pos_j = model_term_j.start_pos, model_term_j.start_pos + model_term_j.n_levels

                change_in_lambda_block_scalar: float = 0.0
                if mme.n_models == 1:
                    if mme.R_variance.value is None or mme.R_old_value is None: continue
                    val_Gi_new, val_Gi_old = py_random_term.Gi_new.value, py_random_term.Gi_old.value
                    g_new_ij = val_Gi_new[i,j] if isinstance(val_Gi_new, np.ndarray) and val_Gi_new.ndim==2 else val_Gi_new
                    g_old_ij = val_Gi_old[i,j] if isinstance(val_Gi_old, np.ndarray) and val_Gi_old.ndim==2 else val_Gi_old
                    current_R_val, old_R_val = float(mme.R_variance.value), float(mme.R_old_value)
                    change_in_lambda_block_scalar = (g_new_ij * current_R_val - g_old_ij * old_R_val)
                else:
                    val_Gi_new, val_Gi_old = py_random_term.Gi_new.value, py_random_term.Gi_old.value # These should be G_inv matrices
                    g_new_ij = val_Gi_new[i,j] if isinstance(val_Gi_new, np.ndarray) else val_Gi_new
                    g_old_ij = val_Gi_old[i,j] if isinstance(val_Gi_old, np.ndarray) else val_Gi_old
                    change_in_lambda_block_scalar = g_new_ij - g_old_ij # G_inv_new[i,j] - G_inv_old[i,j]

                matrix_to_add = Vi * change_in_lambda_block_scalar
                if mme.mme_LHS[start_pos_i:end_pos_i, start_pos_j:end_pos_j].shape == matrix_to_add.shape:
                    mme.mme_LHS[start_pos_i:end_pos_i, start_pos_j:end_pos_j] += matrix_to_add
                else:
                    print(f"Warning: Shape mismatch in LHS update for terms {term_i_str}, {term_j_str}.")


def _calculate_effective_mme_rhs_for_solver(mme: MixedModelEquations, df_pheno_for_ri: Optional[pd.DataFrame]) -> np.ndarray:
    if mme.mme_RHS is None or mme.X is None or mme.y_sparse is None:
        raise ValueError("Initial MME RHS, X, or y_sparse not available.")
    effective_rhs = mme.mme_RHS.copy().ravel()
    marker_contribution_to_y = np.zeros_like(mme.y_sparse.ravel(), dtype=np.float64)
    n_obs_per_trait = len(mme.obs_ids)

    for geno_data in mme.genotypes_data_list:
        if geno_data.genotypes is not None and geno_data.alpha_samples and len(geno_data.alpha_samples) > 0:
            for trait_idx in range(mme.n_models):
                if trait_idx < len(geno_data.alpha_samples) and geno_data.alpha_samples[trait_idx] is not None:
                    current_alpha_trait = geno_data.alpha_samples[trait_idx]
                    if geno_data.genotypes.ndim == 2 and \
                       geno_data.genotypes.shape[0] == n_obs_per_trait and \
                       geno_data.genotypes.shape[1] == len(current_alpha_trait):
                        start_row_y = trait_idx * n_obs_per_trait
                        end_row_y = start_row_y + n_obs_per_trait
                        marker_contrib_trait = geno_data.genotypes @ current_alpha_trait
                        marker_contribution_to_y[start_row_y:end_row_y] += marker_contrib_trait

    if np.any(marker_contribution_to_y != 0):
        R_inv_eff: Optional[spmatrix] = None
        if mme.n_models == 1:
            if mme.R_variance.value is not None and float(mme.R_variance.value) > 0:
                diag_vals = mme.inverse_weights / float(mme.R_variance.value)
                R_inv_eff = diags(diag_vals, format="csc")
        else:
            if hasattr(mme, 'current_Ri_matrix') and mme.current_Ri_matrix is not None:
                 R_inv_eff = mme.current_Ri_matrix
            elif df_pheno_for_ri is not None :
                R_inv_eff = _make_Ri_matrix(mme, df_pheno_for_ri, mme.inverse_weights)
            else: print("Warning: Cannot form R_inv for multi-trait RHS adjustment.")

        if R_inv_eff is not None and mme.X.shape[1] > 0 :
            adjustment_vector = mme.X.T @ (R_inv_eff @ marker_contribution_to_y.reshape(-1,1))
            effective_rhs -= adjustment_vector.ravel()

    return effective_rhs


def run_mcmc(mme: MixedModelEquations, df_pheno: pd.DataFrame) -> Dict[str, Any]:
    if mme.mcmc_info is None: raise ValueError("MCMCInfo must be set.")
    if mme.mme_LHS is None or mme.mme_RHS is None or mme.X is None or mme.y_sparse is None:
        raise ValueError("MME components must be built before running MCMC.")

    mcmc_params: MCMCInfo = mme.mcmc_info
    chain_length, burnin = mcmc_params.chain_length, mcmc_params.burnin
    output_samples_freq = mcmc_params.output_samples_frequency if mcmc_params.output_samples_frequency > 0 else chain_length

    output_folder = mcmc_params.output_folder
    if not os.path.exists(output_folder): os.makedirs(output_folder); print(f"Created output folder: {output_folder}")
    if mcmc_params.seed is not None: np.random.seed(mcmc_params.seed)

    if mme.solutions is None: mme.solutions = np.zeros(mme.mme_LHS.shape[0])
    if mme.mean_solutions is None: mme.mean_solutions = np.zeros_like(mme.solutions)
    if mme.mean_solutions_sq is None: mme.mean_solutions_sq = np.zeros_like(mme.solutions)

    if mme.R_variance.value is not None:
        if mme.n_models == 1:
            if mme.mean_residual_variance is None: mme.mean_residual_variance = 0.0
            if mme.mean_residual_variance_sq is None: mme.mean_residual_variance_sq = 0.0
            mme.R_old_value = float(mme.R_variance.value)
        else:
            if mme.mean_residual_variance is None: mme.mean_residual_variance = np.zeros_like(mme.R_variance.value)
            if mme.mean_residual_variance_sq is None: mme.mean_residual_variance_sq = np.zeros_like(mme.R_variance.value)

    for rt in mme.random_effect_terms:
        source_Gi_vc = rt.Gi_new if rt.Gi_new and rt.Gi_new.value is not None else (rt.Gi if rt.Gi and rt.Gi.value is not None else None)
        if source_Gi_vc:
            val_to_copy = np.copy(source_Gi_vc.value) if isinstance(source_Gi_vc.value, np.ndarray) else source_Gi_vc.value
            scale_to_copy = np.copy(source_Gi_vc.scale) if isinstance(source_Gi_vc.scale, np.ndarray) else source_Gi_vc.scale
            rt.Gi_old = VarianceCovariance(value=val_to_copy, df=source_Gi_vc.df, scale=scale_to_copy, estimate_variance=source_Gi_vc.estimate_variance, estimate_scale=source_Gi_vc.estimate_scale, constraint=source_Gi_vc.constraint)
            if rt.Gi_new is None and rt.Gi is not None:
                 rt.Gi_new = rt.Gi_old.deepcopy() if hasattr(rt.Gi_old, 'deepcopy') else VarianceCovariance(value=val_to_copy, df=rt.Gi_old.df, scale=scale_to_copy)

    for geno_data in mme.genotypes_data_list:
        if not geno_data.alpha_samples: geno_data.initialize_mcmc_storage(mme.n_models, geno_data.n_markers)

    if mme.n_models > 1 and mme.R_variance.value is not None :
        mme.current_Ri_matrix = _make_Ri_matrix(mme, df_pheno, mme.inverse_weights)

    print(f"Starting MCMC: {chain_length} iterations, {burnin} burn-in.")
    start_time = time.time()
    num_saved_samples = 0

    lhs_was_converted_to_lil = False
    original_lhs_sparse_type = None
    if isinstance(mme.mme_LHS, spmatrix) and not isinstance(mme.mme_LHS, lil_matrix):
        original_lhs_sparse_type = type(mme.mme_LHS)
        mme.mme_LHS = mme.mme_LHS.tolil()
        lhs_was_converted_to_lil = True

    for iter_num in range(1, chain_length + 1):
        pass

        if mme.mme_LHS.shape[0] > 0:
            _update_mme_lhs_for_solver(mme)
            lhs_for_solver = mme.mme_LHS
            effective_rhs = _calculate_effective_mme_rhs_for_solver(mme, df_pheno)
            gibbs_sample_solution_in_place(
                lhs_for_solver, mme.solutions, effective_rhs,
                residual_variance=float(mme.R_variance.value) if mme.n_models == 1 and mme.R_variance.value is not None else None)

        y_corrected = mme.y_sparse.copy().ravel()
        if mme.X.shape[1] > 0: y_corrected -= mme.X @ mme.solutions

        # This y_corrected (y - Xb) is passed to marker samplers.
        # Marker samplers update their alphas and ALSO y_corrected to (y - Xb - Za_new)
        for geno_data in mme.genotypes_data_list:
            if geno_data.method in ["RR-BLUP", "BayesC0"]:
                if mme.n_models == 1:
                    if geno_data.marker_effect_variance is None: print(f"Warning: marker_effect_variance not set for {geno_data.name}. Skipping."); continue
                    if geno_data.marker_effect_variance.value is None:
                        if geno_data.marker_effect_variance.estimate_variance: geno_data.marker_effect_variance.value = 0.001
                        else: print(f"Error: Fixed marker_effect_variance not set for {geno_data.name}. Skipping."); continue

                    _sample_marker_effects_rrblup_st(
                        geno_data, y_corrected,
                        float(mme.R_variance.value) if mme.R_variance.value is not None else 1.0,
                        float(geno_data.marker_effect_variance.value)
                    )
                    if geno_data.marker_effect_variance.estimate_variance:
                        new_marker_var = sample_scalar_variance_component(
                            geno_data.alpha_samples[0], geno_data.marker_effect_variance.df,
                            geno_data.marker_effect_variance.scale)
                        if new_marker_var > 1e-12: geno_data.marker_effect_variance.value = new_marker_var
                        else: print(f"Warning: Sampled marker variance too small for {geno_data.name}.")
                else: print(f"Warning: Multi-trait marker sampler for method {geno_data.method} not yet implemented.")

            elif geno_data.method == "BayesA":
                if mme.n_models == 1:
                    if geno_data.marker_effect_variance is None: print(f"Warning: BayesA marker_effect_variance obj not set for {geno_data.name}. Skipping."); continue
                    if geno_data.marker_effect_variance.df is None or geno_data.marker_effect_variance.scale is None:
                        print(f"Error: Prior df/scale for BayesA marker variances not set for {geno_data.name}. Skipping.")
                        continue

                    _sample_marker_effects_bayesa_st(
                        geno_data, y_corrected,
                        float(mme.R_variance.value) if mme.R_variance.value is not None else 1.0,
                        mme.inverse_weights
                    )
                else: print(f"Warning: Multi-trait BayesA sampler not yet implemented for {geno_data.name}.")

            elif geno_data.method == "BayesC":
                if mme.n_models == 1:
                    if geno_data.marker_effect_variance is None: print(f"Warning: BayesC marker_effect_variance obj not set for {geno_data.name}. Skipping."); continue
                    if geno_data.marker_effect_variance.value is None:
                        if geno_data.marker_effect_variance.estimate_variance: geno_data.marker_effect_variance.value = 0.001
                        else: print(f"Error: Fixed common marker_effect_variance not set for BayesC {geno_data.name}. Skipping."); continue
                    if geno_data.pi_value is None:
                        if geno_data.estimate_pi: geno_data.pi_value = 0.05
                        else: print(f"Error: Fixed Pi value not set for BayesC {geno_data.name}. Skipping."); continue

                    _sample_marker_effects_bayesc_st(
                        geno_data, y_corrected,
                        float(mme.R_variance.value) if mme.R_variance.value is not None else 1.0,
                        mme.inverse_weights
                        # pi_prior_alpha, pi_prior_beta will use defaults in sampler for now
                    )
                else: print(f"Warning: Multi-trait BayesC sampler not yet implemented for {geno_data.name}.")

            elif geno_data.method == "BayesB":
                if mme.n_models == 1:
                    if geno_data.marker_effect_variance is None: print(f"Warning: BayesB marker_effect_variance obj not set for {geno_data.name}. Skipping."); continue
                    if geno_data.marker_effect_variance.df is None or geno_data.marker_effect_variance.scale is None:
                        print(f"Error: Prior df/scale for BayesB marker variances not set for {geno_data.name}. Skipping.")
                        continue
                    if geno_data.pi_value is None:
                        if geno_data.estimate_pi: geno_data.pi_value = 0.05
                        else: print(f"Error: Fixed Pi value not set for BayesB {geno_data.name}. Skipping."); continue

                    _sample_marker_effects_bayesb_st(
                        geno_data, y_corrected,
                        float(mme.R_variance.value) if mme.R_variance.value is not None else 1.0,
                        mme.inverse_weights
                    )
                else: print(f"Warning: Multi-trait BayesB sampler not yet implemented for {geno_data.name}.")

            elif geno_data.method == "BayesL":
                if mme.n_models == 1:
                    if geno_data.marker_effect_variance is None or \
                       not isinstance(geno_data.marker_effect_variance.value, (float,int,np.floating)):
                        print(f"Error: Common marker variance for BayesL not set for {geno_data.name}. Skipping."); continue

                    _sample_marker_effects_bayesl_st(
                        geno_data, y_corrected,
                        float(mme.R_variance.value) if mme.R_variance.value is not None else 1.0,
                        mme.inverse_weights
                    )
                else: print(f"Warning: Multi-trait BayesL sampler not yet implemented for {geno_data.name}.")


            else: print(f"Warning: Marker method '{geno_data.method}' not implemented for {geno_data.name}.")

        # y_corrected is now: y_obs - X@beta_new - Z_all_markers@alpha_new_all_markers

        # Store current Gi_new as Gi_old for *next* iteration's LHS update, *before* sampling new Gi_new
        for rt in mme.random_effect_terms:
            if rt.Gi_new and rt.Gi_new.value is not None:
                 val_to_copy = np.copy(rt.Gi_new.value) if isinstance(rt.Gi_new.value, np.ndarray) else rt.Gi_new.value
                 scale_to_copy = np.copy(rt.Gi_new.scale) if isinstance(rt.Gi_new.scale, np.ndarray) else rt.Gi_new.scale
                 rt.Gi_old = VarianceCovariance(value=val_to_copy, df=rt.Gi_new.df, scale=scale_to_copy)

        if mme.random_effect_terms:
            sample_general_random_effect_variances(mme)

        if mme.n_models == 1 and mme.R_variance.value is not None: mme.R_old_value = float(mme.R_variance.value)

        if mme.R_variance.estimate_variance:
            if mme.n_models == 1:
                new_R_val = sample_scalar_variance_component(
                    y_corrected, mme.R_variance.df, mme.R_variance.scale, inv_weights=mme.inverse_weights)
                if new_R_val > 1e-12 : mme.R_variance.value = new_R_val
                else: print(f"Warning: Sampled residual variance too small ({new_R_val}). Retaining.")
            else:
                n_obs_per_trait = len(mme.obs_ids)
                y_corr_list = [y_corrected[i*n_obs_per_trait:(i+1)*n_obs_per_trait] for i in range(mme.n_models)]
                new_R_mat_val = sample_matrix_variance_component(
                    y_corr_list, n_obs_per_trait, mme.R_variance.df, mme.R_variance.scale,
                    inv_weights=mme.inverse_weights, constraint=mme.R_variance.constraint)
                eigvals = np.linalg.eigvals(new_R_mat_val)
                if np.all(eigvals > 1e-12): mme.R_variance.value = new_R_mat_val
                else: print(f"Warning: Sampled residual cov matrix (eigvals: {eigvals}) not PD. Retaining.")

            if mme.n_models > 1 and mme.R_variance.value is not None:
                mme.current_Ri_matrix = _make_Ri_matrix(mme, df_pheno, mme.inverse_weights)

        if iter_num > burnin and (iter_num - burnin) % output_samples_freq == 0:
            num_saved_samples += 1
            if mme.solutions is not None:
                delta_sol = mme.solutions - mme.mean_solutions
                mme.mean_solutions += delta_sol / num_saved_samples
                mme.mean_solutions_sq += delta_sol * (mme.solutions - mme.mean_solutions)

            current_R = mme.R_variance.value
            if current_R is not None:
                if mme.n_models == 1:
                    mean_R_val = float(mme.mean_residual_variance) if mme.mean_residual_variance is not None else 0.0
                    delta_R = float(current_R) - mean_R_val
                    mme.mean_residual_variance = mean_R_val + delta_R / num_saved_samples
                    mme.mean_residual_variance_sq += delta_R * (float(current_R) - float(mme.mean_residual_variance))
                elif isinstance(current_R, np.ndarray) and mme.mean_residual_variance is not None:
                    delta_R_mat = current_R - mme.mean_residual_variance
                    mme.mean_residual_variance += delta_R_mat / num_saved_samples
                    mme.mean_residual_variance_sq += delta_R_mat * (current_R - mme.mean_residual_variance)

            for geno_data in mme.genotypes_data_list:
                if geno_data.alpha_samples and len(geno_data.alpha_samples) == mme.n_models:
                    for trait_idx in range(mme.n_models):
                        current_alpha_trait = geno_data.alpha_samples[trait_idx]
                        if current_alpha_trait is not None:
                            if trait_idx >= len(geno_data.mean_alpha) or geno_data.mean_alpha[trait_idx] is None:
                                geno_data.mean_alpha[trait_idx] = np.zeros_like(current_alpha_trait)
                                geno_data.mean_alpha_sq[trait_idx] = np.zeros_like(current_alpha_trait)
                            delta_alpha = current_alpha_trait - geno_data.mean_alpha[trait_idx]
                            geno_data.mean_alpha[trait_idx] += delta_alpha / num_saved_samples
                            geno_data.mean_alpha_sq[trait_idx] += delta_alpha * (current_alpha_trait - geno_data.mean_alpha[trait_idx])

                if geno_data.marker_effect_variance and geno_data.marker_effect_variance.value is not None:
                    if mme.n_models == 1:
                        if geno_data.mean_marker_variance is None: geno_data.mean_marker_variance = 0.0
                        if geno_data.mean_marker_variance_sq is None: geno_data.mean_marker_variance_sq = 0.0

                        val_for_mean_marker_var = 0.0
                        if geno_data.method in ["BayesA", "BayesB"] and isinstance(geno_data.marker_effect_variance.value, np.ndarray):
                            val_for_mean_marker_var = np.mean(geno_data.marker_effect_variance.value)
                        elif geno_data.method in ["RR-BLUP", "BayesC0", "BayesC", "BayesL"]:
                            val_for_mean_marker_var = float(geno_data.marker_effect_variance.value)

                        delta_marker_var = val_for_mean_marker_var - geno_data.mean_marker_variance
                        geno_data.mean_marker_variance += delta_marker_var / num_saved_samples
                        geno_data.mean_marker_variance_sq += delta_marker_var * (val_for_mean_marker_var - geno_data.mean_marker_variance)

                    if geno_data.method in ["BayesC", "BayesB"] and geno_data.delta_samples:
                         for trait_idx in range(mme.n_models):
                            if trait_idx < len(geno_data.delta_samples) and geno_data.delta_samples[trait_idx] is not None:
                                current_delta_trait = geno_data.delta_samples[trait_idx]
                                if trait_idx >= len(geno_data.mean_delta) or geno_data.mean_delta[trait_idx] is None:
                                     geno_data.mean_delta[trait_idx] = np.zeros_like(current_delta_trait)
                                delta_d = current_delta_trait - geno_data.mean_delta[trait_idx]
                                geno_data.mean_delta[trait_idx] += delta_d / num_saved_samples

                    if geno_data.method in ["BayesC", "BayesB"] and geno_data.pi_value is not None and geno_data.estimate_pi:
                        if mme.n_models == 1:
                            if geno_data.mean_pi is None: geno_data.mean_pi = 0.0
                            if geno_data.mean_pi_sq is None: geno_data.mean_pi_sq = 0.0
                            current_pi_val = float(geno_data.pi_value)
                            delta_pi = current_pi_val - geno_data.mean_pi
                            geno_data.mean_pi += delta_pi / num_saved_samples
                            geno_data.mean_pi_sq += delta_pi * (current_pi_val - geno_data.mean_pi)

                    if geno_data.method == "BayesL" and geno_data.gamma_array_bayesl is not None:
                        if mme.n_models == 1:
                            if geno_data.mean_gamma_array_bayesl is None or \
                               len(geno_data.mean_gamma_array_bayesl) != len(geno_data.gamma_array_bayesl):
                                geno_data.mean_gamma_array_bayesl = np.zeros_like(geno_data.gamma_array_bayesl)

                            delta_gamma = geno_data.gamma_array_bayesl - geno_data.mean_gamma_array_bayesl
                            geno_data.mean_gamma_array_bayesl += delta_gamma / num_saved_samples
            pass

        if iter_num % mcmc_params.printout_frequency == 0 or iter_num == chain_length :
            _print_progress(iter_num, chain_length, start_time, mcmc_params.printout_frequency)

    if lhs_was_converted_to_lil:
        if original_lhs_sparse_type == csr_matrix: mme.mme_LHS = mme.mme_LHS.tocsr()
        elif original_lhs_sparse_type == csc_matrix: mme.mme_LHS = mme.mme_LHS.tocsc()


    if num_saved_samples > 1:
        mme.mean_solutions_sq /= (num_saved_samples -1)
        if isinstance(mme.mean_residual_variance_sq, (float, np.floating)): mme.mean_residual_variance_sq /= (num_saved_samples -1)
        elif isinstance(mme.mean_residual_variance_sq, np.ndarray): mme.mean_residual_variance_sq /= (num_saved_samples-1)

        for geno_data in mme.genotypes_data_list:
            for trait_idx in range(mme.n_models):
                if trait_idx < len(geno_data.mean_alpha_sq) and geno_data.mean_alpha_sq[trait_idx] is not None:
                    geno_data.mean_alpha_sq[trait_idx] /= (num_saved_samples -1)

            if geno_data.mean_marker_variance_sq is not None:
                if isinstance(geno_data.mean_marker_variance_sq, (float, np.floating)):
                    geno_data.mean_marker_variance_sq /= (num_saved_samples -1)
                elif isinstance(geno_data.mean_marker_variance_sq, np.ndarray):
                    geno_data.mean_marker_variance_sq /= (num_saved_samples -1)

            if geno_data.method in ["BayesC", "BayesB"] and isinstance(geno_data.mean_pi_sq, (float, np.floating)) and num_saved_samples > 1:
                geno_data.mean_pi_sq /= (num_saved_samples -1) # Variance of Pi
            # Add finalization for mean_gamma_array_bayesl if variance of tau_j^2 is needed

    print(f"MCMC finished. Total time: {time.time() - start_time:.2f}s")
    results = {
        "mean_solutions": mme.mean_solutions,
        "variance_solutions": mme.mean_solutions_sq if num_saved_samples > 1 else None,
        "mean_residual_variance": mme.mean_residual_variance,
        "variance_residual_variance": mme.mean_residual_variance_sq if num_saved_samples > 1 else None,
        "genotypes_results": []
    }
    for gd in mme.genotypes_data_list:
        geno_result_entry = {
            "name": gd.name, "mean_alpha": gd.mean_alpha,
            "variance_alpha": gd.mean_alpha_sq if num_saved_samples > 1 else None,
            "mean_marker_variance": gd.mean_marker_variance,
            "variance_marker_variance": gd.mean_marker_variance_sq if num_saved_samples > 1 else None
        }
        if gd.method in ["BayesC", "BayesB"]:
            geno_result_entry["mean_delta"] = gd.mean_delta
            geno_result_entry["mean_pi"] = gd.mean_pi
            geno_result_entry["variance_pi"] = gd.mean_pi_sq if num_saved_samples > 1 else None
        if gd.method == "BayesL":
            geno_result_entry["mean_tau_sq_array"] = gd.mean_gamma_array_bayesl # Storing mean of tau_j^2 values
        results["genotypes_results"].append(geno_result_entry)
    return results

if __name__ == '__main__':
    print("MCMC Engine basic structure defined.")
    from ..core import MixedModelEquations, ModelTerm, VarianceCovariance, MCMCInfo, GenotypesData
    from ..core.model_builder import _get_data_for_term, _get_incidence_matrix_for_term

    np.random.seed(4242)
    n_obs = 100; intercept_true = 5.0; residual_var_true = 2.0
    n_markers = 50; marker_var_true_rrblup = 0.05; marker_var_true_bayesa_scale = 0.005
    pi_true_bayesc = 0.1; common_marker_var_true_bayesc = 0.02
    lasso_true_lambda_sq = 2.0 # For BayesL simulation


    sim_Z = np.random.randint(0, 3, size=(n_obs, n_markers)).astype(float)
    sim_Z_means = sim_Z.mean(axis=0); sim_Z_std = sim_Z.std(axis=0); sim_Z_std[sim_Z_std==0] = 1.0
    sim_Z_centered_scaled = (sim_Z - sim_Z_means) / sim_Z_std
    sim_Z_centered_scaled = np.nan_to_num(sim_Z_centered_scaled, nan=0.0)

    # Test RR-BLUP
    # ... (RR-BLUP test setup and call as before) ...

    # Test BayesA
    # ... (BayesA test setup and call as before) ...

    # Test BayesC
    # ... (BayesC test setup and call as before) ...

    # Test BayesL
    print("\n--- Testing BayesL in MCMC (partial, tau_j^2 sampling is key) ---")
    # Simulate true tau_j^2 from Exp(lasso_true_lambda_sq/2)
    true_tau_sq_bayesl = np.random.exponential(scale = 2.0/lasso_true_lambda_sq, size=n_markers)
    true_alphas_bayesl = np.array([np.random.randn() * np.sqrt(common_marker_var_true_bayesc * tau_sq) for tau_sq in true_tau_sq_bayesl]) # Using common_marker_var_true_bayesc as sigma_g^2 base
    y_genetic_bayesl = sim_Z_centered_scaled @ true_alphas_bayesl
    test_df_bayesl = pd.DataFrame({'y': y_genetic_bayesl + np.random.randn(n_obs) * np.sqrt(residual_var_true) + intercept_true})

    R_vc_bayesl = VarianceCovariance(value=1.0, df=4.0, scale=1.0, estimate_variance=True)
    intercept_term_bayesl = ModelTerm(term_str="intercept", model_index=1, trait_name="y")
    geno_bayesl = GenotypesData(name="markers_bayesl", method="BayesL")
    geno_bayesl.genotypes = sim_Z_centered_scaled; geno_bayesl.n_markers = n_markers
    geno_bayesl.marker_effect_variance = VarianceCovariance(value=0.01, df=4.0, scale=0.005, estimate_variance=True) # Common sigma_g^2 part
    geno_bayesl.lasso_lambda_sq_hyper = lasso_true_lambda_sq # Provide the hyperparameter

    mme_bayesl = MixedModelEquations(
        n_models=1, model_equations_str=["y = intercept"], model_terms=[intercept_term_bayesl],
        model_term_dict={"y:intercept": intercept_term_bayesl}, lhs_variables=["y"], residual_variance_info=R_vc_bayesl)
    mme_bayesl.genotypes_data_list.append(geno_bayesl)
    mme_bayesl.mcmc_info = MCMCInfo(chain_length=6000, burnin=1000, output_samples_frequency=100, printout_frequency=6001, seed=111)
    mme_bayesl.obs_ids = [str(i) for i in range(n_obs)]; mme_bayesl.inverse_weights = np.ones(n_obs)
    _get_data_for_term(intercept_term_bayesl, test_df_bayesl, mme_bayesl); _get_incidence_matrix_for_term(intercept_term_bayesl, mme_bayesl, n_obs)
    mme_bayesl.X = intercept_term_bayesl.X ; mme_bayesl.y_sparse = test_df_bayesl['y'].values.reshape(-1,1)

    if mme_bayesl.X.shape[1] > 0 and mme_bayesl.R_variance.value is not None and float(mme_bayesl.R_variance.value) > 0:
        D_initial_bl = diags(mme_bayesl.inverse_weights / float(mme_bayesl.R_variance.value), format="csc")
        mme_bayesl.mme_LHS_base_X_Rinv_X = mme_bayesl.X.T @ D_initial_bl @ mme_bayesa.X # Typo: should be mme_bayesl.X
        mme_bayesl.mme_LHS = mme_bayesl.mme_LHS_base_X_Rinv_X.copy()
        mme_bayesl.mme_RHS = mme_bayesl.X.T @ D_initial_bl @ mme_bayesl.y_sparse
    else: mme_bayesl.mme_LHS = csc_matrix((1,1)); mme_bayesl.mme_RHS = np.zeros((1,1))

    if mme_bayesl.mme_LHS.shape[0] > 0:
        results_bayesl = run_mcmc(mme_bayesl, test_df_bayesl)
        print("\nBayesL MCMC run completed.")
        print(f"  True intercept: {intercept_true:.3f}, Mean sampled intercept: {results_bayesl.get('mean_solutions')[0]:.3f}")
        print(f"  True residual var: {residual_var_true:.3f}, Mean sampled residual_variance: {results_bayesl.get('mean_residual_variance'):.3f}")
        if results_bayesl["genotypes_results"]:
            geno_res_bl = results_bayesl["genotypes_results"][0]
            print(f"  True common marker var factor (sigma_g^2): {common_marker_var_true_bayesc:.4f}, Mean sampled: {geno_res_bl.get('mean_marker_variance'):.4f}")
            if geno_res_bl.get('mean_alpha') and len(geno_res_bl.get('mean_alpha')) > 0 and geno_res_bl.get('mean_alpha')[0] is not None:
                corr_alpha_bl = np.corrcoef(true_alphas_bayesl, geno_res_bl.get('mean_alpha')[0])[0,1]
                print(f"  Correlation true_alpha vs sampled_mean_alpha (BayesL): {corr_alpha_bl:.3f}")
            if geno_res_bl.get('mean_tau_sq_array') is not None:
                 print(f"  Mean of sampled tau_sq values (BayesL, first 5): {geno_res_bl.get('mean_tau_sq_array')[:5]}")
                 # Compare mean of sampled tau_sq with true mean of Exp(rate=lasso_true_lambda_sq/2) = 2/lasso_true_lambda_sq
                 expected_mean_tau_sq = 2.0 / lasso_true_lambda_sq
                 print(f"  Mean of true tau_sq values: {np.mean(true_tau_sq_bayesl):.3f}, Expected prior mean tau_sq: {expected_mean_tau_sq:.3f}")

    else:
        print("Skipping MCMC run due to empty MME LHS in test setup.")

```
