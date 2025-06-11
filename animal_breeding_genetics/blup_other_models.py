import numpy as np
from animal_breeding_genetics.matrix_ops import (
    multiply_matrices,
    invert_matrix,
    # solve_linear_equations is used by solve_mme from blup_animal_model
)
from animal_breeding_genetics.blup_animal_model import solve_mme, calculate_pev_and_accuracy

def setup_mme_sire_model(
    y_vector: np.ndarray,
    X_matrix: np.ndarray,
    Z_sire_matrix: np.ndarray,
    A_sire_inverse_matrix: np.ndarray,
    sigma_e_sq: float,
    sigma_sire_sq: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sets up MME for a Sire Model.
    MME: [X'X, X'Z_s; Z_s'X, Z_s'Z_s + A_s_inv*k_s] [b;s] = [X'y; Z_s'y]
    k_s = sigma_e_sq / sigma_sire_sq.
    """
    if sigma_e_sq <= 0: raise ValueError("sigma_e_sq must be positive.")
    if sigma_sire_sq <= 0: raise ValueError("sigma_sire_sq must be positive.")

    if y_vector.ndim == 1: y_vector = y_vector.reshape(-1, 1)

    n_obs = y_vector.shape[0]
    if not (X_matrix.shape[0] == n_obs and Z_sire_matrix.shape[0] == n_obs):
        raise ValueError("X and Z_sire matrices must have rows equal to y_vector length.")

    n_sires = Z_sire_matrix.shape[1]
    if not (A_sire_inverse_matrix.shape == (n_sires, n_sires)):
        raise ValueError(f"A_sire_inverse_matrix must be {n_sires}x{n_sires}.")

    X_T = X_matrix.T
    Z_s_T = Z_sire_matrix.T

    X_T_X = multiply_matrices(X_T, X_matrix)
    X_T_Z_s = multiply_matrices(X_T, Z_sire_matrix)
    Z_s_T_X = X_T_Z_s.T
    Z_s_T_Z_s = multiply_matrices(Z_s_T, Z_sire_matrix)

    k_s = sigma_e_sq / sigma_sire_sq
    A_s_inv_k_s = A_sire_inverse_matrix * k_s

    LHS_bottom_right = Z_s_T_Z_s + A_s_inv_k_s

    LHS = np.block([[X_T_X, X_T_Z_s], [Z_s_T_X, LHS_bottom_right]])
    RHS = np.vstack([multiply_matrices(X_T, y_vector), multiply_matrices(Z_s_T, y_vector)])

    return LHS, RHS

def solve_sire_model_blup(
    y_vector: np.ndarray, X_matrix: np.ndarray, Z_sire_matrix: np.ndarray,
    A_sire_inverse_matrix: np.ndarray, sigma_e_sq: float, sigma_sire_sq: float
) -> dict:
    """Orchestrates Sire Model MME setup, solving, and PEV/accuracy."""
    results = {'b_hat': None, 's_hat': None, 'pev_sires': None, 'accuracy_sires': None,
               'LHS': None, 'RHS': None, 'error': None}
    try:
        if sigma_sire_sq <= 0: # Check before setup uses it in k_s
            results['error'] = "Sire variance (sigma_sire_sq) must be positive."
            return results
        LHS, RHS = setup_mme_sire_model(y_vector, X_matrix, Z_sire_matrix, A_sire_inverse_matrix, sigma_e_sq, sigma_sire_sq)
        results['LHS'], results['RHS'] = LHS, RHS
        solutions = solve_mme(LHS, RHS)
        if solutions is None:
            results['error'] = "MME solving failed (LHS might be singular)."
            return results

        num_fixed_effects = X_matrix.shape[1]
        results['b_hat'] = solutions[:num_fixed_effects]
        results['s_hat'] = solutions[num_fixed_effects:]

        LHS_inv = invert_matrix(LHS)
        pev_s, acc_s = calculate_pev_and_accuracy(
            LHS_inv,
            sigma_e_sq=sigma_e_sq,
            sigma_random_effect_sq=sigma_sire_sq,
            start_idx_block=num_fixed_effects,
            end_idx_block=LHS_inv.shape[0]
        )
        results['pev_sires'], results['accuracy_sires'] = pev_s, acc_s
    except (ValueError, np.linalg.LinAlgError) as e: results['error'] = f"Sire model error: {str(e)}"
    except Exception as e: results['error'] = f"Unexpected error in sire model: {str(e)}" # Catch-all for other issues
    return results

def setup_mme_repeatability_model(
    y_vector: np.ndarray, X_matrix: np.ndarray, Z_animal_matrix: np.ndarray,
    Z_pe_matrix: np.ndarray, A_inverse_matrix: np.ndarray,
    sigma_e_sq: float, sigma_a_sq: float, sigma_pe_sq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sets up MME for a Repeatability Model (Animal + Permanent Environment)."""
    if not (sigma_e_sq > 0 and sigma_a_sq > 0 and sigma_pe_sq > 0):
        raise ValueError("All variances (sigma_e_sq, sigma_a_sq, sigma_pe_sq) must be positive.")

    if y_vector.ndim == 1: y_vector = y_vector.reshape(-1, 1)
    n_obs = y_vector.shape[0]
    if not (X_matrix.shape[0] == n_obs and Z_animal_matrix.shape[0] == n_obs and Z_pe_matrix.shape[0] == n_obs):
        raise ValueError("X, Z_animal, Z_pe matrices must have rows equal to y_vector length.")
    if Z_animal_matrix.shape[1] != A_inverse_matrix.shape[0]:
        raise ValueError("Z_animal_matrix columns must match A_inverse_matrix dimensions.")

    n_pe_effects = Z_pe_matrix.shape[1]
    X_T, Z_a_T, Z_pe_T = X_matrix.T, Z_animal_matrix.T, Z_pe_matrix.T

    X_T_X = multiply_matrices(X_T, X_matrix)
    X_T_Z_a = multiply_matrices(X_T, Z_animal_matrix)
    X_T_Z_pe = multiply_matrices(X_T, Z_pe_matrix)

    Z_a_T_X = X_T_Z_a.T
    Z_a_T_Z_a = multiply_matrices(Z_a_T, Z_animal_matrix)
    Z_a_T_Z_pe = multiply_matrices(Z_a_T, Z_pe_matrix)

    Z_pe_T_X = X_T_Z_pe.T
    Z_pe_T_Z_a = Z_a_T_Z_pe.T
    Z_pe_T_Z_pe = multiply_matrices(Z_pe_T, Z_pe_matrix)

    k_a, k_pe = sigma_e_sq / sigma_a_sq, sigma_e_sq / sigma_pe_sq
    LHS_aa_block = Z_a_T_Z_a + A_inverse_matrix * k_a
    LHS_pepe_block = Z_pe_T_Z_pe + np.eye(n_pe_effects) * k_pe

    LHS = np.block([[X_T_X, X_T_Z_a, X_T_Z_pe], [Z_a_T_X, LHS_aa_block, Z_a_T_Z_pe], [Z_pe_T_X, Z_pe_T_Z_a, LHS_pepe_block]])
    RHS = np.vstack([multiply_matrices(X_T, y_vector), multiply_matrices(Z_a_T, y_vector), multiply_matrices(Z_pe_T, y_vector)])
    return LHS, RHS

def solve_repeatability_model_blup(
    y_vector: np.ndarray, X_matrix: np.ndarray, Z_animal_matrix: np.ndarray, Z_pe_matrix: np.ndarray,
    A_inverse_matrix: np.ndarray, sigma_e_sq: float, sigma_a_sq: float, sigma_pe_sq: float
) -> dict:
    """Orchestrates Repeatability Model MME setup, solving, and PEV/accuracy."""
    results = {'b_hat': None, 'u_hat': None, 'pe_hat': None, 'pev_u': None, 'accuracy_u': None,
               'pev_pe': None, 'accuracy_pe': None, 'LHS': None, 'RHS': None, 'error': None}
    try:
        if not (sigma_a_sq > 0 and sigma_pe_sq > 0): # sigma_e_sq checked in setup
            results['error'] = "sigma_a_sq and sigma_pe_sq must be positive."
            return results
        LHS, RHS = setup_mme_repeatability_model(y_vector, X_matrix, Z_animal_matrix, Z_pe_matrix, A_inverse_matrix, sigma_e_sq, sigma_a_sq, sigma_pe_sq)
        results['LHS'], results['RHS'] = LHS, RHS
        solutions = solve_mme(LHS, RHS)
        if solutions is None:
            results['error'] = "MME solving failed."
            return results

        n_fixed, n_animal = X_matrix.shape[1], Z_animal_matrix.shape[1]
        results['b_hat'], results['u_hat'], results['pe_hat'] = solutions[:n_fixed], solutions[n_fixed:n_fixed+n_animal], solutions[n_fixed+n_animal:]

        LHS_inv = invert_matrix(LHS)
        pev_u, acc_u = calculate_pev_and_accuracy(
            LHS_inv,
            sigma_e_sq=sigma_e_sq,
            sigma_random_effect_sq=sigma_a_sq,
            start_idx_block=n_fixed,
            end_idx_block=n_fixed+n_animal
        )
        results['pev_u'], results['accuracy_u'] = pev_u, acc_u

        pev_pe, acc_pe = calculate_pev_and_accuracy(
            LHS_inv,
            sigma_e_sq=sigma_e_sq,
            sigma_random_effect_sq=sigma_pe_sq,
            start_idx_block=n_fixed+n_animal,
            end_idx_block=LHS_inv.shape[0]
        )
        results['pev_pe'], results['accuracy_pe'] = pev_pe, acc_pe

    except (ValueError, np.linalg.LinAlgError) as e: results['error'] = f"Repeatability model error: {str(e)}"
    except Exception as e: results['error'] = f"Unexpected error in repeatability model: {str(e)}"
    return results

def setup_mme_common_env_model(
    y_vector: np.ndarray, X_matrix: np.ndarray, Z_animal_matrix: np.ndarray,
    Z_common_env_matrix: np.ndarray, A_inverse_matrix: np.ndarray,
    sigma_e_sq: float, sigma_a_sq: float, sigma_common_env_sq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sets up MME for a model with Animal + Common Environmental Effects (e.g., litter)."""
    if not (sigma_e_sq > 0 and sigma_a_sq > 0 and sigma_common_env_sq > 0):
        raise ValueError("All variances (sigma_e_sq, sigma_a_sq, sigma_common_env_sq) must be positive.")

    if y_vector.ndim == 1: y_vector = y_vector.reshape(-1, 1)
    n_obs = y_vector.shape[0]
    if not (X_matrix.shape[0] == n_obs and Z_animal_matrix.shape[0] == n_obs and Z_common_env_matrix.shape[0] == n_obs):
        raise ValueError("X, Z_animal, Z_common_env matrices must have rows equal to y_vector length.")
    if Z_animal_matrix.shape[1] != A_inverse_matrix.shape[0]:
        raise ValueError("Z_animal_matrix columns must match A_inverse_matrix dimensions.")

    n_common_env_effects = Z_common_env_matrix.shape[1]
    X_T, Z_a_T, Z_c_T = X_matrix.T, Z_animal_matrix.T, Z_common_env_matrix.T

    X_T_X = multiply_matrices(X_T, X_matrix)
    X_T_Z_a = multiply_matrices(X_T, Z_animal_matrix)
    X_T_Z_c = multiply_matrices(X_T, Z_common_env_matrix)

    Z_a_T_X = X_T_Z_a.T
    Z_a_T_Z_a = multiply_matrices(Z_a_T, Z_animal_matrix)
    Z_a_T_Z_c = multiply_matrices(Z_a_T, Z_common_env_matrix)

    Z_c_T_X = X_T_Z_c.T
    Z_c_T_Z_a = Z_a_T_Z_c.T
    Z_c_T_Z_c = multiply_matrices(Z_c_T, Z_common_env_matrix)

    k_a, k_c = sigma_e_sq / sigma_a_sq, sigma_e_sq / sigma_common_env_sq
    LHS_aa_block = Z_a_T_Z_a + A_inverse_matrix * k_a
    LHS_cc_block = Z_c_T_Z_c + np.eye(n_common_env_effects) * k_c

    LHS = np.block([[X_T_X, X_T_Z_a, X_T_Z_c], [Z_a_T_X, LHS_aa_block, Z_a_T_Z_c], [Z_c_T_X, Z_c_T_Z_a, LHS_cc_block]])
    RHS = np.vstack([multiply_matrices(X_T, y_vector), multiply_matrices(Z_a_T, y_vector), multiply_matrices(Z_c_T, y_vector)])
    return LHS, RHS

def solve_common_env_model_blup(
    y_vector: np.ndarray, X_matrix: np.ndarray, Z_animal_matrix: np.ndarray, Z_common_env_matrix: np.ndarray,
    A_inverse_matrix: np.ndarray, sigma_e_sq: float, sigma_a_sq: float, sigma_common_env_sq: float
) -> dict:
    """Orchestrates Common Environmental Model MME setup, solving, and PEV/accuracy."""
    results = {'b_hat': None, 'u_hat': None, 'c_hat': None, 'pev_u': None, 'accuracy_u': None,
               'pev_c': None, 'accuracy_c': None, 'LHS': None, 'RHS': None, 'error': None}
    try:
        if not (sigma_a_sq > 0 and sigma_common_env_sq > 0): # sigma_e_sq checked in setup
            results['error'] = "sigma_a_sq and sigma_common_env_sq must be positive."
            return results
        LHS, RHS = setup_mme_common_env_model(y_vector, X_matrix, Z_animal_matrix, Z_common_env_matrix, A_inverse_matrix, sigma_e_sq, sigma_a_sq, sigma_common_env_sq)
        results['LHS'], results['RHS'] = LHS, RHS
        solutions = solve_mme(LHS, RHS)
        if solutions is None:
            results['error'] = "MME solving failed."
            return results

        n_fixed, n_animal = X_matrix.shape[1], Z_animal_matrix.shape[1]
        results['b_hat'], results['u_hat'], results['c_hat'] = solutions[:n_fixed], solutions[n_fixed:n_fixed+n_animal], solutions[n_fixed+n_animal:]

        LHS_inv = invert_matrix(LHS)
        pev_u, acc_u = calculate_pev_and_accuracy(
            LHS_inv,
            sigma_e_sq=sigma_e_sq,
            sigma_random_effect_sq=sigma_a_sq,
            start_idx_block=n_fixed,
            end_idx_block=n_fixed+n_animal
        )
        results['pev_u'], results['accuracy_u'] = pev_u, acc_u

        pev_c, acc_c = calculate_pev_and_accuracy(
            LHS_inv,
            sigma_e_sq=sigma_e_sq,
            sigma_random_effect_sq=sigma_common_env_sq,
            start_idx_block=n_fixed+n_animal,
            end_idx_block=LHS_inv.shape[0]
        )
        results['pev_c'], results['accuracy_c'] = pev_c, acc_c

    except (ValueError, np.linalg.LinAlgError) as e: results['error'] = f"Common Env model error: {str(e)}"
    except Exception as e: results['error'] = f"Unexpected error in common env model: {str(e)}"
    return results

# Placeholder for Reduced Animal Model
def setup_mme_reduced_animal_model():
    raise NotImplementedError("Reduced Animal Model is not yet implemented.")

# Note: The calculate_pev_and_accuracy function from blup_animal_model.py is assumed
# to have been updated to accept start_idx_block and end_idx_block arguments
# for slicing LHS_inv correctly for different random effect blocks.
