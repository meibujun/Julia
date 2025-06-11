import numpy as np
import scipy.linalg # For eig, and potentially block_diag if needed in future
from animal_breeding_genetics.matrix_ops import (
    multiply_matrices,
    invert_matrix,
)
from animal_breeding_genetics.blup_animal_model import solve_mme

def setup_mme_multivariate_animal_model_equal_design(
    Y_matrix: np.ndarray,      # n_records x n_traits
    X_common: np.ndarray,      # n_records x n_fixed_effects (common for all traits)
    Z_common: np.ndarray,      # n_records x n_animals (common for all traits)
    A_inverse: np.ndarray,    # n_animals x n_animals
    R_0_inv: np.ndarray,      # n_traits x n_traits, inverse of residual cov matrix
    G_0_inv: np.ndarray       # n_traits x n_traits, inverse of genetic cov matrix
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Sets up MME for a Multivariate Animal Model assuming equal design matrices (X, Z)
    for all traits and no missing data (Y_matrix is dense).

    Solutions ordered as: b = [b_fx1_tr1, b_fx1_tr2, ..., b_fxp_trT]',
                           u = [u_an1_tr1, u_an1_tr2, ..., u_anN_trT]'
    This requires specific Kronecker product order: e.g., X'X (x) R_0_inv.

    Args:
        Y_matrix (np.ndarray): Observation matrix (n_records x n_traits).
        X_common (np.ndarray): Common incidence matrix for fixed effects (n_records x n_common_fixed).
        Z_common (np.ndarray): Common incidence matrix for random animal effects (n_records x n_animals).
        A_inverse (np.ndarray): Inverse of Numerator Relationship Matrix (n_animals x n_animals).
        R_0_inv (np.ndarray): Inverse of residual covariance matrix (n_traits x n_traits).
        G_0_inv (np.ndarray): Inverse of genetic covariance matrix (n_traits x n_traits).

    Returns:
        tuple[np.ndarray, np.ndarray]: LHS matrix and RHS vector.
    """
    n_records, num_traits_y = Y_matrix.shape
    n_common_fixed = X_common.shape[1]
    num_animals = Z_common.shape[1]

    if X_common.shape[0] != n_records or Z_common.shape[0] != n_records:
        raise ValueError("X_common and Z_common must have n_records rows.")
    if A_inverse.shape != (num_animals, num_animals):
        raise ValueError("A_inverse dimension mismatch.")
    if R_0_inv.shape != (num_traits_y, num_traits_y) or G_0_inv.shape != (num_traits_y, num_traits_y):
        raise ValueError("R_0_inv and G_0_inv must be (num_traits x num_traits).")

    # LHS blocks using Kronecker product
    # Ordering of solutions: b_fx1_tr1, b_fx1_tr2, ..., b_fxp_trT for fixed effects
    #                       u_an1_tr1, u_an1_tr2, ..., u_anN_trT for random effects
    # This means fixed effects are clustered by effect, then trait.
    # And random effects are clustered by animal, then trait.

    X_T_X = X_common.T @ X_common
    X_T_Z = X_common.T @ Z_common
    Z_T_X = Z_common.T @ X_common # or X_T_Z.T
    Z_T_Z = Z_common.T @ Z_common

    LHS_XX = np.kron(X_T_X, R_0_inv)
    LHS_XZ = np.kron(X_T_Z, R_0_inv)
    LHS_ZX = np.kron(Z_T_X, R_0_inv) # or LHS_XZ.T if R_0_inv is symmetric
    LHS_ZZ = np.kron(Z_T_Z, R_0_inv) + np.kron(A_inverse, G_0_inv)

    LHS = np.block([
        [LHS_XX, LHS_XZ],
        [LHS_ZX, LHS_ZZ]
    ])

    # RHS blocks
    # Y_matrix is (n_records x n_traits)
    # RHS_b must be (n_common_fixed * n_traits x 1)
    # RHS_u must be (num_animals * n_traits x 1)

    # X_common.T @ Y_matrix results in (n_common_fixed x n_traits)
    # We need to "vectorize" this by columns, pre-multiplied by R_0_inv
    # (X_common.T @ Y_matrix) @ R_0_inv (use R_0_inv.T if R_0_inv is not symmetric, but it should be)
    # Result is (n_common_fixed x n_traits). Flatten column-wise for Fortran order.
    RHS_b_matrix = multiply_matrices(X_common.T @ Y_matrix, R_0_inv)
    RHS_b = RHS_b_matrix.flatten(order='F').reshape(-1,1)

    RHS_u_matrix = multiply_matrices(Z_common.T @ Y_matrix, R_0_inv)
    RHS_u = RHS_u_matrix.flatten(order='F').reshape(-1,1)

    RHS = np.vstack([RHS_b, RHS_u])

    return LHS, RHS

# Placeholder for the general version initially attempted.
def setup_mme_multivariate_animal_model_general(
    Y_list: list[np.ndarray], X_list: list[np.ndarray], Z_list: list[np.ndarray],
    A_inverse: np.ndarray, R_0_inv: np.ndarray, G_0_inv: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    raise NotImplementedError("General multivariate MME setup is complex and deferred.")


def canonical_transform(
    G_0: np.ndarray, R_0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs canonical transformation on genetic (G_0) and residual (R_0)
    covariance matrices for traits.
    Solves G_0 q = lambda R_0 q.
    Returns Q, Lambda (diag eigenvalues), Q'G_0Q, Q'R_0Q.
    Theoretically, Q'R_0Q should be I and Q'G_0Q should be Lambda.
    """
    num_traits = G_0.shape[0]
    if G_0.shape != (num_traits, num_traits) or R_0.shape != (num_traits, num_traits):
        raise ValueError("G_0 and R_0 must be square matrices of the same dimension.")

    # scipy.linalg.eig solves A x = lambda B x, returns eigenvalues and RIGHT eigenvectors
    # Normalization: For symmetric A and B, eigenvectors are normalized such that
    # if B is positive definite, Q.T @ B @ Q = I.
    try:
        eigenvalues, eigenvectors = scipy.linalg.eig(G_0, R_0)

        # Ensure eigenvalues are real, take real part if complex (can happen with num. instability)
        # For real symmetric G_0 and real symmetric positive-definite R_0, eigenvalues must be real.
        # Eigenvectors can be chosen to be real.
        if np.any(np.iscomplex(eigenvalues)):
            # This case should ideally not happen for valid inputs.
            # If it does, it indicates numerical issues or problematic matrices.
            # Taking .real might hide problems but allow code to run.
            # print("Warning: Complex eigenvalues encountered in canonical_transform. Taking real part.")
            eigenvalues = eigenvalues.real

        if np.any(np.iscomplex(eigenvectors)):
            # print("Warning: Complex eigenvectors encountered in canonical_transform. Taking real part.")
            eigenvectors = eigenvectors.real

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1] # np.argsort works on real or complex arrays
        eigenvalues_sorted = eigenvalues[sorted_indices]
        Q_transform = eigenvectors[:, sorted_indices]

        Lambda_diag_sorted = np.diag(eigenvalues_sorted)

        # Calculate transformed matrices for verification by the caller/test
        G_0_canonical = Q_transform.T @ G_0 @ Q_transform
        R_0_canonical = Q_transform.T @ R_0 @ Q_transform

        return Q_transform, Lambda_diag_sorted, G_0_canonical, R_0_canonical
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Eigen-decomposition for canonical transform failed: {e}")


def solve_multivariate_animal_model_blup(
    Y_matrix: np.ndarray, X_common: np.ndarray, Z_common: np.ndarray,
    A_inverse: np.ndarray, R_0_inv: np.ndarray, G_0_inv: np.ndarray, G_0: np.ndarray
) -> dict:
    """
    Orchestrates setup (for equal design matrices), solving, and PEV/accuracy
    for a Multivariate Animal Model.
    Solutions ordered b_fx1_tr1,... then u_an1_tr1,...
    """
    results = {
        'b_hat_stacked': None, 'u_hat_stacked': None,
        'pev_stacked': None, 'accuracy_stacked_traits': None, # List of accuracies per trait
        'LHS': None, 'RHS': None, 'error': None
    }

    n_common_fixed = X_common.shape[1]
    num_animals = A_inverse.shape[0]
    num_traits = G_0.shape[0]

    try:
        LHS, RHS = setup_mme_multivariate_animal_model_equal_design(
            Y_matrix, X_common, Z_common, A_inverse, R_0_inv, G_0_inv
        )
        results['LHS'], results['RHS'] = LHS, RHS

        solution_vector = solve_mme(LHS, RHS)
        if solution_vector is None:
            results['error'] = "Multivariate MME solving failed."
            return results

        total_fixed_effects = n_common_fixed * num_traits
        results['b_hat_stacked'] = solution_vector[:total_fixed_effects]
        results['u_hat_stacked'] = solution_vector[total_fixed_effects:]

        LHS_inv = invert_matrix(LHS)
        C_random_effects_full_block = LHS_inv[total_fixed_effects:, total_fixed_effects:]

        all_pev_diagonal = np.diag(C_random_effects_full_block)
        all_pev_diagonal = np.maximum(all_pev_diagonal, 0)
        results['pev_stacked'] = all_pev_diagonal.reshape(-1,1)

        # Accuracies: u_hat_stacked is [u_an1_tr1, u_an1_tr2, ..., u_anN_trT]'
        # PEVs also correspond to this ordering.
        # G_0[t,t] is var(g_t)
        accuracies_per_animal_trait = []
        for i in range(num_animals):
            for t in range(num_traits):
                idx = i * num_traits + t # Index in the stacked u_hat and pev vectors
                pev_it = all_pev_diagonal[idx]
                var_g_t = G_0[t,t]
                if var_g_t <= 0: acc = 0.0
                else:
                    reliability_it = 1 - pev_it / var_g_t
                    acc = np.sqrt(np.maximum(reliability_it, 0))
                accuracies_per_animal_trait.append(acc)
        results['accuracy_stacked_traits'] = np.array(accuracies_per_animal_trait).reshape(-1,1)

    except (ValueError, np.linalg.LinAlgError) as e:
        results['error'] = f"Error in multivariate BLUP: {str(e)}"
    except Exception as e:
        results['error'] = f"An unexpected error in multivariate BLUP: {str(e)}"
    return results
