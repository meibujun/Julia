import numpy as np
from animal_breeding_genetics.matrix_ops import (
    multiply_matrices,
    transpose_matrix, # Will need to implement or use .T
    invert_matrix,
    solve_linear_equations
)

def setup_mme_animal_model(
    y_vector: np.ndarray,
    X_matrix: np.ndarray,
    Z_matrix: np.ndarray,
    A_inverse_matrix: np.ndarray,
    sigma_e_sq: float,
    sigma_a_sq: float
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Sets up the Left-Hand Side (LHS) matrix and Right-Hand Side (RHS) vector
    for the Mixed Model Equations (MME) for a univariate animal model.

    MME structure:
    [ X'X         X'Z         ] [ b_hat ]   [ X'y ]
    [ Z'X         Z'Z + A_inv * k ] [ u_hat ] = [ Z'y ]
    where k = sigma_e_sq / sigma_a_sq.

    Args:
        y_vector (np.ndarray): Vector of phenotypic observations (n_obs x 1).
        X_matrix (np.ndarray): Incidence matrix for fixed effects (n_obs x n_fixed).
        Z_matrix (np.ndarray): Incidence matrix for random animal effects (n_obs x n_animals).
                               Columns should align with A_inverse_matrix.
        A_inverse_matrix (np.ndarray): Inverse of the Numerator Relationship Matrix (n_animals x n_animals).
        sigma_e_sq (float): Residual (error) variance. Must be positive.
        sigma_a_sq (float): Additive genetic variance. Must be positive.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None,None]:
            - LHS (np.ndarray): The LHS matrix of the MME.
            - RHS (np.ndarray): The RHS vector of the MME.
            Returns None, None if inputs are invalid.

    Raises:
        ValueError: If variances are not positive or dimensions are incompatible.
    """
    if sigma_e_sq <= 0:
        raise ValueError("Residual variance (sigma_e_sq) must be positive.")
    if sigma_a_sq <= 0:
        # This implies k is infinite or undefined. Model needs re-evaluation.
        # For now, strictly positive sigma_a_sq is required.
        raise ValueError("Additive genetic variance (sigma_a_sq) must be positive.")

    # Reshape y_vector to be a column vector if it's 1D
    if y_vector.ndim == 1:
        y_vector = y_vector.reshape(-1, 1)

    # Dimension checks
    n_obs_y = y_vector.shape[0]
    if X_matrix.ndim != 2 or X_matrix.shape[0] != n_obs_y:
        raise ValueError(f"X_matrix must be 2D with {n_obs_y} rows (matching y_vector).")
    if Z_matrix.ndim != 2 or Z_matrix.shape[0] != n_obs_y:
        raise ValueError(f"Z_matrix must be 2D with {n_obs_y} rows (matching y_vector).")

    n_animals_Z = Z_matrix.shape[1]
    if A_inverse_matrix.ndim != 2 or \
       A_inverse_matrix.shape[0] != n_animals_Z or \
       A_inverse_matrix.shape[1] != n_animals_Z:
        raise ValueError(f"A_inverse_matrix must be a square matrix with dimensions "
                         f"{n_animals_Z}x{n_animals_Z} (matching Z_matrix columns).")

    X_T = X_matrix.T # Transpose using .T attribute for NumPy arrays
    Z_T = Z_matrix.T

    # Calculate components of LHS
    X_T_X = multiply_matrices(X_T, X_matrix)
    X_T_Z = multiply_matrices(X_T, Z_matrix)
    Z_T_X = multiply_matrices(Z_T, X_matrix) # Or use X_T_Z.T if X_T_Z is already computed
    Z_T_Z = multiply_matrices(Z_T, Z_matrix)

    k = sigma_e_sq / sigma_a_sq
    A_inv_k = A_inverse_matrix * k # Element-wise multiplication for scalar k

    LHS_bottom_right = Z_T_Z + A_inv_k

    # Assemble LHS
    n_fixed = X_T_X.shape[1]
    n_random = Z_T_Z.shape[1]

    if X_T_X.shape[0] != n_fixed or X_T_Z.shape[0] != n_fixed or X_T_Z.shape[1] != n_random:
         raise ValueError("Internal dimension mismatch for X'X or X'Z.")
    if Z_T_X.shape[0] != n_random or Z_T_X.shape[1] != n_fixed or LHS_bottom_right.shape[0] != n_random or LHS_bottom_right.shape[1] != n_random:
         raise ValueError("Internal dimension mismatch for Z'X or Z'Z+A_inv*k.")


    LHS = np.block([
        [X_T_X,         X_T_Z],
        [Z_T_X, LHS_bottom_right]
    ])

    # Calculate components of RHS
    X_T_y = multiply_matrices(X_T, y_vector)
    Z_T_y = multiply_matrices(Z_T, y_vector)

    RHS = np.vstack([X_T_y, Z_T_y])

    return LHS, RHS


def solve_mme(
    LHS: np.ndarray, RHS: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Solves the Mixed Model Equations.

    Args:
        LHS (np.ndarray): The LHS matrix of the MME.
        RHS (np.ndarray): The RHS vector of the MME.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
            - fixed_effects_solutions (b_hat): Solutions for fixed effects (n_fixed x 1).
            - random_effects_solutions (u_hat): Solutions for random effects (PBVs) (n_animals x 1).
            Returns None, None if LHS is singular or solving fails.
    """
    try:
        solutions = solve_linear_equations(LHS, RHS)
        # Need to split solutions into b_hat and u_hat
        # Determine the split point from the shape of LHS and RHS, or from X_matrix dimensions if passed.
        # Assuming the number of fixed effects corresponds to the number of columns in X'X part of LHS.
        # If LHS = [[X'X, X'Z], [Z'X, Z'Z+A_inv*k]], then num_fixed_effects = X'X.shape[1]
        # This information is not directly available here without X_matrix.
        # However, the number of fixed effects can be inferred if we know how many rows X'y has.
        # Let's assume RHS is vstacked: X'y first, then Z'y.
        # A robust way is to know n_fixed when calling this.
        # For now, this function is generic. The caller needs to know how to split.
        # We will assume the caller splits it. Or, this function should take n_fixed as arg.
        return solutions # Caller will split.
    except np.linalg.LinAlgError:
        # LHS matrix is singular or not invertible
        return None # Or return (None, None) if expecting two outputs for b and u
    except ValueError as e: # For other value errors from solve_linear_equations
        print(f"Error solving MME: {e}")
        return None


# PEV and Accuracy function will be complex.
# PEV and Accuracy function will be complex.
# Let's assume num_fixed_effects is known for splitting LHS_inv.
def calculate_pev_and_accuracy(
    LHS_inv: np.ndarray,
    sigma_e_sq: float,
    sigma_random_effect_sq: float,
    start_idx_block: int,
    end_idx_block: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates Prediction Error Variances (PEV) and accuracies for a block of random effects.

    Args:
        LHS_inv (np.ndarray): The inverse of the full LHS matrix of the MME.
        sigma_e_sq (float): Residual (error) variance.
        sigma_random_effect_sq (float): Variance of the random effect component being evaluated.
                                         (e.g., sigma_a_sq, sigma_sire_sq, sigma_pe_sq). Must be positive.
        start_idx_block (int): Starting row/column index in LHS_inv for this random effect block.
        end_idx_block (int): Ending row/column index (exclusive) in LHS_inv for this block.


    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
            - pev_values (np.ndarray): Vector of PEVs for the random effects (n_effects x 1).
            - accuracy_values (np.ndarray): Vector of accuracies (n_effects x 1).
    """
    if sigma_random_effect_sq <= 0:
        raise ValueError("Variance of random effect (sigma_random_effect_sq) must be positive for accuracy calculation.")

    if LHS_inv is None: # Should not happen if called after successful inversion
        return None, None # Or raise error

    if not (0 <= start_idx_block < end_idx_block <= LHS_inv.shape[0]):
        raise ValueError(f"Invalid block indices [{start_idx_block}:{end_idx_block}] for LHS_inv of shape {LHS_inv.shape}")

    # Extract the relevant block from LHS_inv
    C_random_block = LHS_inv[start_idx_block:end_idx_block, start_idx_block:end_idx_block]

    pev_diagonal = np.diag(C_random_block)
    pev_values = np.maximum(pev_diagonal * sigma_e_sq, 0)

    ratio_pev_var_random = pev_values / sigma_random_effect_sq

    reliability_values = 1 - ratio_pev_var_random
    accuracy_values = np.sqrt(np.maximum(reliability_values, 0))

    return pev_values.reshape(-1,1), accuracy_values.reshape(-1,1)


def solve_animal_model_blup(
    y_vector: np.ndarray,
    X_matrix: np.ndarray,
    Z_matrix: np.ndarray,
    A_inverse_matrix: np.ndarray,
    sigma_e_sq: float,
    sigma_a_sq: float
) -> dict:
    """
    Orchestrates the setup and solution of Mixed Model Equations for an animal model.

    Args:
        y_vector (np.ndarray): Phenotypic observations.
        X_matrix (np.ndarray): Incidence matrix for fixed effects.
        Z_matrix (np.ndarray): Incidence matrix for random animal effects.
        A_inverse_matrix (np.ndarray): Inverse of Numerator Relationship Matrix.
        sigma_e_sq (float): Residual variance.
        sigma_a_sq (float): Additive genetic variance.

    Returns:
        dict: A dictionary containing solutions and related information:
            - 'b_hat': Solutions for fixed effects (or None if failed).
            - 'u_hat': Solutions for random animal effects (PBVs) (or None if failed).
            - 'pev': Prediction Error Variances for u_hat (or None).
            - 'accuracy': Accuracies for u_hat (or None).
            - 'LHS': The LHS matrix (or None if setup failed).
            - 'RHS': The RHS vector (or None if setup failed).
            - 'error': Error message if any step failed.
    """
    results = {
        'b_hat': None, 'u_hat': None,
        'pev': None, 'accuracy': None,
        'LHS': None, 'RHS': None, 'error': None
    }

    try:
        if sigma_a_sq <= 0: # Early exit for a common issue
            results['error'] = "Additive genetic variance (sigma_a_sq) must be positive."
            # u_hat, pev, accuracy will remain None. b_hat could potentially be solved if model simplifies.
            # For simplicity, we stop here. A more advanced version might solve for fixed effects only.
            return results

        LHS, RHS = setup_mme_animal_model(
            y_vector, X_matrix, Z_matrix, A_inverse_matrix, sigma_e_sq, sigma_a_sq
        )
        results['LHS'] = LHS
        results['RHS'] = RHS

        solutions_vector = solve_mme(LHS, RHS)
        if solutions_vector is None:
            results['error'] = "MME solving failed (LHS might be singular)."
            return results

        num_fixed_effects = X_matrix.shape[1]
        results['b_hat'] = solutions_vector[:num_fixed_effects]
        results['u_hat'] = solutions_vector[num_fixed_effects:]

        # Calculate PEV and Accuracy
        try:
            LHS_inv = invert_matrix(LHS) # Invert LHS for PEV calculation
            pev, accuracy = calculate_pev_and_accuracy(
                LHS_inv, sigma_e_sq, sigma_a_sq,
                start_idx_block=num_fixed_effects, # Animal effects start after fixed effects
                end_idx_block=LHS_inv.shape[0]     # Animal effects go to the end of LHS_inv
            )
            results['pev'] = pev
            results['accuracy'] = accuracy
        except np.linalg.LinAlgError:
            results['error'] = "Inverting LHS for PEV calculation failed (LHS might be singular)."
            # Solutions (b_hat, u_hat) might still be valid if solve_mme used a pseudo-inverse or iterative method
            # but our solve_mme currently relies on invert_matrix which would also fail.
            # If solve_mme succeeded (e.g. with np.linalg.solve), LHS_inv might still fail if ill-conditioned.
        except ValueError as e: # Catch specific errors from calculate_pev_and_accuracy
             results['error'] = f"PEV/Accuracy calculation failed: {str(e)}"


    except ValueError as e:
        results['error'] = f"Error in MME setup or solving: {str(e)}"
    except np.linalg.LinAlgError as e:
        results['error'] = f"Linear algebra error (often singular matrix): {str(e)}"
    except Exception as e: # Catch any other unexpected errors
        results['error'] = f"An unexpected error occurred: {str(e)}"

    return results
