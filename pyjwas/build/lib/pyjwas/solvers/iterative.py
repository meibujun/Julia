from typing import List, Dict, Any, Union, Optional, Callable
import numpy as np
from scipy.sparse import issparse, dia_matrix, csc_matrix, csr_matrix

from pyjwas.core.definitions import MME, DefaultFloat, Variance, MCMCInfo # Added Variance, MCMCInfo
# Functions like check_pedigree_genotypes_phenotypes, set_default_priors_for_variance_components, get_mme_components
# are expected to be available from other modules (e.g., model_builder, validation) when solve() is called.
# For now, we'll assume they are imported or mme object is pre-configured.
# from pyjwas.core.model_builder import get_mme_components # Example
# from pyjwas.io.validation import check_pedigree_genotypes_phenotypes # Example
# from pyjwas.core.components import set_default_priors_for_variance_components # Example
# from pyjwas.core.utils import get_effect_names # Example

# --- Iterative Solvers ---

def solve_mme_jacobi(
    A: Union[np.ndarray, csr_matrix, csc_matrix], # LHS matrix
    x_initial: np.ndarray,      # Initial solution vector
    b: np.ndarray,              # RHS vector
    p_relaxation: float = 0.7,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
    printout_frequency: int = 100
) -> np.ndarray:
    """
    Solves Ax = b using the Jacobi iteration method with relaxation.
    Corresponds to Julia's Jacobi.
    """
    x = x_initial.copy().astype(A.dtype) # Ensure x has same dtype as A for operations
    n_equations = A.shape[0]

    if A.shape[0] != A.shape[1] or A.shape[0] != len(b) or A.shape[0] != len(x):
        raise ValueError("Dimension mismatch in Jacobi solver inputs.")

    diagonal_A = A.diagonal()
    if np.any(diagonal_A == 0):
        print("Warning: Zero diagonal elements in A for Jacobi. May lead to division by zero or instability.")
        diagonal_A[diagonal_A == 0] = 1e-12

    error = b - A @ x
    current_diff = np.sum(error**2) / n_equations

    print(f"Jacobi starting. Initial diff: {current_diff:.8g}, tolerance: {tolerance:.2g}")

    for iteration in range(max_iterations):
        error = b - A @ x

        x_temp = error / diagonal_A + x
        x = p_relaxation * x_temp + (1.0 - p_relaxation) * x

        error_after_update = b - A @ x
        current_diff = np.sum(error_after_update**2) / n_equations

        if (iteration + 1) % printout_frequency == 0:
            print(f"Jacobi iteration {iteration + 1}: diff = {current_diff:.8g}")

        if current_diff < tolerance:
            print(f"Jacobi converged after {iteration + 1} iterations. Final diff: {current_diff:.8g}")
            return x

    print(f"Jacobi did not converge after {max_iterations} iterations. Final diff: {current_diff:.8g}")
    return x


def solve_mme_gauss_seidel(
    A: Union[np.ndarray, csr_matrix, csc_matrix], # LHS matrix
    x_initial: np.ndarray,      # Initial solution vector
    b: np.ndarray,              # RHS vector
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
    printout_frequency: int = 100
) -> np.ndarray:
    """
    Solves Ax = b using the Gauss-Seidel iteration method.
    Corresponds to Julia's GaussSeidel.
    """
    x = x_initial.copy().astype(A.dtype)
    n_equations = A.shape[0]

    if A.shape[0] != A.shape[1] or A.shape[0] != len(b) or A.shape[0] != len(x):
        raise ValueError("Dimension mismatch in Gauss-Seidel solver inputs.")

    for i in range(n_equations): # Initial update pass
        A_ii = A[i,i]
        if A_ii == 0:
            print(f"Warning: Zero diagonal A[{i},{i}] in Gauss-Seidel. Skipping update for x[{i}].")
            continue
        sigma = A[i, :] @ x - A[i,i] * x[i]
        # Ensure result is scalar for assignment to x[i]
        val_init_pass = (b[i] - sigma) / A_ii
        x[i] = val_init_pass.item() if hasattr(val_init_pass, 'item') else val_init_pass

    error = b - A @ x
    current_diff = np.sum(error**2) / n_equations
    print(f"Gauss-Seidel starting. Initial diff after first pass: {current_diff:.8g}, tolerance: {tolerance:.2g}")

    for iteration in range(max_iterations):
        for i in range(n_equations):
            A_ii = A[i,i]
            if A_ii == 0: continue

            sum_ax_updated_part = A[i, :i] @ x[:i] if i > 0 else 0.0
            sum_ax_old_part = A[i, i+1:] @ x[i+1:] if i < n_equations - 1 else 0.0
            # Ensure result is scalar for assignment to x[i]
            val_iter_pass = (b[i] - sum_ax_updated_part - sum_ax_old_part) / A_ii
            x[i] = val_iter_pass.item() if hasattr(val_iter_pass, 'item') else val_iter_pass

        error = b - A @ x
        current_diff = np.sum(error**2) / n_equations

        if (iteration + 1) % printout_frequency == 0:
            print(f"Gauss-Seidel iteration {iteration + 1}: diff = {current_diff:.8g}")

        if current_diff < tolerance:
            print(f"Gauss-Seidel converged after {iteration + 1} iterations. Final diff: {current_diff:.8g}")
            return x

    print(f"Gauss-Seidel did not converge after {max_iterations} iterations. Final diff: {current_diff:.8g}")
    return x


def solve_mme_gibbs_sampler(
    A: Union[np.ndarray, csr_matrix, csc_matrix], # LHS matrix
    x_initial: np.ndarray,      # Initial solution vector
    b: np.ndarray,              # RHS vector
    n_iterations: int,
    residual_variance: Optional[float] = None,
    printout_frequency: int = 100
) -> np.ndarray:
    """
    Solves Ax = b using a Gibbs sampler approach.
    Corresponds to Julia's Gibbs functions.
    Calculates and returns the mean of the samples for x.
    """
    x = x_initial.copy().astype(A.dtype)
    n_equations = A.shape[0]
    x_mean = np.zeros_like(x)

    if A.shape[0] != A.shape[1] or A.shape[0] != len(b) or A.shape[0] != len(x):
        raise ValueError("Dimension mismatch in Gibbs sampler inputs.")

    print(f"Gibbs Sampler starting for {n_iterations} iterations.")

    for iteration in range(n_iterations):
        if (iteration + 1) % printout_frequency == 0:
            print(f"Gibbs iteration {iteration + 1}/{n_iterations}")

        for i in range(n_equations):
            A_ii = A[i,i]
            if A_ii == 0.0:
                continue

            inv_lhs_diag = 1.0 / A_ii

            conditional_mean_numerator = b[i] - (A[i, :i] @ x[:i] if i > 0 else 0.0) \
                                         - (A[i, i+1:] @ x[i+1:] if i < n_equations - 1 else 0.0)
            conditional_mean = conditional_mean_numerator * inv_lhs_diag

            conditional_variance_scalar: float
            if residual_variance is not None:
                conditional_variance_scalar = inv_lhs_diag * residual_variance
            else:
                conditional_variance_scalar = inv_lhs_diag

            if conditional_variance_scalar < 0:
                print(f"Warning: Negative conditional variance ({conditional_variance_scalar:.3g}) for x[{i}]. Using small positive.")
                conditional_variance_scalar = 1e-12

            x[i] = np.random.normal(loc=conditional_mean, scale=np.sqrt(conditional_variance_scalar))

        x_mean += (x - x_mean) / (iteration + 1)

    print(f"Gibbs Sampler finished. Returning mean of {n_iterations} samples.")
    return x_mean


def solve_mme_gibbs_one_iteration(
    A: Union[np.ndarray, csr_matrix, csc_matrix],
    x: np.ndarray, # Modified in-place
    b: np.ndarray,
    residual_variance: Optional[float] = None
) -> None:
    """
    Performs one iteration of Gibbs sampling for MME. Modifies x in-place.
    Corresponds to Julia's single-iteration Gibbs functions.
    """
    n_equations = A.shape[0]
    for i in range(n_equations):
        A_ii = A[i,i]
        if A_ii == 0.0: continue

        inv_lhs_diag = 1.0 / A_ii

        conditional_mean_numerator = b[i] - (A[i, :i] @ x[:i] if i > 0 else 0.0) \
                                         - (A[i, i+1:] @ x[i+1:] if i < n_equations - 1 else 0.0)
        conditional_mean = conditional_mean_numerator * inv_lhs_diag

        conditional_variance_scalar: float
        if residual_variance is not None:
            conditional_variance_scalar = inv_lhs_diag * residual_variance
        else:
            conditional_variance_scalar = inv_lhs_diag

        if conditional_variance_scalar < 0:
            conditional_variance_scalar = 1e-12

        x[i] = np.random.normal(loc=conditional_mean, scale=np.sqrt(conditional_variance_scalar))

def solve_mme_system(
    mme: MME,
    solver_type: str = "Gauss-Seidel",
    p_relaxation: float = 0.7,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
    printout_frequency: int = 100
) -> Optional[np.ndarray]:
    """
    Solves the MME system using a specified iterative solver.
    """
    if mme.mme_lhs is None or mme.mme_rhs is None:
        print("Error: MME LHS or RHS not initialized. Cannot solve.")
        return None

    A = mme.mme_lhs
    b_sparse = mme.mme_rhs
    b = b_sparse.toarray().ravel() if issparse(b_sparse) else np.array(b_sparse).ravel()

    precision = DefaultFloat
    if mme.mcmc_info and not mme.mcmc_info.double_precision:
        precision = np.float32

    A = A.astype(precision) # type: ignore
    b = b.astype(precision)
    x_initial = np.zeros(A.shape[0], dtype=precision)

    solution: Optional[np.ndarray] = None
    if solver_type.lower() == "jacobi":
        solution = solve_mme_jacobi(A, x_initial, b, p_relaxation, tolerance, max_iterations, printout_frequency)
    elif solver_type.lower() == "gauss-seidel":
        solution = solve_mme_gauss_seidel(A, x_initial, b, tolerance, max_iterations, printout_frequency)
    elif solver_type.lower() == "gibbs":
        res_var_val: Optional[float] = None
        if mme.n_models == 1 and mme.R and mme.R.val is not False:
            try:
                res_var_val = float(np.array(mme.R.val).item())
                if res_var_val <=0:
                     print("Warning: Non-positive mme.R.val for single-trait Gibbs. Using default 1.0.")
                     res_var_val = 1.0
            except (ValueError, TypeError):
                 print("Warning: mme.R.val not a valid scalar for single-trait Gibbs. Using default 1.0.")
                 res_var_val = 1.0
        elif mme.n_models > 1:
            print("Warning: Gibbs solver called for multi-trait model without specific multi-trait residual variance handling for sampling. Using effective res_var=1.0.")
            res_var_val = 1.0 # Or None, if Gibbs sampler should adapt. For now, assume 1.0 for structure.

        solution = solve_mme_gibbs_sampler(A, x_initial, b, max_iterations, res_var_val, printout_frequency)
    else:
        print(f"Error: Unknown solver_type '{solver_type}'. Choose from Jacobi, Gauss-Seidel, Gibbs.")
        return None

    if solution is not None:
        mme.sol = solution
    return solution


if __name__ == '__main__':
    print("--- Iterative Solvers Examples ---")

    A_test = np.array([[10, -1, 2, 0],
                       [-1, 11, -1, 3],
                       [2, -1, 10, -1],
                       [0, 3, -1, 8]], dtype=DefaultFloat)
    b_test = np.array([6, 25, -11, 15], dtype=DefaultFloat)
    x_init_test = np.zeros(A_test.shape[0], dtype=DefaultFloat)

    print(f"System Matrix A:\n{A_test}")
    print(f"RHS b: {b_test}")

    x_direct: Optional[np.ndarray] = None
    try:
        x_direct = np.linalg.solve(A_test, b_test)
        print(f"Direct solution x: {x_direct}")
    except np.linalg.LinAlgError:
        print("Matrix A_test is singular for direct solve.")

    print("\nTesting Jacobi Solver...")
    x_jacobi = solve_mme_jacobi(A_test.copy(), x_init_test.copy(), b_test.copy(), max_iterations=100, printout_frequency=20)
    print(f"Jacobi solution: {x_jacobi}")
    if x_direct is not None: print(f"Difference from direct: {np.linalg.norm(x_jacobi - x_direct):.3g}")

    print("\nTesting Gauss-Seidel Solver...")
    x_gs = solve_mme_gauss_seidel(A_test.copy(), x_init_test.copy(), b_test.copy(), max_iterations=100, printout_frequency=20)
    print(f"Gauss-Seidel solution: {x_gs}")
    if x_direct is not None: print(f"Difference from direct: {np.linalg.norm(x_gs - x_direct):.3g}")

    print("\nTesting Gibbs Sampler (as solver, returning mean)...")
    x_gibbs_mean = solve_mme_gibbs_sampler(A_test.copy(), x_init_test.copy(), b_test.copy(),
                                           n_iterations=5000, residual_variance=1.0, printout_frequency=1000)
    print(f"Gibbs sampler mean solution: {x_gibbs_mean}")
    if x_direct is not None: print(f"Difference from direct: {np.linalg.norm(x_gibbs_mean - x_direct):.3g}")

    print("\nTesting Gibbs Sampler (one iteration, modifies x in place)...")
    x_one_iter = x_init_test.copy()
    solve_mme_gibbs_one_iteration(A_test.copy(), x_one_iter, b_test.copy(), residual_variance=1.0)
    print(f"Solution after one Gibbs iteration: {x_one_iter}")

    print("\nTesting solve_mme_system dispatcher...")
    dummy_mme = MME(n_models=1, model_vec=["y=X"], model_terms=[], model_term_dict={}, lhs_vec=["y"]) # Added dummy model_vec etc.
    dummy_mme.mme_lhs = A_test.copy()
    dummy_mme.mme_rhs = b_test.copy()
    dummy_mme.R = Variance(val=1.0)
    dummy_mme.mcmc_info = MCMCInfo(double_precision=True)

    sol_dispatch_gs = solve_mme_system(dummy_mme, solver_type="Gauss-Seidel", max_iterations=100, printout_frequency=20)
    print(f"Solution from dispatcher (Gauss-Seidel): {sol_dispatch_gs}")
    if sol_dispatch_gs is not None and x_direct is not None:
        print(f"Difference from direct: {np.linalg.norm(sol_dispatch_gs - x_direct):.3g}")

    A_sparse_test = csc_matrix(A_test)
    dummy_mme_sparse = MME(n_models=1, model_vec=["y=X"], model_terms=[], model_term_dict={}, lhs_vec=["y"]) # Added dummy model_vec etc.
    dummy_mme_sparse.mme_lhs = A_sparse_test
    dummy_mme_sparse.mme_rhs = b_test.copy()
    dummy_mme_sparse.R = Variance(val=1.0)
    dummy_mme_sparse.mcmc_info = MCMCInfo(double_precision=True)

    sol_dispatch_gs_sparse = solve_mme_system(dummy_mme_sparse, solver_type="Gauss-Seidel", max_iterations=100, printout_frequency=20)
    print(f"Solution from dispatcher (Gauss-Seidel with sparse A): {sol_dispatch_gs_sparse}")
    if sol_dispatch_gs_sparse is not None and x_direct is not None:
        print(f"Difference from sparse direct: {np.linalg.norm(sol_dispatch_gs_sparse - x_direct):.3g}")
