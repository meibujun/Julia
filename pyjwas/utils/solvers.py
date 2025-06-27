import numpy as np
from scipy.sparse import spmatrix # For type hinting

def gibbs_sample_solution_in_place(
    A: Union[np.ndarray, spmatrix],
    x: np.ndarray,
    b: np.ndarray,
    residual_variance: Optional[float] = None
):
    """
    Performs one iteration of a Gibbs sampler for the system Ax = b,
    updating x in place.
    This is equivalent to the single-iteration Gibbs functions in
    JWAS.jl's solver.jl.

    The sampler draws from x_i ~ N(mu_i, var_i), where:
    mu_i = (b_i - sum_{j!=i} A_ij * x_j) / A_ii
    var_i = sigma^2 / A_ii
    where sigma^2 is 1.0 for multi-trait general MME, or `residual_variance`
    for single-trait lambda MME.

    Args:
        A: The coefficient matrix (LHS of MME). Can be dense or sparse.
        x: The solution vector (e.g., mme.sol). Modified in place.
        b: The right-hand side vector of MME.
        residual_variance: The residual variance (sigma^2 or vare).
                           If None (or 1.0), assumes general MME form (variance = 1/A_ii).
                           If provided, assumes lambda MME form (variance = vare/A_ii).
    """
    n_eqs = A.shape[0]
    if x.shape[0] != n_eqs or b.shape[0] != n_eqs:
        raise ValueError("Dimensions of A, x, and b are incompatible.")

    # Ensure b is a 1D array if it's a column vector from MME RHS
    b_flat = b.ravel()

    variance_scale = residual_variance if residual_variance is not None else 1.0

    for i in range(n_eqs):
        A_ii = A[i, i]
        if A_ii == 0.0:
            # This case was handled in Julia by skipping the update.
            # Depending on the context, x[i] might be set to 0 or another value.
            # For a direct translation of the Gibbs step, we skip.
            # However, a zero diagonal in MME usually indicates a problem.
            continue

        # Calculate sum(A_ij * x_j) for j != i
        # This is A[i,:] @ x - A[i,i] * x[i]
        # For sparse A, A[i,:] gives a sparse row vector.
        # For dense A, A[i,:] gives a dense row vector.

        # Note: A[:,i]'x in Julia for symmetric A is equivalent to A[i,:] @ x
        # The term (b[i] - A[:,i]'x)/A[i,i] + x[i] in Julia is equivalent to
        # (b[i] - (A[i,:] @ x - A[i,i]*x[i])) / A[i,i]

        # Efficiently calculate sum_j_neq_i (A_ij * x_j)
        # A_i_row_dot_x = A[i, :] @ x  # This can be slow if A is very large and not CSC/CSR optimized for row access

        # More direct calculation of the sum for j != i:
        # Get the i-th row of A
        if isinstance(A, spmatrix):
            A_i_row = A.getrow(i)
            A_i_row_dot_x = A_i_row.dot(x)[0] # .dot() with vector returns (1,1) matrix for sparse
        else: # Dense numpy array
            A_i_row_dot_x = A[i, :] @ x

        sum_Aij_xj_neq_i = A_i_row_dot_x - A_ii * x[i]

        # Calculate conditional mean (mu_i)
        # mu_i = (b_i - sum_{j!=i} A_ij * x_j) / A_ii
        conditional_mean = (b_flat[i] - sum_Aij_xj_neq_i) / A_ii

        # Calculate conditional variance (var_i)
        # var_i = variance_scale / A_ii
        conditional_variance = variance_scale / A_ii
        if conditional_variance < 0:
            # This can happen if A_ii is negative, which is unusual for MME LHS.
            # Or if variance_scale is negative (error).
            # JWAS.jl's sqrt(invlhs) or sqrt(invlhs*vare) would error on negative.
            # We should handle or raise. For now, let's assume positive.
            # A robust MME should have positive definite A, so A_ii > 0.
            print(f"Warning: Negative conditional variance ({conditional_variance}) at index {i}. A_ii={A_ii}, variance_scale={variance_scale}. Skipping update.")
            continue

        # Sample new x[i]
        x[i] = np.random.randn() * np.sqrt(conditional_variance) + conditional_mean

# For testing purposes
if __name__ == '__main__':
    # Test case 1: Simple dense system (multi-trait like)
    A_dense = np.array([[4.0, 1.0], [1.0, 3.0]])
    x_dense = np.array([0.0, 0.0])
    b_dense = np.array([1.0, 2.0])

    print("Dense system (multi-trait like, residual_variance=None):")
    print(f"Initial x: {x_dense}")
    for k in range(5): # Run a few iterations
        gibbs_sample_solution_in_place(A_dense, x_dense, b_dense)
        print(f"Iter {k+1} x: {x_dense}")
    # Exact solution for Ax=b: x = A_inv @ b
    # A_inv = (1/11) * [[3, -1], [-1, 4]]
    # x_exact = (1/11) * np.array([3*1 - 1*2, -1*1 + 4*2]) = (1/11) * [1, 7] = [0.0909, 0.6363]
    print(f"Exact solution for mean: {np.linalg.inv(A_dense) @ b_dense}")

    # Test case 2: Simple dense system (single-trait like)
    A_st = np.array([[5.0, 2.0], [2.0, 4.0]])
    x_st = np.array([0.0, 0.0])
    b_st = np.array([3.0, 1.0])
    res_var_st = 0.5

    print("\nDense system (single-trait like, residual_variance=0.5):")
    print(f"Initial x: {x_st}")
    # Accumulate samples to check mean
    x_samples_st = []
    for k in range(10000): # More iterations for averaging
        gibbs_sample_solution_in_place(A_st, x_st, b_st, residual_variance=res_var_st)
        if k > 500: # Burn-in for averaging
             x_samples_st.append(x_st.copy())

    mean_sampled_x_st = np.mean(x_samples_st, axis=0)
    print(f"Mean of 9500 samples (after 500 burn-in): {mean_sampled_x_st}")
    # Exact solution for mean: x = A_inv @ b
    # A_inv_st = (1/16) * [[4, -2], [-2, 5]]
    # x_exact_st = (1/16) * np.array([4*3 - 2*1, -2*3 + 5*1]) = (1/16) * [10, -1] = [0.625, -0.0625]
    print(f"Exact solution for mean: {np.linalg.inv(A_st) @ b_st}")

    # Test case 3: Sparse system
    from scipy.sparse import csc_matrix
    A_sparse_data = np.array([10.0, -2.0, -2.0, 8.0, 1.0, 1.0, 5.0])
    A_sparse_indices = np.array([0, 1, 0, 1, 2, 1, 2]) # row indices
    A_sparse_indptr = np.array([0, 2, 5, 7])      # col pointers
    A_sparse = csc_matrix((A_sparse_data, A_sparse_indices, A_sparse_indptr), shape=(3,3))
    # A = [[10, -2,  0],
    #      [-2,  8,  1],
    #      [ 0,  1,  5]]
    x_sparse = np.array([0.0, 0.0, 0.0])
    b_sparse = np.array([1.0, 0.5, 2.0])
    print("\nSparse system (multi-trait like):")
    print(f"A sparse:\n{A_sparse.toarray()}")
    print(f"Initial x: {x_sparse}")
    for k in range(5):
        gibbs_sample_solution_in_place(A_sparse, x_sparse, b_sparse)
        print(f"Iter {k+1} x: {x_sparse}")
    from scipy.sparse.linalg import inv
    # print(f"Exact solution for mean: {inv(A_sparse) @ b_sparse}") # inv for sparse can be tricky
    print(f"Exact solution for mean (using dense inv): {np.linalg.inv(A_sparse.toarray()) @ b_sparse}")

    # Test with zero diagonal (should skip update for that element)
    A_zero_diag = np.array([[4.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 3.0]])
    x_zero_diag = np.array([0.1, 0.2, 0.3]) # Start with non-zero to see if x[1] changes
    b_zero_diag = np.array([1.0, 2.0, 3.0])
    print("\nSystem with zero diagonal:")
    print(f"Initial x: {x_zero_diag}")
    gibbs_sample_solution_in_place(A_zero_diag, x_zero_diag, b_zero_diag)
    print(f"After 1 iter x: {x_zero_diag}") # x[1] should remain 0.2
    self.assertTrue(x_zero_diag[1] == 0.2, "Element corresponding to zero diagonal should not change.")

    # Test with negative conditional variance (e.g. A_ii < 0 and residual_variance > 0)
    A_neg_diag = np.array([[-4.0, 1.0], [1.0, 3.0]])
    x_neg_diag = np.array([0.1, 0.2])
    b_neg_diag = np.array([1.0, 2.0])
    print("\nSystem with negative diagonal (A_ii < 0):")
    print(f"Initial x: {x_neg_diag}")
    gibbs_sample_solution_in_place(A_neg_diag, x_neg_diag, b_neg_diag, residual_variance=1.0)
    print(f"After 1 iter x: {x_neg_diag}") # x[0] should remain 0.1 due to warning and skip

    print("Done with solver tests.")

# The following would typically be in a test file, but included here for quick check
class TestGibbsSampler(unittest.TestCase):
    def test_zero_diag_skip(self):
        A_zero_diag = np.array([[4.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 3.0]])
        x_initial = np.array([0.1, 0.2, 0.3])
        x_to_update = x_initial.copy()
        b_zero_diag = np.array([1.0, 2.0, 3.0])
        gibbs_sample_solution_in_place(A_zero_diag, x_to_update, b_zero_diag)
        self.assertEqual(x_to_update[1], x_initial[1], "Element for zero diagonal should not change.")

    def test_neg_variance_skip(self):
        A_neg_diag = np.array([[-4.0, 1.0], [1.0, 3.0]])
        x_initial = np.array([0.1, 0.2])
        x_to_update = x_initial.copy()
        b_neg_diag = np.array([1.0, 2.0])
        gibbs_sample_solution_in_place(A_neg_diag, x_to_update, b_neg_diag, residual_variance=1.0)
        self.assertEqual(x_to_update[0], x_initial[0], "Element for negative conditional variance should not change.")

# To run these tests if this file is executed:
# import unittest
# TestGibbsSampler.self = TestGibbsSampler() # Hack to run the example in __main__
# A_zero_diag = np.array([[4.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 3.0]])
# x_zero_diag = np.array([0.1, 0.2, 0.3])
# b_zero_diag = np.array([1.0, 2.0, 3.0])
# gibbs_sample_solution_in_place(A_zero_diag, x_zero_diag, b_zero_diag)
# TestGibbsSampler.self.assertTrue(x_zero_diag[1] == 0.2, "Element corresponding to zero diagonal should not change.")
#
# A_neg_diag = np.array([[-4.0, 1.0], [1.0, 3.0]])
# x_neg_diag = np.array([0.1, 0.2])
# b_neg_diag = np.array([1.0, 2.0])
# gibbs_sample_solution_in_place(A_neg_diag, x_neg_diag, b_neg_diag, residual_variance=1.0)
# TestGibbsSampler.self.assertTrue(x_neg_diag[0] == 0.1, "Element for negative conditional variance should not change.")

# The unittest part is better in a separate test file.
# For now, the __main__ block provides runnable examples.
# Need to import unittest at top level for the class TestGibbsSampler to be defined.
import unittest # Ensure unittest is imported if class TestGibbsSampler is defined.

# Removing the ad-hoc self assignment for TestGibbsSampler in main.
# The class definition is fine, but running its tests from __main__ here is messy.
# The print statements in __main__ serve as basic procedural checks.
# Proper tests should be in tests/utils/test_solvers.py
