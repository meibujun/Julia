import unittest
import numpy as np
from scipy.sparse import csc_matrix
from pyjwas.utils import gibbs_sample_solution_in_place # Assuming __init__.py exports it

class TestGibbsSamplerLinearSystem(unittest.TestCase):

    def setUp(self):
        # Common setup for multiple tests if needed
        np.random.seed(12345) # for reproducibility of sampling

    def test_dense_system_multitrait_like(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        x = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])

        # Expected mean (A_inv @ b)
        x_expected_mean = np.linalg.inv(A) @ b

        n_samples = 10000
        burn_in = 1000
        samples = []
        for i in range(n_samples):
            gibbs_sample_solution_in_place(A, x, b, residual_variance=None) # or 1.0
            if i >= burn_in:
                samples.append(x.copy())

        mean_x = np.mean(samples, axis=0)
        # Check if the sampled mean is close to the expected mean
        np.testing.assert_allclose(mean_x, x_expected_mean, rtol=0.1, atol=0.1)

    def test_dense_system_singletrait_like(self):
        A = np.array([[5.0, 2.0], [2.0, 4.0]])
        x = np.array([0.0, 0.0])
        b = np.array([3.0, 1.0])
        res_var = 0.5

        x_expected_mean = np.linalg.inv(A) @ b

        n_samples = 10000
        burn_in = 1000
        samples = []
        for i in range(n_samples):
            gibbs_sample_solution_in_place(A, x, b, residual_variance=res_var)
            if i >= burn_in:
                samples.append(x.copy())

        mean_x = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean_x, x_expected_mean, rtol=0.1, atol=0.1)

    def test_sparse_system(self):
        A_data = np.array([10.0, -2.0, -2.0, 8.0, 1.0, 1.0, 5.0])
        A_indices = np.array([0, 1, 0, 1, 2, 1, 2])
        A_indptr = np.array([0, 2, 5, 7])
        A_sparse = csc_matrix((A_data, A_indices, A_indptr), shape=(3,3))

        x = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.5, 2.0])

        x_expected_mean = np.linalg.inv(A_sparse.toarray()) @ b

        n_samples = 10000
        burn_in = 1000
        samples = []
        for i in range(n_samples):
            gibbs_sample_solution_in_place(A_sparse, x, b)
            if i >= burn_in:
                samples.append(x.copy())

        mean_x = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean_x, x_expected_mean, rtol=0.1, atol=0.1)

    def test_zero_diagonal_element(self):
        A = np.array([[4.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 3.0]])
        x_initial = np.array([0.1, 0.2, 0.3]) # Element x[1] corresponds to zero diagonal
        x = x_initial.copy()
        b = np.array([1.0, 2.0, 3.0])

        gibbs_sample_solution_in_place(A, x, b)
        # The element x[1] should not have changed from its initial value
        self.assertEqual(x[1], x_initial[1])
        # Other elements should change
        self.assertNotEqual(x[0], x_initial[0])
        self.assertNotEqual(x[2], x_initial[2])

    def test_negative_conditional_variance_warning(self):
        # A_ii < 0, residual_variance > 0 should lead to var < 0
        A = np.array([[-4.0, 1.0], [1.0, 3.0]])
        x_initial = np.array([0.5, 0.5])
        x = x_initial.copy()
        b = np.array([1.0, 1.0])

        # Expect a print warning, and x[0] should not change
        # Suppress print for test cleanliness if possible, or just check behavior
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output # Redirect stdout

        gibbs_sample_solution_in_place(A, x, b, residual_variance=1.0)

        sys.stdout = sys.__stdout__ # Reset redirect
        self.assertIn("Warning: Negative conditional variance", captured_output.getvalue())
        self.assertEqual(x[0], x_initial[0]) # x[0] should not be updated
        self.assertNotEqual(x[1], x_initial[1]) # x[1] should be updated


    def test_incompatible_dimensions(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        x_wrong = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0])
        with self.assertRaises(ValueError):
            gibbs_sample_solution_in_place(A, x_wrong, b)

        x = np.array([0.0,0.0])
        b_wrong = np.array([1.0])
        with self.assertRaises(ValueError):
            gibbs_sample_solution_in_place(A,x,b_wrong)


if __name__ == '__main__':
    unittest.main()
