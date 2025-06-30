import unittest
import numpy as np
from pyjwas.core.residual_variance import ResidualVariance

class TestResidualVarianceHandler(unittest.TestCase):

    def test_get_or_compute_R_inv_for_pattern_caching(self):
        n_traits = 3
        initial_R = np.array([[4.0, 1.0, 0.5],
                              [1.0, 3.0, 0.2],
                              [0.5, 0.2, 2.0]])

        res_handler = ResidualVariance(r0_matrix=np.copy(initial_R))

        pattern1 = (True, True, False) # Traits 0 and 1 observed

        # First call - should compute
        R_inv_p1_call1 = res_handler.get_or_compute_R_inv_for_pattern(pattern1, initial_R)
        self.assertTrue(pattern1 in res_handler.ri_pattern_inverses)
        cached_R_inv_p1 = res_handler.ri_pattern_inverses[pattern1]
        np.testing.assert_array_almost_equal(R_inv_p1_call1, cached_R_inv_p1)

        # Second call with same R and pattern - should use cache
        # To check if it's cached, we can verify that the object ID is the same if no deepcopy is made on return,
        # or more simply, ensure the r0_full_R_matrix hasn't changed and the dict still has the entry.
        # The function is designed to recompute if r0_full_R_matrix doesn't match current_R_matrix_to_use.
        # So if we pass the same initial_R, it should use cache.
        R_inv_p1_call2 = res_handler.get_or_compute_R_inv_for_pattern(pattern1, initial_R)
        self.assertIs(R_inv_p1_call2, cached_R_inv_p1, "Should return cached object for same R and pattern.")

        # Change the full R matrix
        updated_R = np.array([[5.0, 1.5, 0.8],
                              [1.5, 4.0, 0.3],
                              [0.8, 0.3, 2.5]])

        # Call with new R - should recompute and update cache
        R_inv_p1_call3 = res_handler.get_or_compute_R_inv_for_pattern(pattern1, updated_R)
        self.assertTrue(pattern1 in res_handler.ri_pattern_inverses)
        np.testing.assert_array_almost_equal(res_handler.r0_full_R_matrix, updated_R)

        # The returned matrix should be different from the first call's result
        self.assertFalse(np.allclose(R_inv_p1_call1, R_inv_p1_call3), "Cached R_inv should be different after R matrix changes.")

        # Verify the content of the newly computed R_inv_p1_call3
        observed_indices_p1 = np.where(pattern1)[0]
        R_sub_observed_updated = updated_R[np.ix_(observed_indices_p1, observed_indices_p1)]
        expected_R_sub_inv_updated = np.linalg.inv(R_sub_observed_updated)

        expected_RZ_updated = np.zeros_like(updated_R)
        for i_local, r_global_idx in enumerate(observed_indices_p1):
            for j_local, c_global_idx in enumerate(observed_indices_p1):
                expected_RZ_updated[r_global_idx, c_global_idx] = expected_R_sub_inv_updated[i_local, j_local]
        np.testing.assert_array_almost_equal(R_inv_p1_call3, expected_RZ_updated)


    def test_pattern_all_missing(self):
        n_traits = 2
        initial_R = np.eye(n_traits)
        res_handler = ResidualVariance(r0_matrix=initial_R)
        pattern_all_missing = (False, False)

        R_inv_all_missing = res_handler.get_or_compute_R_inv_for_pattern(pattern_all_missing, initial_R)
        np.testing.assert_array_equal(R_inv_all_missing, np.zeros((n_traits, n_traits)))

    def test_pattern_one_observed_singular_submatrix(self):
        # This test is tricky because a 1x1 matrix is singular if its element is 0.
        # If R_sub_observed is [[0]], inv will fail.
        n_traits = 2
        # R where trait 1 has zero variance (will make R singular, but submatrix for trait 1 is [0])
        initial_R_problem = np.array([[0.0, 0.1], [0.1, 1.0]])
        res_handler = ResidualVariance(r0_matrix=initial_R_problem)
        pattern_trait0_obs = (True, False)

        # Expect a warning about singularity and a zero block returned
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            R_inv_singular = res_handler.get_or_compute_R_inv_for_pattern(pattern_trait0_obs, initial_R_problem)
            self.assertTrue(any("singular" in str(warn.message) for warn in w))

        np.testing.assert_array_equal(R_inv_singular, np.zeros((n_traits, n_traits)))


if __name__ == '__main__':
    unittest.main()
```
