import unittest
import numpy as np
from pyjwas.utils.samplers import sample_scalar_variance_component, sample_matrix_variance_component
from pyjwas.utils.distributions import sample_scaled_inverse_chi_squared # For comparison logic

class TestVarianceSamplers(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.n_test_samples = 5000 # Number of samples to draw for checking means

    def test_sample_scalar_variance_component_basic(self):
        # Data N(0, true_var=2.0)
        true_var = 2.0
        n_obs = 100
        data = np.random.randn(n_obs) * np.sqrt(true_var)

        prior_df = 4.0
        prior_scale = 1.0 # S0^2 parameter for ScaledInvChi2 prior

        samples = [sample_scalar_variance_component(data, prior_df, prior_scale) for _ in range(self.n_test_samples)]
        mean_sampled_var = np.mean(samples)

        # Expected posterior: ScaledInvChi2(nu_post, S2_post)
        # nu_post = prior_df + n_obs = 4 + 100 = 104
        # S2_post = (dot(data,data) + prior_df*prior_scale) / nu_post
        # Mean of ScaledInvChi2(nu, S2) is S2 * nu / (nu - 2) if nu > 2, else S2 if nu large.
        # Or, more directly, mean is E[SSE_post / X] where X ~ Chi2(nu_post).
        # E[1/X] for X ~ Chi2(k) is 1/(k-2) for k>2. So mean approx SSE_post / (nu_post - 2)
        sse_post = np.dot(data, data) + prior_df * prior_scale
        nu_post = prior_df + n_obs
        expected_mean_approx = sse_post / (nu_post - 2) if nu_post > 2 else sse_post / nu_post

        # Check if sampled mean is in a reasonable range around true_var and expected_mean_approx
        self.assertGreater(mean_sampled_var, 0)
        self.assertLess(abs(mean_sampled_var - true_var) / true_var, 0.5, "Sampled mean too far from true variance")
        self.assertLess(abs(mean_sampled_var - expected_mean_approx) / expected_mean_approx, 0.2,
                        "Sampled mean too far from approximate expected posterior mean")

    def test_sample_scalar_variance_component_weighted(self):
        n_obs = 100
        true_common_var = 1.0 # sigma^2 in y_i ~ N(0, sigma^2 / w_i)
        # Variances for observations are 2.0 and 0.5. So w_i are 0.5 and 2.0.
        obs_variances = np.repeat([2.0, 0.5], n_obs // 2)
        inv_weights = 1.0 / obs_variances # These are the w_i

        data = np.random.randn(n_obs) * np.sqrt(obs_variances) # y_i ~ N(0, obs_variances_i)
                                                              # y_i ~ N(0, (true_common_var / inv_weights_i))

        prior_df = 4.0
        prior_scale = 0.5

        samples = [sample_scalar_variance_component(data, prior_df, prior_scale, inv_weights=inv_weights) for _ in range(self.n_test_samples)]
        mean_sampled_var = np.mean(samples)

        # Weighted sum of squares: sum ( (data_i * sqrt(inv_weight_i))^2 )
        # = sum ( data_i^2 * inv_weight_i )
        weighted_sum_sq = np.sum((data**2) * inv_weights)
        sse_post = weighted_sum_sq + prior_df * prior_scale
        nu_post = prior_df + n_obs
        expected_mean_approx = sse_post / (nu_post - 2) if nu_post > 2 else sse_post / nu_post

        self.assertGreater(mean_sampled_var, 0)
        # Expect mean_sampled_var to be around true_common_var (1.0)
        self.assertLess(abs(mean_sampled_var - true_common_var) / true_common_var, 0.5, "Weighted sampled mean too far from true common variance")
        self.assertLess(abs(mean_sampled_var - expected_mean_approx) / expected_mean_approx, 0.2,
                        "Weighted sampled mean too far from approximate expected posterior mean")


    def test_sample_matrix_variance_component_unconstrained(self):
        n_obs = 200
        n_traits = 2
        true_R = np.array([[2.0, 0.5], [0.5, 1.0]])
        L = np.linalg.cholesky(true_R)
        # Generate data: res_multi is (n_obs x n_traits)
        res_multi_raw = (L @ np.random.randn(n_traits, n_obs)).T
        data_arrays = [res_multi_raw[:, i] for i in range(n_traits)]

        prior_df = float(n_traits + 3) # nu_0, needs to be > n_traits - 1 for IW
        prior_scale_matrix = np.eye(n_traits) * 1.0 # Psi_0

        samples = [sample_matrix_variance_component(data_arrays, n_obs, prior_df, prior_scale_matrix) for _ in range(self.n_test_samples)]
        mean_R_sampled = np.mean(samples, axis=0)

        # Expected posterior mean for IW(nu_post, Psi_post) is Psi_post / (nu_post - p - 1)
        SSCP_data = np.cov(res_multi_raw, rowvar=False) * (n_obs -1) # More direct SSCP from data
        # Or calculate as in function:
        _SSCP_data = np.zeros((n_traits, n_traits))
        for i in range(n_traits):
            for j in range(i, n_traits):
                dp = np.dot(data_arrays[i], data_arrays[j])
                _SSCP_data[i,j] = dp
                if i!=j: _SSCP_data[j,i] = dp

        Psi_post = prior_scale_matrix + _SSCP_data
        nu_post = prior_df + n_obs
        expected_mean_R = Psi_post / (nu_post - n_traits - 1) if (nu_post - n_traits - 1) > 0 else Psi_post # Approx

        self.assertTrue(np.all(np.linalg.eigvals(mean_R_sampled) > 0), "Mean sampled R not PD")
        np.testing.assert_allclose(mean_R_sampled, expected_mean_R, rtol=0.2, atol=0.2, err_msg="Matrix variance mean deviates significantly from expected posterior mean.")


    def test_sample_matrix_variance_component_constrained(self):
        n_obs = 200
        n_traits = 2
        # True variances are diagonal
        true_R_diag = np.array([2.0, 1.0])
        res_multi_raw = np.random.randn(n_obs, n_traits) * np.sqrt(true_R_diag)
        data_arrays = [res_multi_raw[:, i] for i in range(n_traits)]

        prior_df = 4.0 # This df is used for each scalar variance sampling in Julia's code
        prior_scale_matrix_diag = np.diag([1.0, 0.8]) # Psi_0, diagonal elements are S0_i^2

        samples = [sample_matrix_variance_component(data_arrays, n_obs, prior_df, prior_scale_matrix_diag, constraint=True) for _ in range(self.n_test_samples)]
        mean_R_sampled = np.mean(samples, axis=0)

        self.assertTrue(np.allclose(mean_R_sampled - np.diag(np.diag(mean_R_sampled)), 0),
                        "Constrained sampled matrix mean is not diagonal.")

        # Check diagonal elements against expected scalar posterior means
        for i in range(n_traits):
            data_i = data_arrays[i]
            prior_scale_i = prior_scale_matrix_diag[i,i]
            sse_post_i = np.dot(data_i, data_i) + prior_df * prior_scale_i
            nu_post_i = prior_df + n_obs
            expected_mean_i = sse_post_i / (nu_post_i - 2) if nu_post_i > 2 else sse_post_i / nu_post_i
            self.assertLess(abs(mean_R_sampled[i,i] - expected_mean_i) / expected_mean_i, 0.2,
                            f"Diagonal element {i} for constrained matrix variance deviates from expected.")

    def test_error_conditions_scalar(self):
        data = np.random.randn(10)
        with self.assertRaises(ValueError): # n=0, prior_df=0
            sample_scalar_variance_component(np.array([]), 0, 0.1)
        with self.assertRaises(ValueError): # inv_weights length mismatch
             sample_scalar_variance_component(data, 4.0, 1.0, inv_weights=np.array([1.0]))
        with self.assertRaises(ValueError): # posterior_df <= 0
             sample_scalar_variance_component(np.array([]), -1, 0.1)


    def test_error_conditions_matrix(self):
        data = [np.random.randn(10), np.random.randn(10)]
        with self.assertRaises(ValueError): # empty data_arrays
            sample_matrix_variance_component([], 10, 4.0, np.eye(1))
        with self.assertRaises(ValueError): # n_obs mismatch
            sample_matrix_variance_component([np.random.randn(10), np.random.randn(5)], 10, 4.0, np.eye(2))
        with self.assertRaises(ValueError): # prior_scale_matrix shape
            sample_matrix_variance_component(data, 10, 4.0, np.eye(3))

if __name__ == '__main__':
    unittest.main()
