import unittest
import numpy as np
from scipy.stats import kstest, gamma as scipy_gamma, invgamma as scipy_invgamma, invwishart as scipy_invwishart
from pyjwas.utils.distributions import (
    sample_gamma,
    sample_inverse_gamma,
    sample_scaled_inverse_chi_squared,
    sample_inverse_wishart
)

class TestDistributionSamplers(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples_kstest = 2000 # Number of samples for KS test

    def test_sample_gamma(self):
        shape, scale_theta = 2.0, 3.0
        samples = sample_gamma(shape, scale_theta, size=self.n_samples_kstest)
        self.assertEqual(samples.shape, (self.n_samples_kstest,))

        # KS test against scipy.stats.gamma
        # Scipy's scale is theta. Location is 0 by default.
        D, p_value = kstest(samples, lambda x: scipy_gamma.cdf(x, a=shape, scale=scale_theta))
        self.assertGreater(p_value, 0.01, f"Gamma KS test failed: p-value={p_value}")

        with self.assertRaises(ValueError): sample_gamma(0, 1)
        with self.assertRaises(ValueError): sample_gamma(1, 0)

        single_sample = sample_gamma(shape,scale_theta)
        self.assertIsInstance(single_sample, float)


    def test_sample_inverse_gamma(self):
        shape, scale_param = 3.0, 2.0 # alpha, beta
        samples = sample_inverse_gamma(shape, scale_param, size=self.n_samples_kstest)
        self.assertEqual(samples.shape, (self.n_samples_kstest,))

        # KS test against scipy.stats.invgamma
        # Scipy's a is shape, scale is scale_param. Location is 0.
        D, p_value = kstest(samples, lambda x: scipy_invgamma.cdf(x, a=shape, scale=scale_param))
        self.assertGreater(p_value, 0.01, f"Inverse Gamma KS test failed: p-value={p_value}")

        with self.assertRaises(ValueError): sample_inverse_gamma(0, 1)
        with self.assertRaises(ValueError): sample_inverse_gamma(1, 0)

        single_sample = sample_inverse_gamma(shape,scale_param)
        self.assertIsInstance(single_sample, float)

    def test_sample_scaled_inverse_chi_squared(self):
        df, scale_sq = 4.0, 1.5
        samples = sample_scaled_inverse_chi_squared(df, scale_sq, size=self.n_samples_kstest)
        self.assertEqual(samples.shape, (self.n_samples_kstest,))

        # Equivalent InvGamma parameters:
        ig_shape = df / 2.0
        ig_scale = (df * scale_sq) / 2.0
        D, p_value = kstest(samples, lambda x: scipy_invgamma.cdf(x, a=ig_shape, scale=ig_scale))
        self.assertGreater(p_value, 0.01, f"Scaled Inv Chi2 KS test failed: p-value={p_value}")

        with self.assertRaises(ValueError): sample_scaled_inverse_chi_squared(0, 1)
        with self.assertRaises(ValueError): sample_scaled_inverse_chi_squared(1, 0)

        single_sample = sample_scaled_inverse_chi_squared(df,scale_sq)
        self.assertIsInstance(single_sample, float)


    def test_sample_inverse_wishart(self):
        p_dim = 3
        df = float(p_dim + 4) # df must be > p - 1
        scale_matrix = np.array([[2.0, 0.5, 0.2],
                                 [0.5, 1.5, 0.3],
                                 [0.2, 0.3, 1.0]])
        # Ensure PD for test
        scale_matrix = scale_matrix @ scale_matrix.T

        # Test single sample
        single_sample = sample_inverse_wishart(df, scale_matrix)
        self.assertIsInstance(single_sample, np.ndarray)
        self.assertEqual(single_sample.shape, (p_dim, p_dim))
        self.assertTrue(np.all(np.linalg.eigvals(single_sample) > 0), "IW sample not PD")

        # Test multiple samples (scipy returns (size, p, p))
        num_iw_samples = 5
        multi_samples = sample_inverse_wishart(df, scale_matrix, size=num_iw_samples)
        self.assertIsInstance(multi_samples, np.ndarray)
        self.assertEqual(multi_samples.shape, (num_iw_samples, p_dim, p_dim))
        for i in range(num_iw_samples):
            self.assertTrue(np.all(np.linalg.eigvals(multi_samples[i,:,:]) > 0), f"IW sample {i} not PD")

        # KS test for IW is complex. Check mean for large number of samples.
        # E[X] = Psi / (nu - p - 1)
        n_large_samples = 2000
        large_samples = sample_inverse_wishart(df, scale_matrix, size=n_large_samples)
        mean_of_samples = np.mean(large_samples, axis=0)
        expected_mean = scale_matrix / (df - p_dim - 1)
        np.testing.assert_allclose(mean_of_samples, expected_mean, rtol=0.2, atol=0.2) # Relaxed tolerance for matrix mean

        with self.assertRaises(ValueError): sample_inverse_wishart(df=p_dim - 1.5, scale_matrix=np.eye(p_dim))
        with self.assertRaises(ValueError): sample_inverse_wishart(df=p_dim + 1, scale_matrix=np.array([[1,0],[0,0]])) # Not PD

if __name__ == '__main__':
    unittest.main()
