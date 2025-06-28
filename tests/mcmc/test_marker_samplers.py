import unittest
import numpy as np
from pyjwas.core import GenotypesData, VarianceCovariance
from pyjwas.mcmc.marker_samplers import (
    _sample_marker_effects_rrblup_st,
    _sample_marker_effects_bayesa_st,
    _sample_marker_effects_bayesc_st,
    _sample_marker_effects_bayesb_st
    # _sample_marker_effects_bayesl_st # Not testing yet as it's incomplete
)

class TestMarkerSamplersST(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)
        self.n_obs = 20
        self.n_markers = 10
        self.genotypes = np.random.randint(0, 3, size=(self.n_obs, self.n_markers)).astype(float)
        self.genotypes -= self.genotypes.mean(axis=0) # Center

        self.y_corrected_input = np.random.randn(self.n_obs) * 2.0 # y_obs - X*beta
        self.residual_variance = 1.5
        self.inv_weights = np.ones(self.n_obs)

    def test_sample_rrblup_st(self):
        geno_data = GenotypesData(name="rrblup_test", method="RR-BLUP")
        geno_data.genotypes = self.genotypes.copy()
        geno_data.n_markers = self.n_markers
        geno_data.alpha_samples = [np.zeros(self.n_markers)]

        common_marker_variance = 0.1
        geno_data.marker_effect_variance = VarianceCovariance(value=common_marker_variance) # Fixed

        y_corr = self.y_corrected_input.copy()
        alpha_before = geno_data.alpha_samples[0].copy()

        _sample_marker_effects_rrblup_st(
            geno_data, y_corr, self.residual_variance, common_marker_variance
        )

        self.assertEqual(geno_data.alpha_samples[0].shape, (self.n_markers,))
        self.assertFalse(np.allclose(alpha_before, geno_data.alpha_samples[0]), "Alpha should have changed.")

        # Check if y_corrected was modified (shrunk due to alpha effects)
        # y_corr_new = y_corr_input - Z @ alpha_new
        # Change in y_corr = - Z @ alpha_new (if alpha_old was 0)
        expected_y_corr_after = self.y_corrected_input - geno_data.genotypes @ geno_data.alpha_samples[0]
        # The function modifies y_corr to be y_input - Z @ (alpha_new - alpha_old)
        # If alpha_old was zero, y_corr becomes y_input - Z @ alpha_new
        np.testing.assert_allclose(y_corr, expected_y_corr_after, atol=1e-9)


    def test_sample_bayesa_st(self):
        geno_data = GenotypesData(name="bayesa_test", method="BayesA")
        geno_data.genotypes = self.genotypes.copy()
        geno_data.n_markers = self.n_markers

        # Priors for marker-specific variances sigma_g_j^2
        prior_df_g = 4.0
        prior_scale_s0_g = 0.01
        geno_data.marker_effect_variance = VarianceCovariance(df=prior_df_g, scale=prior_scale_s0_g)
        # Initialize .value as an array for sigma_g_j^2
        geno_data.marker_effect_variance.value = np.full(self.n_markers, prior_scale_s0_g) # Start with prior scale

        geno_data.alpha_samples = [np.random.randn(self.n_markers) * 0.1] # Start with some small random alphas
        alpha_before = geno_data.alpha_samples[0].copy()
        sigma_gj_sq_before = geno_data.marker_effect_variance.value.copy()
        y_corr = self.y_corrected_input.copy()

        _sample_marker_effects_bayesa_st(
            geno_data, y_corr, self.residual_variance, self.inv_weights
        )

        self.assertEqual(geno_data.alpha_samples[0].shape, (self.n_markers,))
        self.assertEqual(geno_data.marker_effect_variance.value.shape, (self.n_markers,))
        self.assertTrue(np.all(geno_data.marker_effect_variance.value > 0), "Marker variances should be positive.")
        # Alphas and sigmas should likely have changed
        self.assertFalse(np.allclose(alpha_before, geno_data.alpha_samples[0]), "BayesA alpha should have changed.")
        self.assertFalse(np.allclose(sigma_gj_sq_before, geno_data.marker_effect_variance.value), "BayesA sigma_g_j^2 should have changed.")

        expected_y_corr_after = self.y_corrected_input - geno_data.genotypes @ geno_data.alpha_samples[0]
        np.testing.assert_allclose(y_corr, expected_y_corr_after, atol=1e-9)


    def test_sample_bayesc_st(self):
        geno_data = GenotypesData(name="bayesc_test", method="BayesC")
        geno_data.genotypes = self.genotypes.copy()
        geno_data.n_markers = self.n_markers

        common_marker_var_val = 0.05
        pi_val = 0.1 # P(effect != 0)
        geno_data.marker_effect_variance = VarianceCovariance(value=common_marker_var_val, df=4.0, scale=0.02, estimate_variance=True)
        geno_data.pi_value = pi_val
        geno_data.estimate_pi = True # Allow pi to be sampled

        geno_data.alpha_samples = [np.zeros(self.n_markers)]
        geno_data.delta_samples = [np.zeros(self.n_markers)] # Start all excluded

        y_corr = self.y_corrected_input.copy()
        pi_before = geno_data.pi_value

        _sample_marker_effects_bayesc_st(
            geno_data, y_corr, self.residual_variance, self.inv_weights,
            pi_prior_alpha=1.0, pi_prior_beta=1.0 # Explicitly pass for test clarity
        )

        self.assertEqual(geno_data.alpha_samples[0].shape, (self.n_markers,))
        self.assertEqual(geno_data.delta_samples[0].shape, (self.n_markers,))
        self.assertTrue(np.all((geno_data.delta_samples[0] == 0) | (geno_data.delta_samples[0] == 1)))
        self.assertTrue(np.all(geno_data.alpha_samples[0][geno_data.delta_samples[0] == 0] == 0.0),
                        "Alpha for excluded markers should be zero.")
        self.assertNotEqual(pi_before, geno_data.pi_value, "Pi should have been sampled.")
        self.assertTrue(0 < geno_data.pi_value < 1)
        self.assertGreater(geno_data.marker_effect_variance.value, 0, "Common marker variance should be positive.")

        expected_y_corr_after = self.y_corrected_input - geno_data.genotypes @ geno_data.alpha_samples[0]
        np.testing.assert_allclose(y_corr, expected_y_corr_after, atol=1e-9)


    def test_sample_bayesb_st(self):
        geno_data = GenotypesData(name="bayesb_test", method="BayesB")
        geno_data.genotypes = self.genotypes.copy()
        geno_data.n_markers = self.n_markers

        prior_df_g_b = 4.0
        prior_scale_s0_g_b = 0.01
        pi_val_b = 0.05
        geno_data.marker_effect_variance = VarianceCovariance(df=prior_df_g_b, scale=prior_scale_s0_g_b, estimate_variance=True)
        # .value will store array of marker-specific variances, initialized by sampler if None
        geno_data.marker_effect_variance.value = None
        geno_data.pi_value = pi_val_b
        geno_data.estimate_pi = True

        geno_data.alpha_samples = [np.zeros(self.n_markers)]
        geno_data.delta_samples = [np.zeros(self.n_markers)]

        y_corr = self.y_corrected_input.copy()
        pi_before = geno_data.pi_value

        _sample_marker_effects_bayesb_st(
            geno_data, y_corr, self.residual_variance, self.inv_weights,
            pi_prior_alpha=1.0, pi_prior_beta=1.0
        )

        self.assertEqual(geno_data.alpha_samples[0].shape, (self.n_markers,))
        self.assertEqual(geno_data.delta_samples[0].shape, (self.n_markers,))
        self.assertEqual(geno_data.marker_effect_variance.value.shape, (self.n_markers,))
        self.assertTrue(np.all((geno_data.delta_samples[0] == 0) | (geno_data.delta_samples[0] == 1)))
        self.assertTrue(np.all(geno_data.alpha_samples[0][geno_data.delta_samples[0] == 0] == 0.0))
        self.assertTrue(np.all(geno_data.marker_effect_variance.value > 0))
        self.assertNotEqual(pi_before, geno_data.pi_value)
        self.assertTrue(0 < geno_data.pi_value < 1)

        expected_y_corr_after = self.y_corrected_input - geno_data.genotypes @ geno_data.alpha_samples[0]
        np.testing.assert_allclose(y_corr, expected_y_corr_after, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
```
