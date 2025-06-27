import unittest
import numpy as np
from scipy.sparse import csc_matrix

from pyjwas.genotypes_io.utils_geno import (
    center_genotype_matrix_inplace,
    calculate_allele_frequencies_from_means,
    make_incidence_matrix_for_ids,
    align_genotypes_with_phenotypes,
    setup_marker_hyperparameters,
    _genetic_to_marker_variance_st,
    _genetic_to_marker_variance_mt
)
from pyjwas.core import GenotypesData, VarianceCovariance # For creating test instances

class TestGenotypeUtils(unittest.TestCase):

    def test_center_genotype_matrix_inplace(self):
        geno = np.array([[0., 1., 2.], [1., 2., 0.], [2., 0., 1.]], dtype=float)
        expected_means = np.array([1., 1., 1.])
        returned_means = center_genotype_matrix_inplace(geno)
        np.testing.assert_allclose(returned_means, expected_means, atol=1e-9)
        np.testing.assert_allclose(np.mean(geno, axis=0), np.zeros(3), atol=1e-9)

        empty_geno = np.array([[]])
        self.assertIsNone(center_genotype_matrix_inplace(empty_geno))

    def test_calculate_allele_frequencies_from_means(self):
        means = np.array([1.0, 0.5, 1.8]) # Assuming diploid, 0,1,2 coding
        expected_p = np.array([0.5, 0.25, 0.9])
        p = calculate_allele_frequencies_from_means(means, ploidy=2)
        np.testing.assert_allclose(p, expected_p)
        with self.assertRaises(ValueError):
            calculate_allele_frequencies_from_means(means, ploidy=0)

    def test_make_incidence_matrix_for_ids(self):
        target = ["C", "A", "D", "B"]
        source = ["A", "B", "C", "D", "E"]
        Z = make_incidence_matrix_for_ids(target, source)
        self.assertIsInstance(Z, csc_matrix)
        self.assertEqual(Z.shape, (4, 5))
        # C at target[0] is source[2] -> Z[0,2]=1
        # A at target[1] is source[0] -> Z[1,0]=1
        # D at target[2] is source[3] -> Z[2,3]=1
        # B at target[3] is source[1] -> Z[3,1]=1
        expected_Z_dense = np.array([
            [0,0,1,0,0],
            [1,0,0,0,0],
            [0,0,0,1,0],
            [0,1,0,0,0]
        ])
        np.testing.assert_array_equal(Z.toarray(), expected_Z_dense)

        with self.assertRaises(ValueError): # Target ID not in source
            make_incidence_matrix_for_ids(["A", "F"], ["A", "B"])

        self.assertEqual(make_incidence_matrix_for_ids([], ["A","B"]).shape, (0,2))
        self.assertEqual(make_incidence_matrix_for_ids(["A"], []).shape, (1,0))


    def test_align_genotypes_with_phenotypes(self):
        pheno_ids = ["id1", "id3", "id2"] # Target order

        geno_data1 = GenotypesData(name="markers1")
        geno_data1.obs_ids = ["id1", "id2", "id3", "id4"]
        geno_data1.genotypes = np.array([[1,1],[2,2],[3,3],[4,4]]) # 4x2

        geno_data_grm = GenotypesData(name="grm1", is_grm=True)
        geno_data_grm.obs_ids = ["id3", "id1", "id2"] # Different order, but all in pheno
        # A 3x3 GRM. Let's make it simple identity for testing reordering
        geno_data_grm.genotypes = np.eye(3)
        # Original mapping: id3->row0, id1->row1, id2->row2 of this eye(3)

        align_genotypes_with_phenotypes([geno_data1, geno_data_grm], pheno_ids)

        # Check geno_data1 (markers)
        self.assertEqual(geno_data1.obs_ids, pheno_ids)
        self.assertEqual(geno_data1.n_obs, 3)
        # Expected genotypes: rows for id1, id3, id2 from original
        # Original: id1->[1,1], id2->[2,2], id3->[3,3]
        # New order: id1, id3, id2 -> [[1,1], [3,3], [2,2]]
        np.testing.assert_array_equal(geno_data1.genotypes, np.array([[1,1],[3,3],[2,2]]))

        # Check geno_data_grm (GRM)
        self.assertEqual(geno_data_grm.obs_ids, pheno_ids)
        self.assertEqual(geno_data_grm.n_obs, 3)
        # Original GRM (eye(3)) was for order id3, id1, id2
        # Target order: id1, id3, id2
        # Z_pheno for GRM: maps [id1,id3,id2] from [id3,id1,id2]
        # id1 (target[0]) <- id1 (source[1])
        # id3 (target[1]) <- id3 (source[0])
        # id2 (target[2]) <- id2 (source[2])
        # Z_pheno = [[0,1,0], [1,0,0], [0,0,1]]
        # G_new = Z_pheno @ eye(3) @ Z_pheno.T
        Z_pheno_grm = np.array([[0,1,0],[1,0,0],[0,0,1]])
        expected_grm_new = Z_pheno_grm @ np.eye(3) @ Z_pheno_grm.T
        np.testing.assert_array_equal(geno_data_grm.genotypes, expected_grm_new)


    def test_genetic_to_marker_variance_st(self):
        gv = 50.0
        sum2pq = 250.0
        pi_eff = 0.95 # Effective proportion of markers with effects
        expected_mv = 50.0 / (0.95 * 250.0) # = 50.0 / 237.5 = 0.2105263
        mv = _genetic_to_marker_variance_st(gv, sum2pq, pi_eff)
        self.assertAlmostEqual(mv, expected_mv)
        self.assertEqual(_genetic_to_marker_variance_st(gv, 0, pi_eff), 0.0) # sum2pq = 0
        self.assertGreater(_genetic_to_marker_variance_st(gv, sum2pq, 0), 1e10) # pi_eff = 0


    def test_genetic_to_marker_variance_mt(self):
        gen_cov = np.array([[10., 2.], [2., 8.]])
        sum2pq = 300.0
        # Pi: only pattern (1,1) has effect with prob 1.0
        pi_d = { (1.0,1.0): 1.0, (1.0,0.0):0.0, (0.0,1.0):0.0, (0.0,0.0):0.0 }

        # Denom for (0,0) = 300 * 1.0 = 300
        # Denom for (0,1) = 300 * 0.0 = 0 (problematic if G_01 != 0)
        # Denom for (1,1) = 300 * 1.0 = 300
        # The current _genetic_to_marker_variance_mt uses 1e-12 if denom_val is 0 and G_ij!=0.

        expected_marker_cov = np.array([
            [10./(300.*1.), 2./(300.*1.)], # Assuming sum_prob_both_effect for (0,1) will be 1 due to (1,1) pattern
            [2./(300.*1.), 8./(300.*1.)]
        ])
        # Recalculate based on _genetic_to_marker_variance_mt logic:
        # For (0,0): sum_prob = Pi[(1,1)] if 1,1 + Pi[(1,0)] if 1,0 + Pi[(0,1)] if 0,1 + Pi[(0,0)] if 0,0
        # This is not what the code does. It sums Pi[pattern] where pattern[i]=1 and pattern[j]=1.
        # For (0,0): pattern[0]=1, pattern[0]=1 -> Pi[(1,1)]=1, Pi[(1,0)]=0. sum_prob_00=1.0. Denom_00=300.
        # For (1,1): pattern[1]=1, pattern[1]=1 -> Pi[(1,1)]=1. sum_prob_11=1.0. Denom_11=300.
        # For (0,1): pattern[0]=1, pattern[1]=1 -> Pi[(1,1)]=1. sum_prob_01=1.0. Denom_01=300.

        marker_cov = _genetic_to_marker_variance_mt(gen_cov, sum2pq, pi_d)
        np.testing.assert_allclose(marker_cov, gen_cov / 300.0)


    def test_setup_marker_hyperparameters(self):
        geno_st = GenotypesData(name="st_g", method="BayesC")
        geno_st.sum_2pq = 200.0
        geno_st.pi_value = 0.9 # This is P(effect), so 1-pi_0
        geno_st.genetic_variance = VarianceCovariance(value=40.0, df=4.0) # Store prior mean here
        geno_st.marker_effect_variance = VarianceCovariance(value=None, df=4.0, scale=None) # To be derived

        setup_marker_hyperparameters(geno_st, n_model_traits=1)

        expected_marker_var_val = 40.0 / (0.9 * 200.0) # 40 / 180 = 0.2222...
        self.assertAlmostEqual(geno_st.marker_effect_variance.value, expected_marker_var_val)
        # Expected scale for InvGamma(df/2, df*S0^2/2) where S0^2 = prior_value (marker_var_val)
        # If VarianceCovariance.scale is the S0^2, then it should be expected_marker_var_val
        # The Julia logic was: scale = val*(df-p-1) for IW-like, or val*(df-p-1)/df for ST
        # For ST (p=1): scale = val*(df-1-1) = val*(df-2) if df > 2
        # This seems to be calculating Psi_0 for IW from E[Sigma_g]
        # If value is S0^2 for InvChi2, then scale should be value.
        # The current python setup_marker_hyperparameters sets scale = value * (df-p-1) if df > p+1
        # For ST, p=1. So scale = value * (df-2).
        expected_marker_var_scale = expected_marker_var_val * (4.0 - 1 - 1) # df=4, p=1
        self.assertAlmostEqual(geno_st.marker_effect_variance.scale, expected_marker_var_scale)

        # Test MT default Pi
        geno_mt = GenotypesData(name="mt_g", method="BayesC") # Pi is float 0.0
        geno_mt.sum_2pq = 1.0 # To make denom simple
        geno_mt.genetic_variance = VarianceCovariance(value=np.eye(2))
        geno_mt.marker_effect_variance = VarianceCovariance(value=None, df=5.0)
        setup_marker_hyperparameters(geno_mt, n_model_traits=2)
        self.assertIsInstance(geno_mt.pi_value, dict)
        self.assertEqual(geno_mt.pi_value.get((1.0,1.0)), 1.0)
        self.assertEqual(geno_mt.pi_value.get((1.0,0.0)), 0.0)
        # Marker var should be identity because genetic var is identity and denom from Pi[(1,1)] is 1*sum2pq
        np.testing.assert_allclose(geno_mt.marker_effect_variance.value, np.eye(2) / geno_mt.sum_2pq)


if __name__ == '__main__':
    unittest.main()
```
