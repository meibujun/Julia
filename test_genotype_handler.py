import unittest
import pandas as pd
import numpy as np
import os
from genotype_handler import (
    read_genotypes_py,
    get_allele_frequencies_from_matrix,
    center_genotype_matrix,
    check_monomorphic
)

class TestGenotypeHandler(unittest.TestCase):

    def setUp(self):
        self.sample_geno_data_csv = """ID,M1,M2,M3,M4,M5
id1,0,1,2,1,0
id2,1,1,1,0,1
id3,2,0,0,1,2
id4,0,0,2,2,0
id5,1,9,1,0,1
""" # id5 has a missing value '9' for M2
        with open("test_geno.csv", "w") as f:
            f.write(self.sample_geno_data_csv)

        self.sample_geno_data_no_header_csv = """id1,0,1,2,1,0
id2,1,1,1,0,1
id3,2,0,0,1,2
"""
        with open("test_geno_no_header.csv", "w") as f:
            f.write(self.sample_geno_data_no_header_csv)


    def tearDown(self):
        for f_name in ["test_geno.csv", "test_geno_no_header.csv"]:
            if os.path.exists(f_name):
                os.remove(f_name)

    def test_read_genotypes_basic_csv(self):
        geno_data = read_genotypes_py("test_geno.csv", header_present=True, separator=',', missing_value_code="9", perform_qc=False, center_genotypes=False)

        self.assertEqual(len(geno_data['obs_ids']), 5)
        self.assertEqual(len(geno_data['marker_ids']), 5)
        self.assertEqual(geno_data['marker_ids'][0], "M1")
        self.assertEqual(geno_data['genotypes_matrix'].shape, (5, 5))
        self.assertFalse(np.isnan(geno_data['genotypes_matrix']).any(), "Missing values should have been imputed by default logic if any were NaN after read")
        # Check if '9' was handled: id5, M2 was 9. Default imputation is mean.
        # M2 data: [1,1,0,0,9]. Mean of non-9 is (1+1+0+0)/4 = 0.5. So M2 for id5 should be 0.5
        self.assertAlmostEqual(geno_data['genotypes_matrix'][4, 1], 0.5)


    def test_read_genotypes_no_header_csv(self):
        geno_data = read_genotypes_py("test_geno_no_header.csv", header_present=False, separator=',', perform_qc=False, center_genotypes=False)
        self.assertEqual(len(geno_data['obs_ids']), 3)
        self.assertEqual(len(geno_data['marker_ids']), 5) # M1 to M5
        self.assertEqual(geno_data['marker_ids'][0], "M1")
        self.assertEqual(geno_data['genotypes_matrix'].shape, (3, 5))
        self.assertEqual(geno_data['genotypes_matrix'][0,0], 0)


    def test_read_genotypes_from_dataframe(self):
        df = pd.read_csv(io.StringIO(self.sample_geno_data_csv))
        geno_data = read_genotypes_py(df, header_present=True, missing_value_code="9", perform_qc=False, center_genotypes=False)
        self.assertEqual(len(geno_data['obs_ids']), 5)
        self.assertEqual(geno_data['marker_ids'][1], "M2")
        self.assertAlmostEqual(geno_data['genotypes_matrix'][4, 1], 0.5) # Imputed missing '9'

    def test_allele_frequencies_and_centering(self):
        # Use a small, predictable matrix
        raw_matrix = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float32)
        obs_ids = ['s1','s2','s3']
        marker_ids = ['m1','m2','m3']
        df = pd.DataFrame(np.hstack((np.array(obs_ids).reshape(-1,1), raw_matrix)), columns=['ID']+marker_ids)

        geno_data = read_genotypes_py(df, header_present=True, perform_qc=False, center_genotypes=True)

        # Expected p: M1: (0+1+2)/3 / 2 = 1/2 = 0.5
        #             M2: (1+1+1)/3 / 2 = 0.5/2 = 0.5
        #             M3: (2+1+0)/3 / 2 = 1/2 = 0.5
        expected_p = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(geno_data['allele_frequencies'], expected_p, decimal=5)

        # Expected means (2p): [1, 1, 1]
        expected_means = 2 * expected_p
        expected_centered_matrix = raw_matrix - expected_means
        np.testing.assert_array_almost_equal(geno_data['genotypes_matrix'], expected_centered_matrix, decimal=5)
        self.assertAlmostEqual(geno_data['sum_2pq'], np.sum(2*expected_p*(1-expected_p)))


    def test_qc_maf_filtering(self):
        # M1: p=0.5 (MAF=0.5)
        # M2: p=0.5 (MAF=0.5)
        # M3: p=0.0 (fixed for 0) -> MAF=0, should be removed
        # M4: p=1.0 (fixed for 2) -> MAF=0, should be removed
        # M5: p=0.1 (MAF=0.1) -> should be kept if maf_threshold = 0.05
        raw_data = """ID,M1,M2,M3,M4,M5
s1,0,1,0,2,0
s2,1,1,0,2,0
s3,1,1,0,2,0
s4,0,1,0,2,1
s5,2,1,0,2,0
""" # M5: one '1', four '0's -> sum=1, mean=0.2, p=0.1
        df = pd.read_csv(io.StringIO(raw_data))
        geno_data = read_genotypes_py(df, header_present=True, perform_qc=True, maf_threshold=0.05, center_genotypes=False)

        self.assertEqual(len(geno_data['marker_ids']), 3) # M1, M2, M5 should remain
        self.assertIn("M1", geno_data['marker_ids'])
        self.assertIn("M2", geno_data['marker_ids'])
        self.assertIn("M5", geno_data['marker_ids'])
        self.assertNotIn("M3", geno_data['marker_ids'])
        self.assertNotIn("M4", geno_data['marker_ids'])
        self.assertEqual(geno_data['genotypes_matrix'].shape[1], 3)


    def test_utility_get_allele_frequencies(self):
        matrix = np.array([[0,2,1],[1,1,1],[2,0,1]], dtype=float) # p for col1=(0+1+2)/6=0.5, col2=(2+1+0)/6=0.5, col3=(1+1+1)/6=0.5
        p_freqs = get_allele_frequencies_from_matrix(matrix)
        np.testing.assert_array_almost_equal(p_freqs, np.array([0.5, 0.5, 0.5]))

    def test_utility_center_genotypes(self):
        matrix = np.array([[0,2,1],[1,1,1],[2,0,1]], dtype=float)
        p_freqs = np.array([0.5, 0.5, 0.5])
        centered_mat, means = center_genotype_matrix(matrix, p_freqs)
        expected_means = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(means, expected_means)
        np.testing.assert_array_almost_equal(centered_mat, matrix - expected_means)

    def test_utility_check_monomorphic(self):
        matrix = np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=float) # Col 0 and 2 are monomorphic
        mono_status = check_monomorphic(matrix)
        np.testing.assert_array_equal(mono_status, np.array([True, False, True]))


if __name__ == '__main__':
    unittest.main()
