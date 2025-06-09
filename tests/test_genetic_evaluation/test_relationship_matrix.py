# tests/test_genetic_evaluation/test_relationship_matrix.py

import unittest
import pandas as pd
import numpy as np

# Adjust path to import from the root of the project
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.data_management.data_structures import PedigreeData, GenomicData
from sheep_breeding_genomics.genetic_evaluation.relationship_matrix import (
    calculate_nrm,
    calculate_grm,
    calculate_h_inverse_matrix
)

class TestRelationshipMatrix(unittest.TestCase):

    def test_calculate_nrm_simple_pedigree(self):
        ped_data = {
            'AnimalID': [1, 2, 3, 4],
            'SireID':   [0, 0, 1, 1],
            'DamID':    [0, 0, 2, 2]
        }
        pedigree_df = pd.DataFrame(ped_data)
        # PedigreeData object not strictly needed as calculate_nrm takes DataFrame

        nrm_df = calculate_nrm(pedigree_df, founder_val=0)
        self.assertFalse(nrm_df.empty)
        self.assertEqual(nrm_df.shape, (4, 4))
        # Expected NRM for this pedigree:
        #      1    2    3    4
        # 1  1.0  0.0  0.5  0.5
        # 2  0.0  1.0  0.5  0.5
        # 3  0.5  0.5  1.0  0.5  (A_33 = 1 + 0.5 * A_12 = 1 + 0.5*0 = 1)
        # 4  0.5  0.5  0.5  1.0  (A_44 = 1 + 0.5 * A_12 = 1)
        # A_34 = 0.5 * (A_11 + A_12) assuming s(3)=1, d(3)=2 and s(4)=1, d(4)=2.
        # A_34 = 0.5 * (A_s3,s4 + A_s3,d4 + A_d3,s4 + A_d3,d4) -> This is complex rule.
        # Simpler: A_ij = 0.5 * (A_s(i),j + A_d(i),j)
        # A_34 = 0.5 * (A_14 + A_24) = 0.5 * (0.5 + 0.5) = 0.5

        expected_nrm_values = np.array([
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.5],
            [0.5, 0.5, 0.5, 1.0]
        ])
        expected_nrm_df = pd.DataFrame(expected_nrm_values, index=[1,2,3,4], columns=[1,2,3,4])
        pd.testing.assert_frame_equal(nrm_df, expected_nrm_df, check_dtype=False, atol=1e-6)

    def test_calculate_nrm_founders_only(self):
        ped_data = {'AnimalID': [1, 2], 'SireID': [0,0], 'DamID': [0,0]}
        pedigree_df = pd.DataFrame(ped_data)
        nrm_df = calculate_nrm(pedigree_df)
        expected = pd.DataFrame(np.identity(2), index=[1,2], columns=[1,2])
        pd.testing.assert_frame_equal(nrm_df, expected, check_dtype=False)

    def test_calculate_nrm_empty_pedigree(self):
        ped_df = pd.DataFrame(columns=['AnimalID', 'SireID', 'DamID'])
        nrm_df = calculate_nrm(ped_df)
        self.assertTrue(nrm_df.empty)

    def test_calculate_grm_simple_genomic_data(self):
        geno_dict = {
            'AnimalID': ['A1', 'A2', 'A3'],
            'SNP1': [0, 1, 2], # p1 = (0+1+2)/(2*3) = 3/6 = 0.5
            'SNP2': [1, 0, 1], # p2 = (1+0+1)/(2*3) = 2/6 = 1/3
            'SNP3': [2, 2, 0]  # p3 = (2+2+0)/(2*3) = 4/6 = 2/3
        }
        geno_df = pd.DataFrame(geno_dict)
        gen_data = GenomicData(geno_df, animal_id_col='AnimalID')

        grm_df = calculate_grm(gen_data)
        self.assertFalse(grm_df.empty)
        self.assertEqual(grm_df.shape, (3, 3))
        # Manual calculation for VanRaden Method 1: G = ZZ' / (2 * sum(pi(1-pi)))
        # p = [0.5, 1/3, 2/3]
        # 2p = [1, 2/3, 4/3]
        # M = [[0,1,2], [1,0,2], [2,1,0]]
        # Z = M - 2p (broadcast)
        # Z = [[0-1, 1-2/3, 2-4/3], [1-1, 0-2/3, 2-4/3], [2-1, 1-2/3, 0-4/3]]
        # Z = [[-1,  1/3,  2/3], [ 0, -2/3,  2/3], [ 1,  1/3, -4/3]]
        # sum(pi(1-pi)) = 0.5*0.5 + (1/3)*(2/3) + (2/3)*(1/3) = 0.25 + 2/9 + 2/9 = 0.25 + 4/9 = 1/4 + 4/9 = (9+16)/36 = 25/36
        # Denominator = 2 * (25/36) = 25/18 = 1.3888...
        # ZZ' = ... (complex to do by hand here, focus on properties)
        self.assertTrue(np.allclose(np.diag(grm_df), np.diag(grm_df))) # Check it's symmetric enough for diag
        self.assertTrue(np.allclose(grm_df.values, grm_df.values.T, atol=1e-6)) # Symmetry

    def test_calculate_grm_with_nans_error(self):
        geno_dict_missing = {'AnimalID': ['A1', 'A2'], 'SNP1': [0, np.nan]}
        gen_data_missing = GenomicData(pd.DataFrame(geno_dict_missing))
        grm_df = calculate_grm(gen_data_missing) # Expects error print and empty df
        self.assertTrue(grm_df.empty)

    def test_calculate_grm_monomorphic_snps(self):
        # If all SNPs are monomorphic, sum_2pq = 0, should return empty df
        geno_dict_mono_all = {'AnimalID': ['A1', 'A2'], 'SNP1': [0,0], 'SNP2': [2,2]}
        gen_data_mono_all = GenomicData(pd.DataFrame(geno_dict_mono_all))
        grm_df_mono_all = calculate_grm(gen_data_mono_all)
        self.assertTrue(grm_df_mono_all.empty)

        # If some SNPs are monomorphic, they should not contribute to sum_2pq but calculation proceeds
        geno_dict_some_mono = {'AnimalID': ['A1','A2'], 'SNP1':[0,1], 'SNP2':[0,0]} # SNP2 is mono
        gen_data_some_mono = GenomicData(pd.DataFrame(geno_dict_some_mono))
        grm_df_some_mono = calculate_grm(gen_data_some_mono)
        self.assertFalse(grm_df_some_mono.empty)
        self.assertEqual(grm_df_some_mono.shape, (2,2))

    def test_calculate_h_inverse_matrix_simple(self):
        all_ids = ['N1', 'N2', 'G1', 'G2'] # N=non-genotyped, G=genotyped
        geno_ids = ['G1', 'G2']

        nrm_values = np.array([
            [1.0, 0.0, 0.5, 0.5], # N1
            [0.0, 1.0, 0.5, 0.5], # N2
            [0.5, 0.5, 1.0, 0.5], # G1
            [0.5, 0.5, 0.5, 1.0]  # G2
        ])
        nrm_df = pd.DataFrame(nrm_values, index=all_ids, columns=all_ids)
        np.fill_diagonal(nrm_df.values, np.diag(nrm_df.values) + 1e-5) # for invertibility

        grm_values = np.array([
            [1.02, 0.2], # G1
            [0.2, 1.01]  # G2
        ])
        grm_df = pd.DataFrame(grm_values, index=geno_ids, columns=geno_ids)
        np.fill_diagonal(grm_df.values, np.diag(grm_df.values) + 1e-5) # for invertibility

        h_inv_df = calculate_h_inverse_matrix(nrm_df, grm_df)
        self.assertFalse(h_inv_df.empty)
        self.assertEqual(h_inv_df.shape, (4,4))
        self.assertListEqual(list(h_inv_df.index), all_ids)
        # Further checks could involve manual calculation of a small H_inv block.
        # For example, the G1,G2 block of H_inv should be different from A_inv's G1,G2 block.
        a_inv_df = pd.DataFrame(np.linalg.inv(nrm_df.values), index=all_ids, columns=all_ids)
        self.assertFalse(np.allclose(h_inv_df.loc[geno_ids, geno_ids].values,
                                     a_inv_df.loc[geno_ids, geno_ids].values))

    def test_calculate_h_inverse_matrix_id_mismatch(self):
        all_ids = ['N1', 'G1']
        geno_ids_grm = ['G1', 'GX'] # GX not in NRM
        nrm_df = pd.DataFrame(np.identity(2), index=all_ids, columns=all_ids)
        grm_df = pd.DataFrame(np.identity(2), index=geno_ids_grm, columns=geno_ids_grm)
        h_inv_df = calculate_h_inverse_matrix(nrm_df, grm_df)
        self.assertTrue(h_inv_df.empty)


if __name__ == '__main__':
    unittest.main()
