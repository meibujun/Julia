import unittest
import numpy as np
import pandas as pd
import os
from scipy.sparse import csc_matrix, eye as sparse_eye

from pyjwas.pedigree import PedigreeData, read_pedigree # calculate_A_inverse is implicitly tested via H_inv
from pyjwas.core import GenotypesData
from pyjwas.single_step import calculate_H_inverse

class TestSingleStepRelationships(unittest.TestCase):

    def _create_dummy_ped_file_for_ss(self, filename="test_ped_ss.csv"):
        # Pedigree:
        # 1: 0 0 (NG)
        # 2: 0 0 (NG)
        # 3: 1 2 (G)
        # 4: 1 0 (G)
        # 5: 3 4 (G)
        ped_content = """1,0,0
2,0,0
3,1,2
4,1,0
5,3,4
"""
        with open(filename, "w") as f:
            f.write(ped_content.strip())
        return filename

    def tearDown(self):
        if os.path.exists("test_ped_ss.csv"): os.remove("test_ped_ss.csv")
        # Clean up pedigree's default ID output file if it gets created by read_pedigree's __main__
        if os.path.exists("IDs_for_individuals_with_pedigree.txt"):
             os.remove("IDs_for_individuals_with_pedigree.txt")


    def test_calculate_H_inverse_simple_case(self):
        filepath = self._create_dummy_ped_file_for_ss()
        ped = read_pedigree(filepath)

        # Genotyped animals: 3, 4, 5.
        # Original GRM for them, assume order ["3", "4", "5"] for G_original
        grm_obs_ids_original_order = ["3", "4", "5"]
        G_original = np.array([[1.1, 0.1, 0.2],
                               [0.1, 1.0, 0.15],
                               [0.2, 0.15, 1.05]])
        G_original = (G_original + G_original.T) / 2.0
        G_original += np.eye(3) * 0.05 # Ensure PD for test

        geno_grm = GenotypesData(name="grm_test", obs_ids=grm_obs_ids_original_order, genotypes=G_original, is_grm=True)
        weight_G_val = 0.95

        H_inv, H_ids = calculate_H_inverse(ped, geno_grm, weight_G=weight_G_val)

        self.assertIsInstance(H_inv, csc_matrix)
        self.assertEqual(H_inv.shape, (5, 5), "H_inv shape mismatch.")
        self.assertEqual(len(H_ids), 5, "H_ids length mismatch.")

        n_non_geno = ped.n_non_genotyped
        self.assertEqual(n_non_geno, 2)
        for i in range(n_non_geno):
            self.assertIn(H_ids[i], ["1", "2"])
        for i in range(n_non_geno, 5):
            self.assertIn(H_ids[i], ["3", "4", "5"])

        np.testing.assert_allclose(H_inv.toarray(), H_inv.toarray().T, atol=1e-9, err_msg="H_inv is not symmetric.")

        # Further numerical validation is complex without a known reference H_inv for this specific G & A.
        # We are testing the refined A22_inv calculation path.
        # If A22_inv_direct was None, it means fallback was used. We'd want to know.
        # This test primarily checks if the new logic runs and produces a symmetric H_inv.


    def test_calculate_H_inverse_all_genotyped(self):
        # Pedigree: 3 animals, all genotyped
        ped_content = "1,0,0\n2,0,0\n3,1,2"
        filepath = self._create_dummy_ped_file(content=ped_content)
        ped = read_pedigree(filepath)

        genotyped_ids = ["1", "2", "3"]
        G_matrix = np.array([[1.0, 0.2, 0.1],
                             [0.2, 1.0, 0.3],
                             [0.1, 0.3, 1.0]])
        G_matrix += np.eye(3) * 0.05 # Ensure PD
        geno_grm = GenotypesData(name="grm_all", obs_ids=genotyped_ids, genotypes=G_matrix, is_grm=True)

        # When all are genotyped, n_non_genotyped = 0.
        # A22 = A. A22_inv = A_inv.
        # H_inv = A_inv + G_tuned_inv - A_inv = G_tuned_inv.
        # If weight_G = 1.0, G_tuned = G. So H_inv = G_inv.

        H_inv, H_ids = calculate_H_inverse(ped, geno_grm, weight_G=1.0)

        self.assertEqual(H_inv.shape, (3,3))
        self.assertEqual(set(H_ids), set(genotyped_ids))

        expected_H_inv = np.linalg.inv(G_matrix)
        np.testing.assert_allclose(H_inv.toarray(), expected_H_inv, atol=1e-6)

    def test_calculate_H_inverse_no_genotyped(self):
        # If no animals are genotyped, H_inv should be A_inv.
        ped_content = "1,0,0\n2,1,0\n3,1,2"
        filepath = self._create_dummy_ped_file(content=ped_content)
        ped = read_pedigree(filepath)

        empty_geno_grm = GenotypesData(name="empty_grm", obs_ids=[], genotypes=np.array([]).reshape(0,0), is_grm=True)

        # The calculate_H_inverse expects geno_data_for_grm.obs_ids to be used for set_genotyped_animals.
        # If this list is empty, n_non_genotyped = n_total_ped.
        # The code should handle this: A_inv_gg, A_inv_gn, A_inv_ng will be empty or not used.
        # A22_inv_direct should become None or empty.
        # The G_aligned part will be empty. G_tuned will be empty.
        # The block addition H_inv[idx_g, idx_g] += ... will be on an empty slice.
        # So H_inv should remain A_inv.

        # calculate_A_inverse is called inside calculate_H_inverse after reordering.
        # So, get A_inv first for comparison.
        A_inv_expected = calculate_A_inverse(ped) # Based on original order before any genoSet call
        # Reset seq_ids if calculate_A_inverse modified them, or re-read ped.
        ped = read_pedigree(filepath) # Reread for fresh state

        H_inv, H_ids = calculate_H_inverse(ped, empty_geno_grm, weight_G=0.95)

        self.assertEqual(H_inv.shape, A_inv_expected.shape)
        np.testing.assert_allclose(H_inv.toarray(), A_inv_expected.toarray(), atol=1e-9)
        self.assertEqual(H_ids, ped.get_ordered_ids_str()) # Order from original full pedigree processing


if __name__ == '__main__':
    unittest.main()

```
