import unittest
import pandas as pd
import io
import os
from pedigree_module import Pedigree, get_pedigree, _calculate_additive_relationship # PedNode not directly tested but used
import numpy as np
import scipy.sparse as sps

class TestPedigreeModule(unittest.TestCase):

    def setUp(self):
        # Pedigree without loops or self-referencing founders for basic tests
        self.simple_ped_data_csv = """A,0,0
B,0,0
C,A,B
D,A,C
E,D,B
F,D,C
G,E,F
H,G,0
I,H,G
J,C,C
"""
        self.simple_ped_df = pd.read_csv(io.StringIO(self.simple_ped_data_csv), header=None,
                                         names=['individual', 'sire', 'dam'], dtype=str)

        # Pedigree with problematic entries for specific tests
        self.complex_ped_data_csv = self.simple_ped_data_csv + """K,K,K
L,M,B
M,L,A
"""
        self.complex_ped_df = pd.read_csv(io.StringIO(self.complex_ped_data_csv), header=None,
                                          names=['individual', 'sire', 'dam'], dtype=str)

        with open("test_simple_ped.csv", "w") as f:
            f.write(self.simple_ped_data_csv)

        # For genoSet tests
        self.genotyped_ids = ["D", "E", "G", "I"]
        self.core_genotyped_ids = ["E", "G"]

        with open("test_geno_ids.txt", "w") as f:
            for item in self.genotyped_ids:
                f.write(f"{item}\n")
        with open("test_core_geno_ids.txt", "w") as f:
            for item in self.core_genotyped_ids:
                f.write(f"{item}\n")


    def tearDown(self):
        for f_name in ["test_simple_ped.csv", "test_geno_ids.txt", "test_core_geno_ids.txt"]:
            if os.path.exists(f_name):
                os.remove(f_name)

    def test_read_pedigree_from_dataframe(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        self.assertIsInstance(pedigree, Pedigree)
        self.assertEqual(len(pedigree.id_map), 10) # A-J
        self.assertIn("A", pedigree.id_map)
        self.assertIn("G", pedigree.id_map)
        self.assertEqual(pedigree.id_map["A"].sire, "missing")
        self.assertEqual(pedigree.id_map["C"].sire, "A")
        self.assertEqual(pedigree.id_map["C"].dam, "B")

    def test_read_pedigree_from_file(self):
        pedigree = get_pedigree("test_simple_ped.csv", header=False, separator=',')
        self.assertIsInstance(pedigree, Pedigree)
        self.assertEqual(len(pedigree.id_map), 10)
        self.assertIn("A", pedigree.id_map)
        self.assertEqual(pedigree.id_map["E"].sire, "D")
        self.assertEqual(pedigree.id_map["E"].dam, "B")

    def test_inbreeding_coefficients_simple_case(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())

        self.assertAlmostEqual(pedigree.id_map["A"].f, 0.0, places=5)
        self.assertAlmostEqual(pedigree.id_map["B"].f, 0.0, places=5)
        self.assertAlmostEqual(pedigree.id_map["C"].f, 0.0, places=5)
        self.assertAlmostEqual(pedigree.id_map["D"].f, 0.25, places=5)
        self.assertAlmostEqual(pedigree.id_map["E"].f, 0.125, places=5)
        self.assertAlmostEqual(pedigree.id_map["F"].f, 0.375, places=5)
        self.assertAlmostEqual(pedigree.id_map["G"].f, 0.34375, places=5)
        self.assertAlmostEqual(pedigree.id_map["H"].f, 0.0, places=5) # Sire G, Dam missing
        # F_I: parents H, G. F_I = 0.5 * a_HG
        # a_HG = 0.5 * (a_H,E + a_H,F) (Parents of G: E,F)
        #   a_HE = 0.5 * (a_G,E + a_0,E) (Parents of H: G,0) = 0.5 * a_GE
        #     a_GE = 0.5 * (1 + F_G + a_G,F_other_parent_of_E_is_D) -> No, E is parent of G
        #     a_GE = 0.5 * (1 + F_E + a_E,F_other_parent_of_G_is_F) = 0.5 * (1 + 0.125 + pedigree.additive_relationships[(pedigree.id_map['E'].seq_id, pedigree.id_map['F'].seq_id)])
        #     This gets complicated, let's use the direct call.
        # F_I = 0.5 * _calculate_additive_relationship(pedigree, "H", "G")
        # Need a_HG. H's parents (G, missing). G's parents (E,F)
        # a_HG = 0.5 * (a_G,G + a_missing,G) = 0.5 * a_GG
        # a_GG = 1 + F_G = 1 + 0.34375 = 1.34375
        # a_HG = 0.5 * 1.34375 = 0.671875
        # F_I = 0.5 * 0.671875 = 0.3359375
        self.assertAlmostEqual(pedigree.id_map["I"].f, 0.3359375, places=5)


    def test_inbreeding_self_fertilization(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        # J parents C, C. F_J = 0.5 * a_CC
        # a_CC = 1 + F_C. F_C = 0 (parents A,B founders)
        # a_CC = 1.0. F_J = 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(pedigree.id_map["J"].f, 0.5, places=5)

    def test_problematic_pedigree_entries(self):
        # K,K,K and L,M,B / M,L,A loops
        with self.assertRaises(RecursionError):
             get_pedigree(self.complex_ped_df.copy())

    def test_additive_relationship(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        self.assertAlmostEqual(_calculate_additive_relationship(pedigree, "A", "C"), 0.5, places=5)
        self.assertAlmostEqual(_calculate_additive_relationship(pedigree, "C", "D"), 0.75, places=5)
        self.assertAlmostEqual(_calculate_additive_relationship(pedigree, "B", "D"), 0.25, places=5)

    def test_ordered_ids(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        expected_order = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.assertEqual(pedigree.ordered_ids, expected_order)
        # Verify seq_ids correspond to this order
        for i, ind_id in enumerate(expected_order):
            self.assertEqual(pedigree.id_map[ind_id].seq_id, i + 1)

    def test_get_inbreeding_coefficients(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        inb_coeffs = pedigree.get_inbreeding_coefficients()
        self.assertEqual(len(inb_coeffs), 10)
        self.assertAlmostEqual(inb_coeffs["D"], 0.25, places=5)
        self.assertAlmostEqual(inb_coeffs["G"], 0.34375, places=5)

    def test_A_inverse(self):
        # Test with a very small pedigree: A,0,0; B,A,0; C,A,B
        # IDs: A, B, C. SeqIDs: A=1, B=2, C=3
        # F_A=0, F_B=0.5*a_A0=0, F_C=0.5*a_AB
        # a_AB = 0.5*(a_AA + a_A0) = 0.5*(1+0+0) = 0.5. So F_C = 0.25
        # H matrix elements (0-indexed for A,B,C -> 0,1,2):
        # A (0): s=m, d=m. H[0,0]=1
        # B (1): s=A(0), d=m. d_val = sqrt(4/(3-F_A)) = sqrt(4/3) = 2/sqrt(3)
        #        H[1,0]=-0.5*d_val, H[1,1]=d_val
        # C (2): s=A(0), d=B(1). d_val = sqrt(4/(2-F_A-F_B)) = sqrt(4/2) = sqrt(2)
        #        H[2,0]=-0.5*d_val, H[2,1]=-0.5*d_val, H[2,2]=d_val
        ped_data = "A,0,0\nB,A,0\nC,A,B"
        ped_df = pd.read_csv(io.StringIO(ped_data), header=None, names=['i','s','d'], dtype=str)
        ped = get_pedigree(ped_df)

        A_inv = ped.calculate_A_inverse().toarray() # Convert sparse to dense for easy check

        # Expected H (example from Henderson's paper or similar):
        # H = [[1,   0,    0  ],
        #      [-0.5773, 1.1547, 0  ],  (d_val_B = 1.1547)
        #      [-0.7071, -0.7071, 1.4142]] (d_val_C = 1.4142)

        # Manual H calculation based on code:
        # A (id A, seq 1, 0-idx 0): F=0. d_val=1. H[0,0]=1.
        # B (id B, seq 2, 0-idx 1): Sire A (seq 1, 0-idx 0), F_A=0. d_val=sqrt(4/3).
        #    H[1,0]=-0.5*sqrt(4/3), H[1,1]=sqrt(4/3)
        # C (id C, seq 3, 0-idx 2): Sire A (seq 1, 0-idx 0), Dam B (seq 2, 0-idx 1). F_A=0, F_B=0.
        #    d_val=sqrt(4/(2-0-0)) = sqrt(2).
        #    H[2,0]=-0.5*sqrt(2), H[2,1]=-0.5*sqrt(2), H[2,2]=sqrt(2)

        h_rows, h_cols, h_vals = ped._h_a_inverse_elements()
        H_matrix = sps.csc_matrix((h_vals, (h_rows, h_cols)), shape=(3,3)).toarray()

        expected_H = np.zeros((3,3))
        expected_H[0,0] = 1.0
        d_val_B = np.sqrt(4/3)
        expected_H[1,0] = -0.5 * d_val_B; expected_H[1,1] = d_val_B
        d_val_C = np.sqrt(2)
        expected_H[2,0] = -0.5 * d_val_C; expected_H[2,1] = -0.5 * d_val_C; expected_H[2,2] = d_val_C

        np.testing.assert_array_almost_equal(H_matrix, expected_H, decimal=4)

        # Expected A_inv = H'H
        expected_A_inv = H_matrix.T @ H_matrix
        np.testing.assert_array_almost_equal(A_inv, expected_A_inv, decimal=4)


    def test_get_info(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        # Test basic printout (capture stdout or just run for coverage)
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        pedigree.get_info(calculate_A_inv=False)
        sys.stdout = old_stdout
        output_str = captured_output.getvalue()
        self.assertIn("Pedigree Information:", output_str)
        self.assertIn("# Individuals with assigned sequence IDs: 10", output_str)
        self.assertIn("# Sires: 3", output_str) # A, D, E, G -> D, E, G are sires. A is sire. C is not sire. H is sire.
                                            # Sires: A, D, E, G, C, H. Unique: A,C,D,E,G,H -> 6 sires
                                            # Correcting: Sires are A (of C,D), D (of E,F), E (of G), G (of H,I), C (of J)
                                            # These are 5 sires.
        # Let's re-verify sires from simple_ped_data_csv:
        # A (sire of C, D)
        # D (sire of E, F)
        # E (sire of G)
        # G (sire of H, I)
        # C (sire of J)
        # Unique sires: A, C, D, E, G. Total 5.
        # Unique dams: B (of C), C (of D,F,J), B (of E), F (of G), G (of I)
        # Unique dams: B, C, F, G. Total 4.

        self.assertIn("# Sires: 5", output_str)
        self.assertIn("# Dams: 4", output_str)
        self.assertIn("# Founders (among ordered_ids): 2", output_str) # A, B

        # Test with A_inv calculation
        ids, A_inv_matrix, inb_coeffs = pedigree.get_info(calculate_A_inv=True)
        self.assertEqual(len(ids), 10)
        self.assertTrue(A_inv_matrix is not None)
        self.assertEqual(A_inv_matrix.shape, (10,10))
        self.assertEqual(len(inb_coeffs), 10)
        self.assertAlmostEqual(inb_coeffs["D"], 0.25)


    def test_set_genotyped_individuals_basic(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        initial_ordered_ids = list(pedigree.ordered_ids) # A,B,C,D,E,F,G,H,I,J

        # Genotyped: D, E, G, I
        # Non-genotyped: A, B, C, F, H, J (6 individuals)
        # Genotyped: D, E, G, I (4 individuals)
        # Expected new order: A,B,C,F,H,J (seq 1-6), then D,E,G,I (seq 7-10)
        # (Order within NG and G groups is alphabetical due to sorted list iteration)

        count_NG, count_G_core, count_G = pedigree.set_genotyped_individuals(genotyped_ids=self.genotyped_ids)

        self.assertEqual(count_NG, 6)
        self.assertEqual(count_G_core, 0) # No core specified
        self.assertEqual(count_G, 4)

        self.assertEqual(pedigree.set_G, set(self.genotyped_ids))
        expected_NG = set(initial_ordered_ids) - set(self.genotyped_ids)
        self.assertEqual(pedigree.set_NG, expected_NG)

        # Check re-coding of seq_ids and ordered_ids
        # Expected order: sorted(NG) + sorted(G)
        expected_new_order = sorted(list(expected_NG)) + sorted(list(self.genotyped_ids))
        self.assertEqual(pedigree.ordered_ids, expected_new_order)

        # Verify seq_ids from the map
        for i, ind_id in enumerate(expected_new_order):
            self.assertEqual(pedigree.id_map[ind_id].seq_id, i + 1, f"Seq ID mismatch for {ind_id}")

        # Check if inbreeding coefficients are still valid (or re-calculated)
        self.assertAlmostEqual(pedigree.id_map["D"].f, 0.25, places=5) # D's inbreeding shouldn't change
        self.assertAlmostEqual(pedigree.id_map["G"].f, 0.34375, places=5)


    def test_set_genotyped_individuals_with_core(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        initial_ordered_ids = list(pedigree.ordered_ids)

        # Genotyped: D, E, G, I (4)
        # Core: E, G (2)
        # Non-Core Genotyped: D, I (2)
        # Non-Genotyped: A, B, C, F, H, J (6)
        # Expected order: NG (sorted), G_core (sorted), G_notcore (sorted)

        count_NG, count_G_core, count_G_notcore = pedigree.set_genotyped_individuals(
            genotyped_ids=self.genotyped_ids,
            core_genotyped_ids=self.core_genotyped_ids
        )

        self.assertEqual(count_NG, 6)
        self.assertEqual(count_G_core, 2)
        self.assertEqual(count_G_notcore, 2)

        expected_NG_set = set(initial_ordered_ids) - set(self.genotyped_ids)
        expected_G_core_set = set(self.core_genotyped_ids)
        expected_G_notcore_set = set(self.genotyped_ids) - set(self.core_genotyped_ids)

        self.assertEqual(pedigree.set_NG, expected_NG_set)
        self.assertEqual(pedigree.set_G_core, expected_G_core_set)
        self.assertEqual(pedigree.set_G_notcore, expected_G_notcore_set)

        expected_new_order = sorted(list(expected_NG_set)) + \
                               sorted(list(expected_G_core_set)) + \
                               sorted(list(expected_G_notcore_set))
        self.assertEqual(pedigree.ordered_ids, expected_new_order)

        for i, ind_id in enumerate(expected_new_order):
            self.assertEqual(pedigree.id_map[ind_id].seq_id, i + 1, f"Seq ID mismatch for {ind_id}")

    def test_set_genotyped_individuals_from_files(self):
        pedigree = get_pedigree(self.simple_ped_df.copy())
        initial_ordered_ids = list(pedigree.ordered_ids)

        count_NG, count_G_core, count_G_notcore = pedigree.set_genotyped_individuals(
            genotyped_ids_file="test_geno_ids.txt",
            core_genotyped_ids_file="test_core_geno_ids.txt"
        )
        self.assertEqual(count_NG, 6)
        self.assertEqual(count_G_core, 2) # E, G
        self.assertEqual(count_G_notcore, 2) # D, I

        expected_NG_set = set(initial_ordered_ids) - set(self.genotyped_ids)
        expected_G_core_set = set(self.core_genotyped_ids)
        expected_G_notcore_set = set(self.genotyped_ids) - set(self.core_genotyped_ids)

        expected_new_order = sorted(list(expected_NG_set)) + \
                               sorted(list(expected_G_core_set)) + \
                               sorted(list(expected_G_notcore_set))
        self.assertEqual(pedigree.ordered_ids, expected_new_order)


if __name__ == '__main__':
    unittest.main()

