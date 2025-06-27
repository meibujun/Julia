import unittest
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import os

from pyjwas.pedigree import PedigreeData, PedNode, read_pedigree, calculate_A_inverse

class TestPedigreeData(unittest.TestCase):

    def _create_dummy_ped_file(self, filename="test_ped.csv", content=None):
        if content is None:
            content = """animal1,0,0
animal2,0,0
animal3,animal1,animal2
animal4,animal1,animal2
animal5,animal3,0
animal6,animal3,animal4
animal7,animal5,animal6
"""
        with open(filename, "w") as f:
            f.write(content.strip())
        return filename

    def tearDown(self):
        # Clean up dummy files
        if os.path.exists("test_ped.csv"): os.remove("test_ped.csv")
        if os.path.exists("test_ped_header.csv"): os.remove("test_ped_header.csv")
        if os.path.exists("IDs_for_individuals_with_pedigree.txt"): # Created by read_pedigree in Julia
             pass # Python version doesn't write this file by default yet.

    def test_read_pedigree_basic(self):
        filepath = self._create_dummy_ped_file()
        ped = read_pedigree(filepath)
        self.assertIsInstance(ped, PedigreeData)
        self.assertTrue(len(ped.id_map) >= 7) # Should have at least these 7, + any unique parents
        self.assertTrue(len(ped.ordered_nodes) >= 7)

        # Check a known animal
        animal3 = ped.id_map.get("animal3")
        self.assertIsNotNone(animal3)
        self.assertEqual(animal3.sire_id, "animal1")
        self.assertEqual(animal3.dam_id, "animal2")
        self.assertTrue(animal3.seq_id > 0)
        self.assertGreaterEqual(animal3.inbreeding_coeff, 0.0) # Should be 0 for this case

        animal6 = ped.id_map.get("animal6") # child of full sibs animal3 and animal4
        self.assertIsNotNone(animal6)
        # F(animal3) = 0 (assuming animal1, animal2 unrelated founders)
        # F(animal4) = 0
        # A(animal3, animal4) = 0.5 * (A(animal1,animal1)+A(animal1,animal2)+A(animal2,animal1)+A(animal2,animal2)) / 2 ??? No.
        # A(animal3, animal4) where 3&4 are full sibs from unrelated parents 1&2:
        # A31=0.5, A32=0.5. A41=0.5, A42=0.5
        # A34 = 0.5(A31+A32) if 1 is sire of 4, 2 is dam of 4. No.
        # A34 = 0.5(A(animal3,sire of 4) + A(animal3,dam of 4))
        # A34 = 0.5(A(animal3,animal1) + A(animal3,animal2))
        # A(animal3,animal1) = 0.5(A(sire of 3, animal1) + A(dam of 3, animal1)) = 0.5(A(animal1,animal1) + A(animal2,animal1)) = 0.5(1+0)=0.5
        # A(animal3,animal2) = 0.5(A(sire of 3, animal2) + A(dam of 3, animal2)) = 0.5(A(animal1,animal2) + A(animal2,animal2)) = 0.5(0+1)=0.5
        # So A34 = 0.5(0.5+0.5) = 0.5
        # F(animal6) = 0.5 * A(animal3,animal4) = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(animal6.inbreeding_coeff, 0.25, places=5)


    def test_read_pedigree_from_dataframe(self):
        data = {
            'ID': ['A', 'B', 'C', 'D'],
            'Sire': ['0', 'A', 'A', 'C'],
            'Dam': ['0', '0', 'B', 'B']
        }
        df = pd.DataFrame(data)
        ped = read_pedigree(df, col_names=['ID', 'Sire', 'Dam']) # Provide col_names if df cols are different
        self.assertEqual(len(ped.id_map), 4)
        node_D = ped.id_map.get('D')
        self.assertEqual(node_D.sire_id, 'C')
        self.assertEqual(node_D.dam_id, 'B')
        # F_A = 0, F_B = 0
        # F_C = 0.5 * A_AB = 0.5 * 0 = 0 (A,B unrelated founders)
        # F_D = 0.5 * A_CB
        # A_CB = 0.5 * (A_C_sireB + A_C_damB) = 0.5 * (A_AB + A_BB) = 0.5 * (0 + (1+F_B)) = 0.5 * 1 = 0.5
        # F_D = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(node_D.inbreeding_coeff, 0.25, places=5)

    def test_calculate_A_inverse_simple(self):
        # Ped: 1(0,0), 2(0,0), 3(1,2)
        content = "1,0,0\n2,0,0\n3,1,2"
        filepath = self._create_dummy_ped_file(content=content)
        ped = read_pedigree(filepath)
        A_inv = calculate_A_inverse(ped)

        self.assertIsInstance(A_inv, csc_matrix)
        self.assertEqual(A_inv.shape, (3,3))

        # Expected A-inv for 3(1,2) where 1,2 are unrelated founders (F=0)
        # d_inv_11 = 1, d_inv_22 = 1
        # d_inv_33 = 1 / (0.5 * (1 - 0.5*(F1+F2))) = 1 / 0.5 = 2
        # T = [[1,0,0], [0,1,0], [-0.5, -0.5, 1]]
        # D_inv = diag([1,1,2])
        # A_inv = T' D_inv T
        # Expected:
        # [[1.5, 0.5, -1],
        #  [0.5, 1.5, -1],
        #  [-1,  -1,  2]]
        expected_A_inv_dense = np.array([
            [1.5, 0.5, -1.0],
            [0.5, 1.5, -1.0],
            [-1.0, -1.0, 2.0]
        ])
        np.testing.assert_array_almost_equal(A_inv.toarray(), expected_A_inv_dense, decimal=5)

    def test_wright_path_example_A_inv_and_F(self):
        # Pedigree: X(0,0), B(X,0), C(X,0), S(B,C), D(B,C), A(S,0), P(D,S)
        wright_ped_data = [
            ['X', '0', '0'], ['B', 'X', '0'], ['C', 'X', '0'],
            ['S', 'B', 'C'], ['D', 'B', 'C'], ['A', 'S', '0'],
            ['P', 'D', 'S']
        ]
        wright_df = pd.DataFrame(wright_ped_data, columns=['Ind', 'Sire', 'Dam'])
        ped = read_pedigree(wright_df)

        # Check inbreeding of P
        node_P = ped.id_map.get('P')
        self.assertIsNotNone(node_P)
        # F_X=0. F_B=0 (parent X known, other unknown). F_C=0.
        # A_BC = 0.5(A_BX + A_CX) (assuming B,C are from common sire X, different dams)
        # No, A_BC = 0.5(A_B_sireC + A_B_damC) = 0.5(A_BX + 0) if C=(X,0)
        # A_BX = 0.5(A_XX + A_X0) = 0.5(1+0) = 0.5. So A_BC = 0.25.
        # F_S = F_D = 0.5 * A_BC = 0.5 * 0.25 = 0.125.
        # A_DS (full sibs S,D with F_B=0, F_C=0): A_DS = 0.5 * (1 + A_BC) (if parents are B,C for both)
        # A_DS = 0.5 * (1 + 0.25) = 0.625
        # F_P = 0.5 * A_DS = 0.5 * 0.625 = 0.3125
        self.assertAlmostEqual(node_P.inbreeding_coeff, 0.3125, places=5)

        # A_inv = calculate_A_inverse(ped)
        # Validating full A-inv is complex here, check shape
        # Order: X, B, C, S, D, A, P (example, depends on tie-breaking in sort)
        # self.assertEqual(A_inv.shape, (7,7))
        # For now, just check if it runs
        try:
            A_inv = calculate_A_inverse(ped)
            self.assertEqual(A_inv.shape, (len(ped.ordered_nodes), len(ped.ordered_nodes)))
        except Exception as e:
            self.fail(f"calculate_A_inverse failed for Wright's example: {e}")


if __name__ == '__main__':
    unittest.main()
```
