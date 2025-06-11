import unittest
import numpy as np
from animal_breeding_genetics.relationship_matrix import (
    construct_a_matrix,
    construct_a_inverse_matrix,
    _get_ordered_animal_list_and_map, # For testing the helper if needed, or remove if not directly tested
)

class TestRelationshipMatrix(unittest.TestCase):

    def test_get_ordered_animal_list_and_map_simple(self):
        pedigree = [
            (1, None, None),
            (2, None, None),
            (3, 1, 2)
        ]
        animal_list, animal_to_idx, ped_map = _get_ordered_animal_list_and_map(pedigree)
        self.assertEqual(animal_list, [1, 2, 3])
        self.assertEqual(animal_to_idx, {1: 0, 2: 1, 3: 2})
        self.assertEqual(ped_map, {1: (None, None), 2: (None, None), 3: (1, 2)})

    def test_get_ordered_animal_list_and_map_unsorted_pedigree(self):
        pedigree = [
            (4, 2, 3),
            (2, None, None),
            (3, 1, None), # Dam of 3 is unknown, Sire 1
            (1, None, None)
        ]
        # Expected order: 1, 2 (base), then 3 (child of 1), then 4 (child of 2,3)
        animal_list, animal_to_idx, ped_map = _get_ordered_animal_list_and_map(pedigree)
        self.assertEqual(animal_list, [1, 2, 3, 4])
        self.assertEqual(animal_to_idx, {1:0, 2:1, 3:2, 4:3})
        self.assertEqual(ped_map[4], (2,3))
        self.assertEqual(ped_map[3], (1,None))

    def test_get_ordered_animal_list_and_map_with_missing_parents_not_in_pedigree_individually(self):
        pedigree = [(3, 1, 2)] # Parents 1 and 2 are not listed as individuals
        animal_list, animal_to_idx, ped_map = _get_ordered_animal_list_and_map(pedigree)
        # Parents should be included and ordered first
        self.assertEqual(sorted(animal_list), [1, 2, 3]) # Order can be [1,2,3] or [2,1,3]
        self.assertTrue(animal_to_idx[1] < animal_to_idx[3])
        self.assertTrue(animal_to_idx[2] < animal_to_idx[3])
        self.assertEqual(ped_map[1], (None,None)) # Parents assumed base
        self.assertEqual(ped_map[2], (None,None))
        self.assertEqual(ped_map[3], (1,2))

    def test_get_ordered_animal_list_cycle_detection(self):
        pedigree = [(1, None, None), (2, 1, 3), (3, 2, 1)] # 2 and 3 are parent of each other
        with self.assertRaisesRegex(ValueError, "Could not resolve parentage order"):
            _get_ordered_animal_list_and_map(pedigree)

        pedigree_self_parent = [(1,1,None)]
        with self.assertRaisesRegex(ValueError, "Animal 1 cannot be its own parent"):
             _get_ordered_animal_list_and_map(pedigree_self_parent)


    def test_construct_a_matrix_simple_no_inbreeding(self):
        pedigree = [
            (1, None, None),
            (2, None, None),
            (3, 1, 2)
        ]
        # Expected A:
        #   1    2    3
        # 1[1.0  0.0  0.5]
        # 2[0.0  1.0  0.5]
        # 3[0.5  0.5  1.0]
        animal_to_idx, a_matrix, _, _ = construct_a_matrix(pedigree)
        expected_a = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0]
        ])
        np.testing.assert_array_almost_equal(a_matrix, expected_a, decimal=4)

    def test_construct_a_matrix_single_parent(self):
        pedigree = [
            (1, None, None),
            (2, 1, None) # Animal 2 has only sire 1
        ]
        # Expected A:
        #   1    2
        # 1[1.0  0.5]
        # 2[0.5  1.0] (No inbreeding for 2 from one parent)
        animal_to_idx, a_matrix, _, _ = construct_a_matrix(pedigree)
        idx1 = animal_to_idx[1]
        idx2 = animal_to_idx[2]

        # Reorder expected A based on actual indices
        expected_a_ordered = np.zeros((2,2))
        expected_a_ordered[idx1,idx1] = 1.0
        expected_a_ordered[idx2,idx2] = 1.0
        expected_a_ordered[idx1,idx2] = 0.5
        expected_a_ordered[idx2,idx1] = 0.5
        np.testing.assert_array_almost_equal(a_matrix, expected_a_ordered, decimal=4)

    def test_construct_a_matrix_inbreeding_full_sibs(self):
        pedigree = [
            (1, None, None), (2, None, None), # Parents P1, P2
            (3, 1, 2),                     # Offspring F1 (non-inbred)
            (4, 1, 2),                     # Offspring F2 (non-inbred, full sib to F1)
            (5, 3, 4)                      # Offspring I1 (inbred, parents are F1, F2)
        ]
        # A(3,3) = 1.0, A(4,4) = 1.0
        # A(3,4) = A(F1,F2). F1,F2 are full sibs.
        # A(1,1)=1, A(2,2)=1, A(1,2)=0.
        # A(F1,P1) = 0.5*(A(P1,P1)+A(P1,P2)) = 0.5*(1+0) = 0.5
        # A(F1,P2) = 0.5*(A(P2,P1)+A(P2,P2)) = 0.5*(0+1) = 0.5
        # A(F1,F1) = 1 + 0.5*A(P1,P2) = 1+0 = 1.
        # A(F1,F2) = 0.5*(A(F2,P1)+A(F2,P2)) assuming F1 is j, F2 is i (F2 processed after F1)
        # A(F2,P1) = 0.5*(A(P1,P1)+A(P1,P2)) = 0.5 (same as A(F1,P1))
        # A(F2,P2) = 0.5*(A(P2,P1)+A(P2,P2)) = 0.5 (same as A(F1,P2))
        # So, A(F1,F2) = 0.5*(0.5+0.5) = 0.5. This is relationship between full-sibs.
        #
        # For animal 5 (parents 3 and 4):
        # F_5 = 0.5 * A(3,4) = 0.5 * 0.5 = 0.25
        # A(5,5) = 1 + F_5 = 1.25

        animal_to_idx, a_matrix, animal_list, _ = construct_a_matrix(pedigree)
        idx5 = animal_to_idx[5]
        self.assertAlmostEqual(a_matrix[idx5, idx5], 1.25, places=4)

        # Verify relationships
        idx3 = animal_to_idx[3]
        idx4 = animal_to_idx[4]
        self.assertAlmostEqual(a_matrix[min(idx3,idx4), max(idx3,idx4)], 0.5, places=4) # A(3,4)

    def test_construct_a_matrix_complex_inbreeding(self):
        # Example from Mrode (2005), page 42
        pedigree = [
            (1,None,None), (2,None,None), (3,None,None), (4,None,None),
            (5,1,2), (6,1,3), (7,2,4),
            (8,5,6), # F_8 = 0.5 * A(5,6)
            (9,5,7)  # F_9 = 0.5 * A(5,7)
        ]
        # A(5,6): 5=(1,2), 6=(1,3). A(5,6)=0.5(A(5,1)+A(5,3))
        # A(5,1)=0.5(A(1,1)+A(1,2))=0.5(1+0)=0.5
        # A(5,3)=0.5(A(3,1)+A(3,2))=0.5(A(1,3)+A(2,3))=0.5(0+0)=0
        # So A(5,6)=0.5(0.5+0) = 0.25.
        # F_8 = 0.5 * 0.25 = 0.125. A(8,8) = 1.125.

        # A(5,7): 5=(1,2), 7=(2,4). A(5,7)=0.5(A(5,2)+A(5,4))
        # A(5,2)=0.5(A(2,1)+A(2,2))=0.5(0+1)=0.5
        # A(5,4)=0.5(A(4,1)+A(4,2))=0.5(0+0)=0
        # So A(5,7)=0.5(0.5+0) = 0.25.
        # F_9 = 0.5 * 0.25 = 0.125. A(9,9) = 1.125.

        animal_to_idx, a_matrix, animal_list, _ = construct_a_matrix(pedigree)

        # Find actual indices based on sorted order
        idx8 = animal_to_idx[8]
        idx9 = animal_to_idx[9]

        self.assertAlmostEqual(a_matrix[idx8, idx8], 1.125, places=4, msg=f"A(8,8) failed. Animal list: {animal_list}")
        self.assertAlmostEqual(a_matrix[idx9, idx9], 1.125, places=4, msg=f"A(9,9) failed. Animal list: {animal_list}")

    def test_a_inverse_simple_no_inbreeding(self):
        pedigree = [(1,None,None), (2,None,None), (3,1,2)]
        animal_to_idx, a_matrix, animal_list, ped_map = construct_a_matrix(pedigree)
        a_inv_matrix = construct_a_inverse_matrix(a_matrix, animal_list, animal_to_idx, ped_map)

        # Expected A-inverse for this pedigree (Mrode, p. 58)
        #      1     2     3
        # 1 [ 1.5  0.5  -1.0]
        # 2 [ 0.5  1.5  -1.0]
        # 3 [-1.0 -1.0   2.0]
        # This assumes order 1,2,3. We need to map to actual order.

        # Reconstruct expected based on actual animal_to_idx
        # This is tricky to write generally. Let's test A @ A_inv = I
        identity_matrix = np.identity(a_matrix.shape[0])
        np.testing.assert_array_almost_equal(a_matrix @ a_inv_matrix, identity_matrix, decimal=3)

    def test_a_inverse_with_inbreeding(self):
        # Pedigree from VanRaden (1992) example for A-inverse with inbreeding
        # Animal 3 is by 1 and 2. Animal 4 is by 1 and 3.
        # A(1,1)=1, A(2,2)=1, A(1,2)=0
        # A(3,1)=0.5, A(3,2)=0.5, A(3,3)=1 (F3=0)
        # A(4,1)=0.5(A(1,1)+A(1,3)) = 0.5(1+0.5) = 0.75
        # A(4,3)=0.5(A(3,1)+A(3,3)) = 0.5(0.5+1) = 0.75
        # A(4,4)=1+F4. F4 = 0.5*A(1,3) = 0.5*0.5 = 0.25. So A(4,4)=1.25
        pedigree = [
            (1,None,None), (2,None,None),
            (3,1,2),
            (4,1,3)
        ]
        animal_to_idx, a_matrix, animal_list, ped_map = construct_a_matrix(pedigree)

        idx4 = animal_to_idx[4]
        self.assertAlmostEqual(a_matrix[idx4,idx4], 1.25, places=4)

        a_inv_matrix = construct_a_inverse_matrix(a_matrix, animal_list, animal_to_idx, ped_map)
        identity_matrix = np.identity(a_matrix.shape[0])
        np.testing.assert_array_almost_equal(a_matrix @ a_inv_matrix, identity_matrix, decimal=3)

    def test_empty_pedigree(self):
        pedigree = []
        animal_to_idx, a_matrix, animal_list, ped_map = construct_a_matrix(pedigree)
        self.assertEqual(animal_to_idx, {})
        self.assertEqual(a_matrix.shape, (0,) if a_matrix.ndim == 1 else (0,0) ) # handles np.array([]) vs np.empty((0,0))
        self.assertEqual(animal_list, [])
        self.assertEqual(ped_map, {})

        a_inv_matrix = construct_a_inverse_matrix(a_matrix, animal_list, animal_to_idx, ped_map)
        self.assertEqual(a_inv_matrix.shape, (0,) if a_inv_matrix.ndim == 1 else (0,0) )

    def test_pedigree_with_only_base_animals(self):
        pedigree = [(1, None, None), (2, None, None)]
        animal_to_idx, a_matrix, animal_list, ped_map = construct_a_matrix(pedigree)
        # Expected A:
        #   1    2
        # 1[1.0  0.0]
        # 2[0.0  1.0]
        expected_a = np.identity(2) # Order might vary, map idx1, idx2

        # Create expected_a based on actual order
        idx1_actual = animal_to_idx[1]
        idx2_actual = animal_to_idx[2]

        # This is simpler:
        manual_expected_a = np.zeros((2,2))
        manual_expected_a[idx1_actual, idx1_actual] = 1.0
        manual_expected_a[idx2_actual, idx2_actual] = 1.0
        # Off-diagonals are 0 for unrelated base animals

        np.testing.assert_array_almost_equal(a_matrix, manual_expected_a, decimal=4)

        a_inv_matrix = construct_a_inverse_matrix(a_matrix, animal_list, animal_to_idx, ped_map)
        identity_matrix = np.identity(a_matrix.shape[0])
        np.testing.assert_array_almost_equal(a_matrix @ a_inv_matrix, identity_matrix, decimal=3)

if __name__ == "__main__":
    unittest.main()
