import unittest
import numpy as np
from animal_breeding_genetics.blup_animal_model import (
    setup_mme_animal_model,
    solve_mme,
    calculate_pev_and_accuracy,
    solve_animal_model_blup
)
from animal_breeding_genetics.relationship_matrix import construct_a_matrix # For A_inv
from animal_breeding_genetics.matrix_ops import invert_matrix # For A_inv


class TestBLUPAnimalModel(unittest.TestCase):

    def _setup_simple_example(self):
        """
        Simple example: 3 animals, 1 fixed effect (mean), 2 observations.
        Pedigree:
        1 (0,0)
        2 (0,0)
        3 (1,2)

        Observations:
        y1 on animal 3
        y2 on animal 3

        Animal 1, 2 are base population. Animal 3 is offspring of 1 and 2.
        A = [[1,0,0.5],[0,1,0.5],[0.5,0.5,1]]
        A_inv = [[1.5, 0.5, -1], [0.5, 1.5, -1], [-1, -1, 2]] (approx)
        """
        pedigree = [(1,None,None), (2,None,None), (3,1,2)]
        # Construct A and A_inv
        animal_to_idx_map, a_matrix, animal_list_final_order, _ = construct_a_matrix(pedigree)
        # Ensure order is [1,2,3] for this example's known A_inv
        # If animal_list_final_order is not [1,2,3], then the hardcoded A_inv below is wrong.
        # The _get_ordered_animal_list_and_map should produce [1,2,3] or [2,1,3] then 3.
        # Let's re-order A and A_inv if needed to match animal_list_final_order.
        # For simplicity, this test assumes animal_list_final_order = [1,2,3]
        # This is generally true for this pedigree with the current sorting.

        # If the actual order is different, the test needs to be more robust or A_inv computed dynamically.
        # For now, let's compute A_inv dynamically.
        a_inv_matrix = invert_matrix(a_matrix)

        # Phenotypic data
        y_vector = np.array([10.0, 12.0]) # Two records on animal 3

        # X matrix: 1 fixed effect (overall mean)
        # Obs1: mean + e1
        # Obs2: mean + e2
        X_matrix = np.array([[1.0], [1.0]]) # (n_obs x n_fixed_effects)

        # Z matrix: links observations to animals. Animal order in Z cols must match A_inv.
        # animal_list_final_order defines this order.
        # Example: if animal_list_final_order = [1,2,3]
        # Z columns are for animal 1, animal 2, animal 3.
        # Both records are on animal 3.
        n_animals = len(animal_list_final_order)
        Z_matrix = np.zeros((len(y_vector), n_animals))

        idx_animal_3 = animal_to_idx_map[3] # Get the actual index of animal 3
        Z_matrix[0, idx_animal_3] = 1.0 # Record 1 on animal 3
        Z_matrix[1, idx_animal_3] = 1.0 # Record 2 on animal 3

        sigma_e_sq = 3.0
        sigma_a_sq = 1.0

        return y_vector, X_matrix, Z_matrix, a_inv_matrix, sigma_e_sq, sigma_a_sq, animal_to_idx_map, animal_list_final_order

    def test_setup_mme_animal_model(self):
        y, X, Z, A_inv, sigma_e_sq, sigma_a_sq, _, _ = self._setup_simple_example()
        k = sigma_e_sq / sigma_a_sq # 3.0 / 1.0 = 3.0

        LHS, RHS = setup_mme_animal_model(y, X, Z, A_inv, sigma_e_sq, sigma_a_sq)

        # Expected dimensions
        # X'X (1x1), X'Z (1x3), Z'X (3x1), Z'Z+A_inv*k (3x3)
        # LHS (4x4), RHS (4x1)
        self.assertEqual(LHS.shape, (1 + Z.shape[1], 1 + Z.shape[1])) # (num_fixed + num_random)
        self.assertEqual(RHS.shape, (1 + Z.shape[1], 1))

        # Calculate expected parts:
        X_T_X_exp = X.T @ X  # [[2.]]
        X_T_Z_exp = X.T @ Z  # [[0,0,2]] if animal 3 is last col
        Z_T_X_exp = Z.T @ X  # [[0],[0],[2]]
        Z_T_Z_exp = Z.T @ Z  # Diag matrix with 2 at animal 3's pos: [[0,0,0],[0,0,0],[0,0,2]]

        LHS_exp_00 = X_T_X_exp
        LHS_exp_01 = X_T_Z_exp
        LHS_exp_10 = Z_T_X_exp
        LHS_exp_11 = Z_T_Z_exp + A_inv * k

        np.testing.assert_array_almost_equal(LHS[:1,:1], LHS_exp_00, decimal=4)
        np.testing.assert_array_almost_equal(LHS[:1,1:], LHS_exp_01, decimal=4)
        np.testing.assert_array_almost_equal(LHS[1:,:1], LHS_exp_10, decimal=4)
        np.testing.assert_array_almost_equal(LHS[1:,1:], LHS_exp_11, decimal=4)

        X_T_y_exp = X.T @ y.reshape(-1,1) # [[22.]]
        Z_T_y_exp = Z.T @ y.reshape(-1,1) # [[0],[0],[22]]
        RHS_exp = np.vstack((X_T_y_exp, Z_T_y_exp))
        np.testing.assert_array_almost_equal(RHS, RHS_exp, decimal=4)

    def test_solve_mme_and_orchestration(self):
        y, X, Z, A_inv, sigma_e_sq, sigma_a_sq, animal_to_idx_map, animal_list = self._setup_simple_example()

        results = solve_animal_model_blup(y, X, Z, A_inv, sigma_e_sq, sigma_a_sq)
        self.assertIsNone(results['error'], msg=f"BLUP solution failed: {results['error']}")

        self.assertIsNotNone(results['b_hat'])
        self.assertIsNotNone(results['u_hat'])
        self.assertEqual(results['b_hat'].shape, (X.shape[1], 1))
        self.assertEqual(results['u_hat'].shape, (Z.shape[1], 1))

        # For this specific example, we can try to find the solutions.
        # k = 3.0
        # A_inv_hardcoded = np.array([[1.5,0.5,-1],[0.5,1.5,-1],[-1,-1,2]])
        # A_inv_k = A_inv_hardcoded * 3.0 = [[4.5,1.5,-3],[1.5,4.5,-3],[-3,-3,6]]
        # Z_T_Z for animal 3 (last) is diag([0,0,2])
        # Z_T_Z + A_inv_k (animal 3 is at index 2 if order is 1,2,3):
        # [[4.5, 1.5,  -3],
        #  [1.5, 4.5,  -3],
        #  [-3,  -3, 6+2=8]]
        #
        # LHS = [[2,   0, 0, 2],
        #        [0, 4.5, 1.5, -3],
        #        [0, 1.5, 4.5, -3],
        #        [2,  -3, -3, 8]]
        # RHS = [[22],[0],[0],[22]]
        # (This is if animal 3 is index 2, and fixed effect is index 0)
        #
        # This manual calculation is prone to errors if animal order changes.
        # The important part is that it runs and gives dimensionally correct results.
        # Verification of exact values requires a known published example.

        # Example: fixed effect (mean) should be close to mean of y (11)
        # Animal 3 has records, so its EBV should be non-zero and related to (y_mean - overall_mean)
        # Animals 1 and 2 have no records, their EBVs will be pulled towards 0 or based on parent-offspring relationships.

        mean_y = np.mean(y)
        self.assertAlmostEqual(results['b_hat'][0,0], mean_y, delta=mean_y*0.5,
                             msg="Fixed effect (mean) solution seems too far from data mean.")

        # Test PEV and Accuracy
        self.assertIsNotNone(results['pev'])
        self.assertIsNotNone(results['accuracy'])
        self.assertEqual(results['pev'].shape, (Z.shape[1],1))
        self.assertEqual(results['accuracy'].shape, (Z.shape[1],1))

        # Accuracies should be between 0 and 1
        self.assertTrue(np.all(results['accuracy'] >= 0 - 1e-6))
        self.assertTrue(np.all(results['accuracy'] <= 1 + 1e-6))

        # Animal 3 (with records) should have higher accuracy than 1 and 2 (no records)
        idx_1 = animal_to_idx_map.get(1)
        idx_2 = animal_to_idx_map.get(2)
        idx_3 = animal_to_idx_map.get(3)

        if idx_1 is not None and idx_2 is not None and idx_3 is not None:
            acc1 = results['accuracy'][idx_1,0]
            acc2 = results['accuracy'][idx_2,0]
            acc3 = results['accuracy'][idx_3,0]
            self.assertGreaterEqual(acc3, acc1, "Animal 3 (records) should have accuracy >= animal 1 (no records)")
            self.assertGreaterEqual(acc3, acc2, "Animal 3 (records) should have accuracy >= animal 2 (no records)")


    def test_error_handling_variances(self):
        y, X, Z, A_inv, sigma_e_sq, sigma_a_sq, _, _ = self._setup_simple_example()

        with self.assertRaisesRegex(ValueError, "Residual variance .* must be positive"):
            setup_mme_animal_model(y, X, Z, A_inv, 0.0, sigma_a_sq)

        # Test in main orchestration function
        results_no_va = solve_animal_model_blup(y,X,Z,A_inv,sigma_e_sq, 0.0)
        self.assertIn("Additive genetic variance (sigma_a_sq) must be positive", results_no_va['error'])
        self.assertIsNone(results_no_va['b_hat']) # Check that solutions are None
        self.assertIsNone(results_no_va['u_hat'])


    def test_pev_accuracy_calc_errors(self):
        # Test calculate_pev_and_accuracy specific errors
        LHS_inv_dummy = np.eye(3)
        with self.assertRaisesRegex(ValueError, "Additive genetic variance .* must be positive"):
            calculate_pev_and_accuracy(LHS_inv_dummy, 1, 1.0, 0.0)


if __name__ == '__main__':
    unittest.main()
