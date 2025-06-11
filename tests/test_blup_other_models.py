import unittest
import numpy as np
from animal_breeding_genetics.blup_other_models import (
    setup_mme_sire_model, solve_sire_model_blup,
    setup_mme_repeatability_model, solve_repeatability_model_blup,
    setup_mme_common_env_model, solve_common_env_model_blup
)
from animal_breeding_genetics.relationship_matrix import construct_a_matrix
from animal_breeding_genetics.matrix_ops import invert_matrix

class TestBLUPOtherModels(unittest.TestCase):

    def _setup_sire_model_example(self):
        """ Simple Sire Model example """
        # Pedigree for Sires (e.g., 2 unrelated sires)
        sire_pedigree = [(101, None, None), (102, None, None)]
        s_map, s_A, s_list, _ = construct_a_matrix(sire_pedigree)
        A_s_inv = invert_matrix(s_A)

        # Data: 5 progeny from sire 101, 5 from sire 102
        # y: observations on progeny
        # X: fixed effect (e.g., overall mean)
        # Z_sire: links progeny to sires
        y_vector = np.array([10,11,9,12,10, 15,14,16,13,15]).reshape(-1,1) # 10 progeny
        X_matrix = np.ones((10,1)) # Overall mean

        Z_sire_matrix = np.zeros((10, len(s_list)))
        idx_s101 = s_map[101]
        idx_s102 = s_map[102]
        Z_sire_matrix[0:5, idx_s101] = 1 # First 5 progeny from sire 101
        Z_sire_matrix[5:10, idx_s102] = 1 # Next 5 progeny from sire 102

        sigma_e_sq = 4.0
        sigma_sire_sq = 1.0 # Variance of sire effects

        return y_vector, X_matrix, Z_sire_matrix, A_s_inv, sigma_e_sq, sigma_sire_sq, s_map, s_list

    def test_sire_model_setup(self):
        y,X,Zs,As_inv,se,ss,_,_ = self._setup_sire_model_example()
        LHS, RHS = setup_mme_sire_model(y,X,Zs,As_inv,se,ss)

        num_fixed = X.shape[1]
        num_sires = Zs.shape[1]
        self.assertEqual(LHS.shape, (num_fixed + num_sires, num_fixed + num_sires))
        self.assertEqual(RHS.shape, (num_fixed + num_sires, 1))
        # Basic check: X'X part
        np.testing.assert_array_almost_equal(LHS[:num_fixed, :num_fixed], X.T @ X)

    def test_sire_model_solve(self):
        y,X,Zs,As_inv,se,ss,s_map,s_list = self._setup_sire_model_example()
        results = solve_sire_model_blup(y,X,Zs,As_inv,se,ss)

        self.assertIsNone(results['error'], msg=f"Sire model solution failed: {results['error']}")
        self.assertIsNotNone(results['b_hat'])
        self.assertIsNotNone(results['s_hat'])
        self.assertEqual(results['s_hat'].shape, (len(s_list),1))
        self.assertIsNotNone(results['pev_sires'])
        self.assertIsNotNone(results['accuracy_sires'])
        self.assertTrue(np.all((results['accuracy_sires'] >= 0) & (results['accuracy_sires'] <= 1)))


    def _setup_repeatability_model_example(self):
        """ Animal model with PE - 2 animals, 2 records each """
        pedigree = [(1,None,None), (2,None,None)] # Base animals
        a_map, a_matrix, a_list, _ = construct_a_matrix(pedigree)
        A_inv = invert_matrix(a_matrix)

        y = np.array([10,12, 15,14]).reshape(-1,1) # A1 rec1, A1 rec2, A2 rec1, A2 rec2
        X = np.ones((4,1)) # Overall mean

        # Z_animal links records to animals
        Z_animal = np.zeros((4, len(a_list)))
        idx_a1 = a_map[1]
        idx_a2 = a_map[2]
        Z_animal[0:2, idx_a1] = 1
        Z_animal[2:4, idx_a2] = 1

        # Z_pe links records to PE effect (animal itself if PE is animal specific)
        Z_pe = Z_animal.copy()

        sigma_e_sq = 2.0
        sigma_a_sq = 1.5
        sigma_pe_sq = 1.0
        return y,X,Z_animal,Z_pe,A_inv,sigma_e_sq,sigma_a_sq,sigma_pe_sq, a_map, a_list

    def test_repeatability_model_setup(self):
        y,X,Za,Zpe,A_inv,sigma_e_sq,sa,spe,_,_ = self._setup_repeatability_model_example()
        LHS, RHS = setup_mme_repeatability_model(y,X,Za,Zpe,A_inv,sigma_e_sq,sa,spe)

        nF = X.shape[1]
        nA = Za.shape[1]
        nPE = Zpe.shape[1]
        self.assertEqual(LHS.shape, (nF+nA+nPE, nF+nA+nPE))
        self.assertEqual(RHS.shape, (nF+nA+nPE, 1))
        np.testing.assert_array_almost_equal(LHS[:nF,:nF], X.T @ X) # X'X block

    def test_repeatability_model_solve(self):
        y,X,Za,Zpe,A_inv,sigma_e_sq,sa,spe,a_map,a_list = self._setup_repeatability_model_example()
        results = solve_repeatability_model_blup(y,X,Za,Zpe,A_inv,sigma_e_sq,sa,spe)

        self.assertIsNone(results['error'], msg=f"Repeatability model solution failed: {results['error']}")
        self.assertIsNotNone(results['b_hat'])
        self.assertIsNotNone(results['u_hat'])
        self.assertIsNotNone(results['pe_hat'])
        self.assertEqual(results['u_hat'].shape, (len(a_list),1))
        self.assertEqual(results['pe_hat'].shape, (Zpe.shape[1],1))
        self.assertIsNotNone(results['accuracy_u'])
        self.assertIsNotNone(results['accuracy_pe'])
        self.assertTrue(np.all((results['accuracy_u'] >= 0) & (results['accuracy_u'] <= 1)))
        self.assertTrue(np.all((results['accuracy_pe'] >= 0) & (results['accuracy_pe'] <= 1)))

    # Common Env model tests would be very similar to Repeatability model
    # Just need to set up Z_common_env and sigma_common_env_sq
    def _setup_common_env_model_example(self):
        """ Animal model with common env (e.g. litter) """
        pedigree = [(1,None,None), (2,None,None), (3,1,None), (4,1,None), (5,2,None)] # 3 offspring of A1, 1 of A2
        a_map, a_matrix, a_list, _ = construct_a_matrix(pedigree)
        A_inv = invert_matrix(a_matrix)

        # y: A3, A4, A5 records
        y = np.array([10,11, 15]).reshape(-1,1)
        X = np.ones((3,1)) # Overall mean

        Z_animal = np.zeros((3, len(a_list)))
        Z_animal[0, a_map[3]] = 1
        Z_animal[1, a_map[4]] = 1
        Z_animal[2, a_map[5]] = 1

        # Assume A3,A4 from litter 1; A5 from litter 2
        # Litter incidence matrix (n_obs x n_litters)
        # Litters L1, L2
        Z_litter = np.array([[1,0],[1,0],[0,1]])

        sigma_e_sq = 2.0
        sigma_a_sq = 1.0
        sigma_litter_sq = 0.5
        return y,X,Z_animal,Z_litter,A_inv,sigma_e_sq,sigma_a_sq,sigma_litter_sq, a_map, a_list

    def test_common_env_model_setup(self):
        y,X,Za,Zc,A_inv,se,sa,sc,_,_ = self._setup_common_env_model_example()
        LHS, RHS = setup_mme_common_env_model(y,X,Za,Zc,A_inv,se,sa,sc)

        nF = X.shape[1]
        nA = Za.shape[1]
        nC = Zc.shape[1]
        self.assertEqual(LHS.shape, (nF+nA+nC, nF+nA+nC))
        self.assertEqual(RHS.shape, (nF+nA+nC, 1))

    def test_common_env_model_solve(self):
        y,X,Za,Zc,A_inv,se,sa,sc,a_map,a_list = self._setup_common_env_model_example()
        results = solve_common_env_model_blup(y,X,Za,Zc,A_inv,se,sa,sc)

        self.assertIsNone(results['error'], msg=f"Common Env model solution failed: {results['error']}")
        self.assertIsNotNone(results['b_hat'])
        self.assertIsNotNone(results['u_hat'])
        self.assertIsNotNone(results['c_hat']) # Common env effects
        self.assertEqual(results['u_hat'].shape, (len(a_list),1))
        self.assertEqual(results['c_hat'].shape, (Zc.shape[1],1))
        self.assertIsNotNone(results['accuracy_u'])
        self.assertIsNotNone(results['accuracy_c'])


if __name__ == "__main__":
    unittest.main()
