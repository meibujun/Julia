import unittest
import numpy as np
from animal_breeding_genetics.selection_index import (
    calculate_selection_index_b_values,
    calculate_selection_index_accuracy,
    index_own_performance,
    index_progeny_records,
    index_own_and_full_sib_mean,
    economic_selection_index,
)
from animal_breeding_genetics.matrix_ops import invert_matrix, multiply_matrices # For P_inv in tests if needed

class TestSelectionIndex(unittest.TestCase):

    def test_calculate_b_values_simple(self):
        P_matrix = np.array([[10.0]])
        G_vector = np.array([[4.0]])
        # b = P_inv @ G = (1/10) * 4 = 0.4
        expected_b = np.array([[0.4]])
        np.testing.assert_array_almost_equal(
            calculate_selection_index_b_values(P_matrix, G_vector), expected_b, decimal=4
        )

        P_matrix_2x2 = np.array([[10, 2], [2, 5]]) # P_inv = (1/46) * [[5,-2],[-2,10]]
        G_vector_2x1 = np.array([[4], [1]])      # b = P_inv @ G = (1/46) * [[5*4-2*1],[-2*4+10*1]] = (1/46) * [[18],[2]]
        expected_b_2x1 = (1/46) * np.array([[18.0], [2.0]])
        np.testing.assert_array_almost_equal(
            calculate_selection_index_b_values(P_matrix_2x2, G_vector_2x1), expected_b_2x1, decimal=4
        )

    def test_calculate_accuracy_simple(self):
        # b'G / var_H
        b_vector = np.array([[0.4]])
        G_vector = np.array([[4.0]])
        var_H = 4.0
        # num = 0.4 * 4.0 = 1.6.  acc_sq = 1.6 / 4.0 = 0.4. acc = sqrt(0.4) = 0.6325
        expected_acc = np.sqrt(0.4)
        self.assertAlmostEqual(
            calculate_selection_index_accuracy(b_vector, G_vector, var_H), expected_acc, places=4
        )

        # Test with var_H = 0
        self.assertAlmostEqual(calculate_selection_index_accuracy(b_vector, G_vector, 0.0), 0.0, places=4)


    def test_index_own_performance(self):
        h2 = 0.25
        pheno_sd = 2.0
        # PV = 4.0, GV = 0.25 * 4.0 = 1.0
        # P = [[4]], G = [[1]], var_H = 1
        # b = (1/4)*1 = 0.25
        # acc_sq = (0.25*1)/1 = 0.25. acc = sqrt(0.25) = 0.5 (which is sqrt(h2))
        b, acc = index_own_performance(h2, pheno_sd)
        self.assertAlmostEqual(b[0,0], 0.25, places=4)
        self.assertAlmostEqual(acc, np.sqrt(h2), places=4)

        # Test h2 = 0
        b_h0, acc_h0 = index_own_performance(0.0, pheno_sd)
        self.assertAlmostEqual(b_h0[0,0], 0.0, places=4)
        self.assertAlmostEqual(acc_h0, 0.0, places=4)

        # Test pheno_sd = 0
        b_sd0, acc_sd0 = index_own_performance(h2, 0.0)
        self.assertAlmostEqual(b_sd0[0,0], 0.0, places=4) # PV = 0 -> b=0
        self.assertAlmostEqual(acc_sd0, 0.0, places=4)   # PV = 0 -> acc=0

    def test_index_progeny_records(self):
        n = 25
        h2 = 0.36
        pheno_sd = 10.0
        # PV = 100, GV = 36
        # t_hs_pheno = 0.25 * 0.36 = 0.09
        # Var(prog_mean) = (100/25) * (1 + (24 * 0.09)) = 4 * (1 + 2.16) = 4 * 3.16 = 12.64
        # P = [[12.64]]
        # G = Cov(Sire_BV, Prog_Mean) = 0.5 * GV = 0.5 * 36 = 18
        # G = [[18]]
        # var_H = GV = 36
        # b = (1/12.64) * 18 = 1.42405
        # acc_sq = (1.42405 * 18) / 36 = 1.42405 * 0.5 = 0.712025. acc = sqrt(0.712025) = 0.8438
        # Standard formula for progeny test accuracy: sqrt( (n*r^2) / (1 + (n-1)t) ) where r=0.5, t=0.25h2
        # acc_PT = sqrt( (n * (0.5^2)) / (1 + (n-1) * 0.25*h2) )
        # acc_PT = sqrt( (25*0.25) / (1 + 24*0.25*0.36) ) = sqrt(6.25 / (1 + 2.16)) = sqrt(6.25 / 3.16) = sqrt(1.9778) = 1.406... this is not accuracy.
        # Accuracy formula for progeny testing is r_IH = r_gs * sqrt( n / (1+(n-1)t_pheno_sib) ) where r_gs is sire-offspring (0.5)
        # r_IH = 0.5 * sqrt( n_progeny / (1 + (n_progeny-1) * (0.25*h_squared + 0 * h_squared_dam_effect)) )
        # This is for accuracy of EBV of sire based on progeny mean.
        # The Hazel (1943) formula for accuracy of progeny test: sqrt( (n*h2) / (4*(1-h2) + n*h2) ) for unrelated dams.
        # acc_hazel = np.sqrt( (n*h2) / (4*(1-h2) + n*h2) ) = np.sqrt( (25*0.36) / (4*0.64 + 25*0.36) )
        #           = np.sqrt( 9 / (2.56 + 9) ) = np.sqrt(9 / 11.56) = np.sqrt(0.7785) = 0.8823

        b, acc = index_progeny_records(n, h2, pheno_sd)
        # Using my derived b and acc:
        self.assertAlmostEqual(b[0,0], 18.0/12.64, places=4)
        self.assertAlmostEqual(acc, np.sqrt( ( (18.0/12.64)*18 )/36 ), places=4)
        # Compare with standard progeny test accuracy formula: acc_sq = n / (n + k) where k = (4-h^2)/h^2
        # acc_sq = (n*h2) / (n*h2 + 4 - h2) = (n*h2) / (4 + (n-1)*h2)
        k_factor = (4 - h2) / h2 if h2 > 0 else float('inf')
        if h2 == 0:
            expected_acc_sq_progeny = 0.0
        else:
            expected_acc_sq_progeny = n / (n + k_factor)

        expected_acc_progeny = np.sqrt(expected_acc_sq_progeny)
        self.assertAlmostEqual(acc, expected_acc_progeny, places=4, msg="Accuracy differs from standard progeny test formula")


    def test_index_own_and_full_sib_mean(self):
        h2 = 0.5
        pheno_sd = 1.0
        n_sibs = 10
        c2 = 0.1
        # PV = 1.0, GV = 0.5, Var_Ec = 0.1
        # P11 = 1.0
        # t_fs_pheno = 0.5*h2 + c2 = 0.5*0.5 + 0.1 = 0.25 + 0.1 = 0.35
        # P22 = (1/10) * (1 + (9 * 0.35)) = 0.1 * (1 + 3.15) = 0.1 * 4.15 = 0.415
        # P12 = 0.5*GV + Var_Ec = 0.5*0.5 + 0.1 = 0.25 + 0.1 = 0.35
        # P = [[1.0, 0.35], [0.35, 0.415]]
        # G = [[GV], [0.5*GV]] = [[0.5], [0.25]]
        # var_H = GV = 0.5

        P_expected = np.array([[1.0, 0.35], [0.35, 0.415]])
        G_expected = np.array([[0.5], [0.25]])
        var_H_expected = 0.5

        P_inv = invert_matrix(P_expected) # (1/(0.415-0.35^2)) * [[0.415, -0.35],[-0.35,1]]
                                        # det = 0.415 - 0.1225 = 0.2925
                                        # P_inv = (1/0.2925) * [[0.415, -0.35],[-0.35,1]]
        b_expected = multiply_matrices(P_inv, G_expected)
        # b_expected = (1/0.2925) * [[0.415*0.5 - 0.35*0.25], [-0.35*0.5 + 1*0.25]]
        #            = (1/0.2925) * [[0.2075 - 0.0875], [-0.175 + 0.25]]
        #            = (1/0.2925) * [[0.12], [0.075]]
        #            = [[0.410256], [0.256410]]

        acc_sq_expected = (multiply_matrices(b_expected.T, G_expected)[0,0]) / var_H_expected
        acc_expected = np.sqrt(acc_sq_expected) # sqrt( (0.410256*0.5 + 0.256410*0.25) / 0.5 )
                                               # sqrt( (0.205128 + 0.0641025) / 0.5)
                                               # sqrt(0.2692305 / 0.5) = sqrt(0.538461) = 0.7338

        b, acc = index_own_and_full_sib_mean(h2, pheno_sd, n_sibs, c2)
        np.testing.assert_array_almost_equal(b, b_expected, decimal=4)
        self.assertAlmostEqual(acc, acc_expected, places=4)

    def test_economic_selection_index(self):
        # Example based on simplified scenario
        # 2 traits in H, 2 info sources
        economic_weights = np.array([1.0, 0.5]) # m=2

        # Info sources I1, I2 (n=2)
        P_matrix_info = np.array([[10.0, 2.0], [2.0, 5.0]]) # Phenotypic var-cov of I1, I2

        # Traits g1, g2
        # G_matrix_traits_info: Cov(g_traits, I_sources) (m x n) = (2 x 2)
        # Row1: Cov(g1,I1), Cov(g1,I2)
        # Row2: Cov(g2,I1), Cov(g2,I2)
        G_matrix_traits_info = np.array([
            [4.0, 1.0], # Cov(g1,I1)=4, Cov(g1,I2)=1
            [0.5, 2.0]  # Cov(g2,I1)=0.5, Cov(g2,I2)=2
        ])

        # V_A_matrix_traits: Genetic var-cov of g1, g2 (m x m) = (2 x 2)
        V_A_matrix_traits = np.array([
            [2.0, 0.5], # Var(g1)=2, Cov(g1,g2)=0.5
            [0.5, 1.0]  # Cov(g2,g1)=0.5, Var(g2)=1
        ])

        # var_H = a.T @ V_A @ a = [1, 0.5] @ [[2,0.5],[0.5,1]] @ [[1],[0.5]]
        #       = [1*2+0.5*0.5, 1*0.5+0.5*1] @ [[1],[0.5]]
        #       = [2.25, 1.0] @ [[1],[0.5]] = 2.25*1 + 1.0*0.5 = 2.25 + 0.5 = 2.75
        var_H_expected = 2.75

        # C_IH = G_traits_info.T @ a = [[4,0.5],[1,2]] @ [[1],[0.5]]
        #      = [[4*1+0.5*0.5],[1*1+2*0.5]] = [[4.25],[2.0]]
        C_IH_expected = np.array([[4.25], [2.0]])

        P_inv = invert_matrix(P_matrix_info) # (1/46) * [[5,-2],[-2,10]]
        # b = P_inv @ C_IH = (1/46) * [[5,-2],[-2,10]] @ [[4.25],[2.0]]
        #   = (1/46) * [[5*4.25-2*2], [-2*4.25+10*2]]
        #   = (1/46) * [[21.25-4], [-8.5+20]] = (1/46) * [[17.25],[11.5]]
        #   = [[0.375],[0.25]]
        b_expected = np.array([[17.25/46.0], [11.5/46.0]]) # [[0.375], [0.25]]

        # acc_num = b.T @ C_IH = [0.375, 0.25] @ [[4.25],[2.0]]
        #         = 0.375*4.25 + 0.25*2.0 = 1.59375 + 0.5 = 2.09375
        acc_num_expected = (17.25/46.0)*4.25 + (11.5/46.0)*2.0

        acc_sq_expected = acc_num_expected / var_H_expected
        acc_expected = np.sqrt(acc_sq_expected)

        b, acc, var_H = economic_selection_index(
            economic_weights, P_matrix_info, G_matrix_traits_info, V_A_matrix_traits
        )

        np.testing.assert_array_almost_equal(b, b_expected, decimal=4)
        self.assertAlmostEqual(var_H, var_H_expected, places=4)
        self.assertAlmostEqual(acc, acc_expected, places=4)


    def test_error_handling(self):
        # calculate_b_values errors
        with self.assertRaises(ValueError): # Non-square P
            calculate_selection_index_b_values(np.array([[1,2,3],[4,5,6]]), np.array([[1],[1],[1]]))
        with self.assertRaises(ValueError): # Incompatible P, G
            calculate_selection_index_b_values(np.array([[1,2],[3,4]]), np.array([[1],[1],[1]]))
        with self.assertRaises(np.linalg.LinAlgError): # Singular P
            calculate_selection_index_b_values(np.array([[1,1],[1,1]]), np.array([[1],[1]]))

        # calculate_accuracy errors
        with self.assertRaises(ValueError): # var_H < 0
            calculate_selection_index_accuracy(np.array([[1]]), np.array([[1]]), -1.0)
        with self.assertRaises(ValueError): # b.T @ G < 0
            calculate_selection_index_accuracy(np.array([[-1]]), np.array([[1]]), 1.0)

        # specific index errors
        with self.assertRaises(ValueError): # h2 out of bounds
            index_own_performance(1.5, 1.0)
        with self.assertRaises(ValueError): # n_progeny <=0
            index_progeny_records(0, 0.5, 1.0)
        with self.assertRaises(ValueError): # c2 out of bounds
            index_own_and_full_sib_mean(0.5, 1.0, 10, 1.1)
        with self.assertRaises(ValueError): # h2+c2 > 1
            index_own_and_full_sib_mean(0.6,1.0,10,0.5)


if __name__ == "__main__":
    unittest.main()
