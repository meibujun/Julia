import unittest
import math
from statistical_genomics.statistical_genomics.population_genetics.hwe import calculate_hwe_chi_squared
# Ensure scipy is available, or tests requiring it might need to be skipped
# For this environment, assume it can be installed by the system if needed.

class TestHWE(unittest.TestCase):

    def test_hwe_perfect_equilibrium(self):
        # p=0.5, q=0.5, N=100. Exp AA=25, Aa=50, aa=25
        observed = {"AA": 25, "Aa": 50, "aa": 25}
        chi_sq, p_val = calculate_hwe_chi_squared(observed)
        self.assertAlmostEqual(chi_sq, 0.0, places=5)
        self.assertAlmostEqual(p_val, 1.0, places=5)

        # p=0.7, q=0.3, N=200
        # Exp AA = 0.7^2 * 200 = 0.49 * 200 = 98
        # Exp Aa = 2 * 0.7 * 0.3 * 200 = 0.42 * 200 = 84
        # Exp aa = 0.3^2 * 200 = 0.09 * 200 = 18
        observed2 = {"AA": 98, "Aa": 84, "aa": 18}
        chi_sq2, p_val2 = calculate_hwe_chi_squared(observed2)
        self.assertAlmostEqual(chi_sq2, 0.0, places=5)
        self.assertAlmostEqual(p_val2, 1.0, places=5)

    def test_hwe_clear_disequilibrium(self):
        # p=0.5, q=0.5, N=100. Exp AA=25, Aa=50, aa=25
        # Obs AA=50, Aa=0, aa=50
        # Chi_sq = (50-25)^2/25 + (0-50)^2/50 + (50-25)^2/25
        #        = 625/25 + 2500/50 + 625/25
        #        = 25 + 50 + 25 = 100
        observed = {"AA": 50, "Aa": 0, "aa": 50}
        chi_sq, p_val = calculate_hwe_chi_squared(observed)
        self.assertAlmostEqual(chi_sq, 100.0, places=5)
        self.assertTrue(p_val < 0.00001) # p-value for chi_sq=100, df=1 is extremely small

    def test_hwe_standard_example(self):
        # AA=290, Aa=420, aa=290, N=1000
        # p(A) = (2*290 + 420)/(2*1000) = (580+420)/2000 = 1000/2000 = 0.5. q(a)=0.5.
        # Exp AA = 0.5^2*1000 = 250.
        # Exp Aa = 2*0.5*0.5*1000 = 500.
        # Exp aa = 0.5^2*1000 = 250.
        # Chi_sq = (290-250)^2/250 + (420-500)^2/500 + (290-250)^2/250
        #        = 1600/250 + 6400/500 + 1600/250
        #        = 6.4 + 12.8 + 6.4 = 25.6
        observed = {"AA": 290, "Aa": 420, "aa": 290}
        chi_sq, p_val = calculate_hwe_chi_squared(observed)
        self.assertAlmostEqual(chi_sq, 25.6, places=5)
        self.assertTrue(p_val < 0.00001) # p-value for chi_sq=25.6, df=1 is very small

    def test_hwe_zero_observed_genotype(self):
        # AA=0, Aa=50, aa=50, N=100
        # p(A) = (2*0 + 50)/(2*100) = 50/200 = 0.25
        # q(a) = (2*50 + 50)/(2*100) = 150/200 = 0.75
        # Exp AA = (0.25)^2 * 100 = 0.0625 * 100 = 6.25
        # Exp Aa = 2 * 0.25 * 0.75 * 100 = 0.375 * 100 = 37.5
        # Exp aa = (0.75)^2 * 100 = 0.5625 * 100 = 56.25
        # Chi_sq = (0-6.25)^2/6.25 + (50-37.5)^2/37.5 + (50-56.25)^2/56.25
        #        = 39.0625/6.25 + 156.25/37.5 + 39.0625/56.25
        #        = 6.25 + 4.166666... + 0.694444...
        #        = 11.111111...
        observed = {"AA": 0, "Aa": 50, "aa": 50}
        chi_sq, p_val = calculate_hwe_chi_squared(observed)
        self.assertAlmostEqual(chi_sq, 11.11111111, places=5)
        # p-value for 11.11, df=1: scipy.stats.chi2.sf(11.111111, 1) approx 0.00085
        self.assertTrue(p_val < 0.001 and p_val > 0.0005)


    def test_hwe_zero_expected_genotype(self):
        # Fixation: p=1, q=0. Observed AA=100, Aa=0, aa=0. N=100.
        # Exp AA = 1^2 * 100 = 100
        # Exp Aa = 2 * 1 * 0 * 100 = 0
        # Exp aa = 0^2 * 100 = 0
        # Rule: if any expected count is <= 0 (or very small), return (0.0, 1.0)
        observed_A_fix = {"AA": 100} # Implies Aa=0, aa=0 based on allele inference
        chi_sq_A, p_val_A = calculate_hwe_chi_squared(observed_A_fix)
        self.assertAlmostEqual(chi_sq_A, 0.0, places=5, msg="Fixation of A allele")
        self.assertAlmostEqual(p_val_A, 1.0, places=5, msg="Fixation of A allele")

        observed_a_fix = {"aa": 50} # Implies AA=0, Aa=0
        chi_sq_a, p_val_a = calculate_hwe_chi_squared(observed_a_fix)
        self.assertAlmostEqual(chi_sq_a, 0.0, places=5, msg="Fixation of a allele")
        self.assertAlmostEqual(p_val_a, 1.0, places=5, msg="Fixation of a allele")

        # Case where observed doesn't match fixation, but E=0 still triggers rule
        # e.g. p=1, q=0 for allele 'A'. Observed: AA=90, Aa=10, aa=0. N=100
        # Allele 'A' freq p_A = (2*90 + 10)/200 = 190/200 = 0.95
        # Allele 'a' freq p_a = (0*0 + 10)/200 = 10/200 = 0.05
        # Exp AA = 0.95^2 * 100 = 90.25
        # Exp Aa = 2 * 0.95 * 0.05 * 100 = 9.5
        # Exp aa = 0.05^2 * 100 = 0.25
        # This case should NOT return (0.0, 1.0) by E=0 rule unless N is tiny.
        # The E=0 rule in the function applies if p or q is exactly 0.
        # If p_A=1 from observed counts (e.g. {"AA":100, "Aa":0, "aa":0}), then q_a=0.
        # Exp_Aa = 0, Exp_aa = 0. Rule applies.
        observed_mismatch_fix = {"AA": 100, "Aa": 0, "aa": 0} # This IS perfect fixation.
        chi_sq_m, p_val_m = calculate_hwe_chi_squared(observed_mismatch_fix)
        self.assertAlmostEqual(chi_sq_m, 0.0, places=5)
        self.assertAlmostEqual(p_val_m, 1.0, places=5)


    def test_hwe_empty_input(self):
        with self.assertRaisesRegex(ValueError, "Total number of individuals \\(N\\) must be greater than 0."):
            calculate_hwe_chi_squared({})

        # Test with counts that sum to zero
        with self.assertRaisesRegex(ValueError, "Total number of individuals \\(N\\) must be greater than 0."):
            calculate_hwe_chi_squared({"AA":0, "Aa":0, "aa":0})

    def test_hwe_monoallelic_input_edge_case(self):
        # Test with only one type of allele present, e.g. all "AA"
        # This is fixation. p=1, q=0. Expected AA=N, Aa=0, aa=0. ChiSq should be 0.
        observed = {"AA": 50}
        chi_sq, p_val = calculate_hwe_chi_squared(observed)
        self.assertAlmostEqual(chi_sq, 0.0, places=5)
        self.assertAlmostEqual(p_val, 1.0, places=5)

        observed_other_allele = {"BB": 30} # Using B to show flexibility
        chi_sq_B, p_val_B = calculate_hwe_chi_squared(observed_other_allele)
        self.assertAlmostEqual(chi_sq_B, 0.0, places=5)
        self.assertAlmostEqual(p_val_B, 1.0, places=5)

    def test_hwe_only_heterozygotes(self):
        # E.g. {"Aa": 50}. N=50.
        # Alleles 'A', 'a'. p_A = 0.5, p_a = 0.5
        # Exp AA = 0.5^2 * 50 = 12.5
        # Exp Aa = 2*0.5*0.5*50 = 25
        # Exp aa = 0.5^2 * 50 = 12.5
        # ChiSq = (0-12.5)^2/12.5 + (50-25)^2/25 + (0-12.5)^2/12.5
        #       = 12.5 + 625/25 + 12.5
        #       = 12.5 + 25 + 12.5 = 50
        observed = {"Aa": 50}
        chi_sq, p_val = calculate_hwe_chi_squared(observed)
        self.assertAlmostEqual(chi_sq, 50.0, places=5)
        self.assertTrue(p_val < 0.00001)

if __name__ == '__main__':
    unittest.main()
