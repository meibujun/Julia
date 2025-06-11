import unittest
import numpy as np
from scipy import stats # Required for HWE test comparison, if we test p-value directly
from animal_breeding_genetics.genetic_params import (
    calculate_allele_frequencies,
    calculate_genotype_frequencies,
    check_hardy_weinberg_equilibrium,
    calculate_phenotypic_variance,
    calculate_h2_from_parent_offspring_regression,
    partition_variance,
)

class TestGeneticParams(unittest.TestCase):
    def test_calculate_allele_frequencies(self):
        genotypes1 = {"AA": 25, "Aa": 50, "aa": 25} # Total 100 individuals
        # A: (25*2 + 50*1) = 100; a: (25*2 + 50*1) = 100. Total alleles = 200
        # Freq A = 100/200 = 0.5; Freq a = 100/200 = 0.5
        expected1 = {"A": 0.5, "a": 0.5}
        self.assertEqual(calculate_allele_frequencies(genotypes1), expected1)

        genotypes2 = {"BB": 70, "Bb": 20, "bb": 10} # Total 100 individuals
        # B: (70*2 + 20*1) = 160; b: (10*2 + 20*1) = 40. Total alleles = 200
        # Freq B = 160/200 = 0.8; Freq b = 40/200 = 0.2
        expected2 = {"B": 0.8, "b": 0.2}
        self.assertEqual(calculate_allele_frequencies(genotypes2), expected2)

        genotypes3 = {"AA": 0, "Aa": 0, "aa": 0}
        expected3 = {"A": 0.0, "a": 0.0}
        self.assertEqual(calculate_allele_frequencies(genotypes3), expected3)

        genotypes4 = {"AA": 10} # Only one genotype
        expected4 = {"A": 1.0}
        self.assertEqual(calculate_allele_frequencies(genotypes4), expected4)

        genotypes5 = {"AA": 0, "AB": 0} # Alleles A and B, but count is 0
        expected5 = {"A": 0.0, "B": 0.0}
        self.assertEqual(calculate_allele_frequencies(genotypes5), expected5)


        with self.assertRaisesRegex(ValueError, "Genotypes dictionary cannot be empty."):
            calculate_allele_frequencies({})
        with self.assertRaisesRegex(ValueError, "Invalid genotype string: AAA"):
            calculate_allele_frequencies({"AAA": 10})
        with self.assertRaisesRegex(ValueError, "Count for genotype AA must be a non-negative integer."):
            calculate_allele_frequencies({"AA": -10})
        with self.assertRaisesRegex(ValueError, "More than two alleles detected"):
            calculate_allele_frequencies({"AA": 10, "AB": 5, "CC": 5})

    def test_calculate_genotype_frequencies(self):
        genotypes1 = {"AA": 25, "Aa": 50, "aa": 25} # Total 100
        expected1 = {"AA": 0.25, "Aa": 0.50, "aa": 0.25}
        self.assertEqual(calculate_genotype_frequencies(genotypes1), expected1)

        genotypes2 = {"BB": 70, "Bb": 20, "bb": 10} # Total 100
        expected2 = {"BB": 0.7, "Bb": 0.2, "bb": 0.1}
        self.assertEqual(calculate_genotype_frequencies(genotypes2), expected2)

        genotypes3 = {"AA": 0, "Aa": 0, "aa": 0}
        expected3 = {"AA": 0.0, "Aa": 0.0, "aa": 0.0}
        self.assertEqual(calculate_genotype_frequencies(genotypes3), expected3)

        with self.assertRaisesRegex(ValueError, "Genotypes dictionary cannot be empty."):
            calculate_genotype_frequencies({})
        with self.assertRaisesRegex(ValueError, "Count for genotype AA must be a non-negative integer."):
            calculate_genotype_frequencies({"AA": -10})


    def test_check_hardy_weinberg_equilibrium(self):
        # Example 1: In HWE
        # p=0.5, q=0.5. Expected: AA=0.25, Aa=0.5, aa=0.25. Counts for 100 individuals.
        observed1 = {"AA": 25, "Aa": 50, "aa": 25}
        is_hwe1, chi2_1, exp_freqs1 = check_hardy_weinberg_equilibrium(observed1)
        self.assertTrue(is_hwe1)
        self.assertAlmostEqual(chi2_1, 0.0, places=5) # Expect chi2 to be very low
        # Corrected expected frequencies for genotypes
        # The exp_freqs1 dictionary contains expected GENOTYPE frequencies.
        # For example: {'AA': 0.25, 'Aa': 0.5, 'aa': 0.25}
        self.assertAlmostEqual(exp_freqs1["AA"], 0.25)
        self.assertAlmostEqual(exp_freqs1["Aa"], 0.50)
        self.assertAlmostEqual(exp_freqs1["aa"], 0.25)


        # Example 2: Not in HWE
        # p=0.7, q=0.3. Expected: DD=0.49 (49), Dd=0.42 (42), dd=0.09 (9) for 100 individuals
        observed2 = {"DD": 60, "Dd": 20, "dd": 20} # Deviates significantly
        is_hwe2, chi2_2, exp_freqs2 = check_hardy_weinberg_equilibrium(observed2)
        self.assertFalse(is_hwe2)
        self.assertGreater(chi2_2, 5.99)  # Typical critical value for 1 df at alpha=0.01 (more stringent) or 3.84 for alpha=0.05
        # Expected allele freqs: D = (60*2 + 20)/200 = 140/200 = 0.7. d = (20*2+20)/200 = 60/200 = 0.3
        # Expected genotype freqs: DD = 0.7^2 = 0.49, Dd = 2*0.7*0.3 = 0.42, dd = 0.3^2 = 0.09
        self.assertAlmostEqual(exp_freqs2["DD"], 0.49)
        self.assertAlmostEqual(exp_freqs2["Dd"], 0.42)
        self.assertAlmostEqual(exp_freqs2["dd"], 0.09)

        # Example 3: Different alleles, in HWE
        observed3 = {"RR": 4, "Rr": 32, "rr": 64} # p=0.2 (R), q=0.8 (r)
        # Allele freqs: R = (4*2+32)/200 = 40/200 = 0.2. r = (32+64*2)/200 = 160/200 = 0.8
        # Expected: RR = 0.04 (4), Rr = 0.32 (32), rr = 0.64 (64)
        is_hwe3, chi2_3, exp_freqs3 = check_hardy_weinberg_equilibrium(observed3)
        self.assertTrue(is_hwe3)
        self.assertAlmostEqual(chi2_3, 0.0, places=5)
        self.assertAlmostEqual(exp_freqs3["RR"], 0.04)
        self.assertAlmostEqual(exp_freqs3["Rr"], 0.32)
        self.assertAlmostEqual(exp_freqs3["rr"], 0.64)

        # Test case with genotype "aA" to ensure it's handled as "Aa"
        observed4 = {"AA": 25, "aA": 50, "aa": 25}
        is_hwe4, _, _ = check_hardy_weinberg_equilibrium(observed4)
        self.assertTrue(is_hwe4)

        # Test with zero count for one genotype
        observed5 = {"BB": 50, "Bb": 0, "bb": 50} # p=0.5, q=0.5. Expected Bb=50.
        is_hwe5, chi2_5, exp_freqs5 = check_hardy_weinberg_equilibrium(observed5)
        self.assertFalse(is_hwe5) # Should not be in HWE
        self.assertAlmostEqual(exp_freqs5["BB"], 0.25)
        self.assertAlmostEqual(exp_freqs5["Bb"], 0.50)
        self.assertAlmostEqual(exp_freqs5["bb"], 0.25)


        with self.assertRaisesRegex(ValueError, "Observed genotypes dictionary must contain at least two genotypes."):
            check_hardy_weinberg_equilibrium({"AA": 100})
        with self.assertRaisesRegex(ValueError, "Total number of observed individuals cannot be zero."):
            check_hardy_weinberg_equilibrium({"AA": 0, "Aa": 0, "aa": 0})
        with self.assertRaisesRegex(ValueError, "Invalid genotype string: A"):
            check_hardy_weinberg_equilibrium({"A": 10, "B": 20, "C":30})
        with self.assertRaisesRegex(ValueError, "Hardy-Weinberg equilibrium check requires exactly two alleles."):
            check_hardy_weinberg_equilibrium({"AB": 10, "BC": 20, "CA":30}) # 3 alleles

        # Test case where one genotype is missing (count implicitly zero)
        # Input: {"AA": 10, "Aa": 20} (total 30 individuals)
        # Alleles: A, a. Counts: AA=10, Aa=20, aa=0
        # Freq A (p) = (10*2 + 20) / (30*2) = 40/60 = 2/3
        # Freq a (q) = 20 / 60 = 1/3
        # Expected HWE counts (N=30):
        # AA = p^2 * N = (2/3)^2 * 30 = (4/9)*30 = 13.333
        # Aa = 2pq * N = 2 * (2/3) * (1/3) * 30 = (4/9)*30 = 13.333
        # aa = q^2 * N = (1/3)^2 * 30 = (1/9)*30 = 3.333
        observed6 = {"AA": 10, "Aa": 20}
        is_hwe6, chi2_6, exp_freqs6 = check_hardy_weinberg_equilibrium(observed6)
        self.assertFalse(is_hwe6) # Should deviate from HWE
        self.assertAlmostEqual(exp_freqs6["AA"], (2/3)**2, places=5)
        self.assertAlmostEqual(exp_freqs6["Aa"], 2*(2/3)*(1/3), places=5)
        self.assertAlmostEqual(exp_freqs6["aa"], (1/3)**2, places=5)
        # Chi-squared: obs=[10,20,0], exp=[13.333, 13.333, 3.333]
        # (10-13.333)^2/13.333 = (-3.333)^2/13.333 = 11.10889/13.333 = 0.833
        # (20-13.333)^2/13.333 = (6.667)^2/13.333 = 44.44889/13.333 = 3.333
        # (0-3.333)^2/3.333 = (-3.333)^2/3.333 = 11.10889/3.333 = 3.333
        # chi2 = 0.833 + 3.333 + 3.333 = 7.499 (approx)
        # Using scipy for a more precise chi2 value if needed for the test assertion:
        obs_values = np.array([10, 20, 0])
        exp_values = np.array([30*(2/3)**2, 30*2*(2/3)*(1/3), 30*(1/3)**2])
        chi2_expected_val, _ = stats.chisquare(f_obs=obs_values, f_exp=exp_values, ddof=1)
        self.assertAlmostEqual(chi2_6, chi2_expected_val, places=5)


    def test_calculate_phenotypic_variance(self):
        phenotypes1 = [10, 12, 11, 9, 13] # Var = 2.5
        self.assertAlmostEqual(calculate_phenotypic_variance(phenotypes1), 2.5)

        phenotypes2 = [5.0, 5.0, 5.0, 5.0] # Var = 0
        self.assertAlmostEqual(calculate_phenotypic_variance(phenotypes2), 0.0)

        with self.assertRaisesRegex(TypeError, "Phenotypes input must be a list."):
            calculate_phenotypic_variance("not a list")
        with self.assertRaisesRegex(ValueError, "Phenotypes list cannot be empty."):
            calculate_phenotypic_variance([])
        with self.assertRaisesRegex(ValueError, "All phenotypic values must be numeric."):
            calculate_phenotypic_variance([10, "a", 12])
        with self.assertRaisesRegex(ValueError, "Phenotypic variance calculation requires at least two data points."):
            calculate_phenotypic_variance([10])


    def test_calculate_h2_from_parent_offspring_regression(self):
        # Perfect heritability example: offspring = 0.5 * parent (assuming intercept 0 for simplicity of b_op)
        # if parent = [1,2,3,4], offspring = [0.5, 1, 1.5, 2], b_op = 0.5, h2 = 2 * 0.5 = 1.0
        parents1 = [10, 12, 14, 16, 18]
        offspring1 = [5, 6, 7, 8, 9] # b_op = 0.5
        self.assertAlmostEqual(calculate_h2_from_parent_offspring_regression(parents1, offspring1), 1.0, places=5)

        # No heritability example: offspring values are random relative to parent
        parents2 = [10, 12, 14, 16, 18]
        offspring2 = [7, 5, 8, 6, 9] # Expect b_op close to 0
        # Note: with few data points, b_op might not be exactly 0.
        # For this test, let's use values that give a known slope.
        # If y = c (constant), slope is 0.
        offspring2_flat = [7, 7, 7, 7, 7]
        self.assertAlmostEqual(calculate_h2_from_parent_offspring_regression(parents2, offspring2_flat), 0.0, places=5)

        # Example from a source: Parents: 10,11,12,13,14. Offspring: 5,7,6,8,9
        # Slope (b_op) is approx 0.8. h2 = 2 * 0.8 = 1.6 (can be > 1 due to sampling)
        # Or, let's use a cleaner example where b_op = 0.25 => h2 = 0.5
        parents3 = [10, 11, 12, 13, 14, 15]
        offspring3 = [2.5, 2.75, 3.0, 3.25, 3.5, 3.75] # Slope is 0.25
        self.assertAlmostEqual(calculate_h2_from_parent_offspring_regression(parents3, offspring3), 0.5, places=5)


        with self.assertRaisesRegex(TypeError, "Parent and offspring phenotypes must be lists."):
            calculate_h2_from_parent_offspring_regression("not a list", offspring1)
        with self.assertRaisesRegex(ValueError, "Phenotype lists cannot be empty."):
            calculate_h2_from_parent_offspring_regression([], [])
        with self.assertRaisesRegex(ValueError, "Parent and offspring phenotype lists must have the same length."):
            calculate_h2_from_parent_offspring_regression(parents1, offspring1[:-1])
        with self.assertRaisesRegex(ValueError, "All phenotypic values must be numeric."):
            calculate_h2_from_parent_offspring_regression(parents1, ["a"] * len(parents1))
        with self.assertRaisesRegex(ValueError, "Regression requires at least two data points"):
            calculate_h2_from_parent_offspring_regression([10], [5])


    def test_partition_variance(self):
        P1 = 100.0
        h2_1 = 0.5
        expected1 = {"genetic_variance": 50.0, "environmental_variance": 50.0}
        self.assertEqual(partition_variance(P1, h2_1), expected1)

        P2 = 20.0
        h2_2 = 0.25
        expected2 = {"genetic_variance": 5.0, "environmental_variance": 15.0}
        self.assertEqual(partition_variance(P2, h2_2), expected2)

        P3 = 50.0
        h2_3 = 0.0 # No genetic variance
        expected3 = {"genetic_variance": 0.0, "environmental_variance": 50.0}
        self.assertEqual(partition_variance(P3, h2_3), expected3)

        P4 = 50.0
        h2_4 = 1.0 # All genetic variance
        expected4 = {"genetic_variance": 50.0, "environmental_variance": 0.0}
        self.assertEqual(partition_variance(P4, h2_4), expected4)

        with self.assertRaisesRegex(ValueError, "Phenotypic variance and heritability must be numeric."):
            partition_variance("100", 0.5)
        with self.assertRaisesRegex(ValueError, "Heritability .* must be between 0 and 1."):
            partition_variance(100, 1.5)
        with self.assertRaisesRegex(ValueError, "Heritability .* must be between 0 and 1."):
            partition_variance(100, -0.1)
        with self.assertRaisesRegex(ValueError, "Phenotypic variance cannot be negative."):
            partition_variance(-10, 0.5)


if __name__ == "__main__":
    unittest.main()
