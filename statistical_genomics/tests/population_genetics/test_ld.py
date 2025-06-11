import unittest
import math
from statistical_genomics.statistical_genomics.population_genetics.ld import (
    calculate_allele_frequencies_from_haplotypes,
    calculate_d,
    calculate_d_prime,
    calculate_r_squared
)

class TestLD(unittest.TestCase):

    def test_allele_freqs_from_haplotypes_valid(self):
        hap_freqs = {'AB': 0.5, 'Ab': 0.1, 'aB': 0.1, 'ab': 0.3} # Sum = 1.0
        # pA = 0.5 + 0.1 = 0.6
        # pa = 0.1 + 0.3 = 0.4
        # pB = 0.5 + 0.1 = 0.6
        # pb = 0.1 + 0.3 = 0.4
        expected_allele_freqs = {'A': 0.6, 'a': 0.4, 'B': 0.6, 'b': 0.4}
        # Note: the function internally sorts alleles, so 'A'<'a', 'B'<'b'
        # If input was {'gt':0.5, 'gT':0.1, 'Gt':0.1, 'GT':0.3} -> G,g,T,t
        # keys in result would be G,T,g,t if sorted that way by my code.
        # My code sorts unique chars from first pos, then second pos.
        # Locus1 alleles: A,a. Locus2 alleles: B,b.
        # Sorted: A1='A', a1='a'. B2='B', b2='b'.
        # Keys in result: 'A', 'a', 'B', 'b'.
        allele_freqs = calculate_allele_frequencies_from_haplotypes(hap_freqs)
        for allele, freq in expected_allele_freqs.items():
            self.assertAlmostEqual(allele_freqs[allele], freq)

        hap_freqs_case = {'ab': 0.5, 'aB': 0.1, 'Ab': 0.1, 'AB': 0.3} # Sum = 1.0
        # pA = 0.1 + 0.3 = 0.4
        # pa = 0.5 + 0.1 = 0.6
        # pB = 0.1 + 0.3 = 0.4
        # pb = 0.5 + 0.1 = 0.6
        expected_allele_freqs_case = {'A': 0.4, 'a': 0.6, 'B': 0.4, 'b': 0.6}
        allele_freqs_case = calculate_allele_frequencies_from_haplotypes(hap_freqs_case)
        for allele, freq in expected_allele_freqs_case.items():
            self.assertAlmostEqual(allele_freqs_case[allele], freq, places=5)


    def test_allele_freqs_from_haplotypes_invalid_sum(self):
        hap_freqs = {'AB': 0.5, 'Ab': 0.1, 'aB': 0.1, 'ab': 0.2} # Sum = 0.9
        # Adjusted regex to match the more specific error message from the code
        with self.assertRaisesRegex(ValueError, "Sum of input haplotype frequencies .* must be close to 1.0"):
            calculate_allele_frequencies_from_haplotypes(hap_freqs)

    def test_allele_freqs_from_haplotypes_malformed_keys(self):
        hap_freqs_short_key = {'A': 0.5, 'Ab': 0.1, 'aB': 0.1, 'ab': 0.3} # Sum is 1.0 but key 'A' is bad
        with self.assertRaisesRegex(ValueError, "Malformed haplotype key: 'A'"):
            calculate_allele_frequencies_from_haplotypes(hap_freqs_short_key)

        hap_freqs_long_key = {'ABC': 0.5, 'Ab': 0.1, 'aB': 0.1, 'ab': 0.3} # Sum is 1.0 but key 'ABC' is bad
        with self.assertRaisesRegex(ValueError, "Malformed haplotype key: 'ABC'"):
            calculate_allele_frequencies_from_haplotypes(hap_freqs_long_key)

        # The following sub-test was problematic because {'AB':0.4,'Ab':0.1,'BB':0.3,'Bb':0.2}
        # correctly infers alleles L1:{'A','B'}, L2:{'B','b'} by the fixed code.
        # It does not raise "Expected 2 alleles at locus 1" anymore.
        # hap_freqs_not_2_alleles_locus1 = {'AB': 0.4, 'Ab': 0.1, 'BB': 0.3, 'Bb':0.2}
        # with self.assertRaisesRegex(ValueError, "Expected 2 alleles at locus 1"):
        #      calculate_allele_frequencies_from_haplotypes(hap_freqs_not_2_alleles_locus1)


    def test_d_calculation(self):
        # P_AB=0.5, P_Ab=0.1, P_aB=0.1, P_ab=0.3
        # D = P_AB * P_ab - P_Ab * P_aB = 0.5*0.3 - 0.1*0.1 = 0.15 - 0.01 = 0.14
        hap_freqs = {'AB': 0.5, 'Ab': 0.1, 'aB': 0.1, 'ab': 0.3}
        d_value, allele_freqs = calculate_d(hap_freqs)
        self.assertAlmostEqual(d_value, 0.14)
        # pA = 0.6, pa = 0.4, pB = 0.6, pb = 0.4
        self.assertAlmostEqual(allele_freqs['A'], 0.6)
        self.assertAlmostEqual(allele_freqs['a'], 0.4)
        self.assertAlmostEqual(allele_freqs['B'], 0.6)
        self.assertAlmostEqual(allele_freqs['b'], 0.4)

    def test_d_prime_calculation_positive_d(self):
        # D=0.14. pA=0.6, pa=0.4, pB=0.6, pb=0.4
        # D > 0: D_max = min(pA*pb, pa*pB) = min(0.6*0.4, 0.4*0.6) = min(0.24, 0.24) = 0.24
        # D' = 0.14 / 0.24 approx 0.583333...
        allele_freqs = {'A': 0.6, 'a': 0.4, 'B': 0.6, 'b': 0.4}
        d_prime = calculate_d_prime(0.14, allele_freqs)
        self.assertAlmostEqual(d_prime, 0.14 / 0.24, places=5)

    def test_d_prime_calculation_negative_d(self):
        # P_AB=0.1, P_Ab=0.4, P_aB=0.4, P_ab=0.1. Sum=1.0
        # pA=0.5, pa=0.5, pB=0.5, pb=0.5.
        # D = 0.1*0.1 - 0.4*0.4 = 0.01 - 0.16 = -0.15.
        d_value, allele_freqs = calculate_d({'AB':0.1, 'Ab':0.4, 'aB':0.4, 'ab':0.1})
        self.assertAlmostEqual(d_value, -0.15)
        self.assertAlmostEqual(allele_freqs['A'], 0.5)
        # D < 0: D_max_abs_val = min(pA*pB, pa*pb) = min(0.5*0.5, 0.5*0.5) = min(0.25, 0.25) = 0.25
        # D' = D / D_max_abs_val = -0.15 / 0.25 = -0.6
        d_prime = calculate_d_prime(d_value, allele_freqs)
        self.assertAlmostEqual(d_prime, -0.15 / 0.25, places=5) # Should be -0.6

    def test_d_prime_perfect_ld(self):
        # P_AB=0.5, P_Ab=0, P_aB=0, P_ab=0.5. (Complete coupling AB/ab)
        # pA=0.5, pa=0.5, pB=0.5, pb=0.5.
        # D = 0.5*0.5 - 0*0 = 0.25.
        d_value, allele_freqs = calculate_d({'AB':0.5, 'Ab':0.0, 'aB':0.0, 'ab':0.5})
        self.assertAlmostEqual(d_value, 0.25)
        # D > 0: D_max = min(pA*pb, pa*pB) = min(0.5*0.5, 0.5*0.5) = 0.25.
        # D' = 0.25 / 0.25 = 1.0
        d_prime = calculate_d_prime(d_value, allele_freqs)
        self.assertAlmostEqual(d_prime, 1.0, places=5)

        # P_AB=0, P_Ab=0.5, P_aB=0.5, P_ab=0. (Complete repulsion Ab/aB)
        # pA=0.5, pa=0.5, pB=0.5, pb=0.5
        # D = 0*0 - 0.5*0.5 = -0.25
        d_value_neg, allele_freqs_neg = calculate_d({'AB':0.0, 'Ab':0.5, 'aB':0.5, 'ab':0.0})
        self.assertAlmostEqual(d_value_neg, -0.25)
        # D < 0: D_max_abs = min(pA*pB, pa*pb) = min(0.5*0.5, 0.5*0.5) = 0.25
        # D' = -0.25 / 0.25 = -1.0
        d_prime_neg = calculate_d_prime(d_value_neg, allele_freqs_neg)
        self.assertAlmostEqual(d_prime_neg, -1.0, places=5)


    def test_d_prime_zero_d(self):
        # Linkage equilibrium: P_AB = pA*pB. D=0.
        # pA=0.6, pa=0.4, pB=0.7, pb=0.3
        # P_AB = 0.6*0.7 = 0.42
        # P_Ab = 0.6*0.3 = 0.18
        # P_aB = 0.4*0.7 = 0.28
        # P_ab = 0.4*0.3 = 0.12
        # Sum = 0.42+0.18+0.28+0.12 = 1.0
        hap_freqs = {'AB': 0.42, 'Ab': 0.18, 'aB': 0.28, 'ab': 0.12}
        d_value, allele_freqs = calculate_d(hap_freqs)
        self.assertAlmostEqual(d_value, 0.0, places=5)
        d_prime = calculate_d_prime(d_value, allele_freqs)
        self.assertAlmostEqual(d_prime, 0.0, places=5)

    def test_d_prime_fixation(self):
        # Fixation at locus A (pA=1, pa=0)
        # P_AB=0.7 (so B=1), P_Ab=0 (so b=0), P_aB=0, P_ab=0 -> This is pA=0.7, pB=0.7 - not fixation.
        # If pA=1, then P_AB=pB, P_Ab=pb, P_aB=0, P_ab=0.
        # Let pB=0.7, pb=0.3. So P_AB=0.7, P_Ab=0.3, P_aB=0, P_ab=0. Sum=1.0
        hap_freqs_fix_A = {'AB': 0.7, 'Ab': 0.3, 'aB': 0.0, 'ab': 0.0}
        d_value, allele_freqs = calculate_d(hap_freqs_fix_A)
        # pA=1, pa=0, pB=0.7, pb=0.3
        # D = P_AB*P_ab - P_Ab*P_aB = 0.7*0 - 0.3*0 = 0
        self.assertAlmostEqual(d_value, 0.0, places=5)
        d_prime = calculate_d_prime(d_value, allele_freqs)
        # D=0, so D' is 0. Denominator for D' would be min(1*0.3, 0*0.7)=0 if D>0 or min(1*0.7,0*0.3)=0 if D<0.
        # My function returns 0 if D=0.
        self.assertAlmostEqual(d_prime, 0.0, places=5) # D=0 means D'=0

        # What if D is non-zero but denom is zero? -> nan
        # This happens if pA=0.5, pa=0.5, but pB=1, pb=0.
        # Haps: AB=0.5, Ab=0, aB=0.5, ab=0. Sum=1.0
        # pA=0.5, pa=0.5, pB=1, pb=0.
        # D = 0.5*0 - 0*0.5 = 0.
        # So D is 0.
        hap_freqs_fix_B = {'AB': 0.5, 'Ab': 0.0, 'aB': 0.5, 'ab': 0.0}
        d_val_fix_B, af_fix_B = calculate_d(hap_freqs_fix_B)
        self.assertAlmostEqual(d_val_fix_B, 0.0)
        dp_fix_B = calculate_d_prime(d_val_fix_B, af_fix_B)
        self.assertAlmostEqual(dp_fix_B, 0.0)


    def test_r_squared_calculation(self):
        # D=0.14, pA=0.6, pa=0.4, pB=0.6, pb=0.4
        # Denom = pA * pa * pB * pb = 0.6 * 0.4 * 0.6 * 0.4 = 0.24 * 0.24 = 0.0576
        # r^2 = D^2 / Denom = 0.14^2 / 0.0576 = 0.0196 / 0.0576 approx 0.340277...
        allele_freqs = {'A': 0.6, 'a': 0.4, 'B': 0.6, 'b': 0.4}
        r_sq = calculate_r_squared(0.14, allele_freqs)
        self.assertAlmostEqual(r_sq, (0.14**2) / (0.6*0.4*0.6*0.4), places=5)

    def test_r_squared_perfect_ld(self):
        # P_AB=0.5, P_Ab=0, P_aB=0, P_ab=0.5.
        # D=0.25. pA=0.5, pa=0.5, pB=0.5, pb=0.5.
        # Denom = 0.5^4 = 0.0625.
        # r^2 = 0.25^2 / 0.0625 = 0.0625 / 0.0625 = 1.0.
        d_value, allele_freqs = calculate_d({'AB':0.5, 'Ab':0.0, 'aB':0.0, 'ab':0.5})
        self.assertAlmostEqual(d_value, 0.25)
        r_sq = calculate_r_squared(d_value, allele_freqs)
        self.assertAlmostEqual(r_sq, 1.0, places=5)

    def test_r_squared_zero_d(self):
        hap_freqs = {'AB': 0.42, 'Ab': 0.18, 'aB': 0.28, 'ab': 0.12} # D=0
        d_value, allele_freqs = calculate_d(hap_freqs)
        self.assertAlmostEqual(d_value, 0.0, places=5)
        r_sq = calculate_r_squared(d_value, allele_freqs)
        self.assertAlmostEqual(r_sq, 0.0, places=5)

    def test_r_squared_fixation(self):
        # Fixation at locus A (pA=1, pa=0). D=0.
        hap_freqs_fix_A = {'AB': 0.7, 'Ab': 0.3, 'aB': 0.0, 'ab': 0.0}
        d_value, allele_freqs = calculate_d(hap_freqs_fix_A) # D=0
        self.assertAlmostEqual(d_value, 0.0)
        # Denom for r^2 = pA*pa*pB*pb = 1*0*0.7*0.3 = 0.
        r_sq = calculate_r_squared(d_value, allele_freqs)
        # My func returns 0 if D=0, nan if D!=0 and Denom=0.
        self.assertAlmostEqual(r_sq, 0.0, places=5) # D=0 leads to r_sq=0

        # Case where D is non-zero (hypothetically, due to float issues) but denom is zero
        # This tests the nan return for r_squared
        allele_freqs_fix = {'A': 1.0, 'a': 0.0, 'B': 0.7, 'b': 0.3}
        r_sq_nan = calculate_r_squared(0.001, allele_freqs_fix) # Non-zero D, zero denom
        self.assertTrue(math.isnan(r_sq_nan))


if __name__ == '__main__':
    unittest.main()
