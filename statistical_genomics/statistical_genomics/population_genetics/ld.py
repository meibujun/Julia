import math
from collections import Counter # Not strictly needed for current impl, but good for future.

def calculate_allele_frequencies_from_haplotypes(hap_freqs: dict[str, float]) -> dict[str, float]:
    """
    Calculates allele frequencies from observed haplotype frequencies for a
    2-locus, bi-allelic system.

    Args:
        hap_freqs: A dictionary of haplotype frequencies.
                   Keys are expected to be 2-character strings (e.g., "AB", "Ab", "aB", "ab").
                   Values are their corresponding frequencies.
                   The sum of frequencies should be close to 1.0.

    Returns:
        A dictionary of allele frequencies with canonical keys 'A', 'a' (for locus 1 alleles)
        and 'B', 'b' (for locus 2 alleles). 'A' and 'B' are lexicographically smaller versions
        of the inferred alleles for each locus.
        e.g., {'A': p_A1, 'a': p_a1, 'B': p_B2, 'b': p_b2}.

    Raises:
        ValueError: If haplotype keys are malformed, if not exactly two alleles per locus
                    can be inferred, or if the sum of haplotype frequencies is not close to 1.0.
    """
    if not hap_freqs or len(hap_freqs) == 0:
        raise ValueError("Haplotype frequencies dictionary cannot be empty.")

    if not math.isclose(sum(hap_freqs.values()), 1.0):
        raise ValueError(f"Sum of input haplotype frequencies ({sum(hap_freqs.values())}) must be close to 1.0.")

    alleles_locus1 = set()
    alleles_locus2 = set()

    for hap_key in hap_freqs: # No .items() needed yet, just keys for allele inference
        if not isinstance(hap_key, str) or len(hap_key) != 2:
            raise ValueError(
                f"Malformed haplotype key: '{hap_key}'. Keys must be 2-character strings (e.g., 'AB')."
            )
        alleles_locus1.add(hap_key[0]) # Use original case for inferring allele characters
        alleles_locus2.add(hap_key[1])

    if len(alleles_locus1) != 2 or len(alleles_locus2) != 2:
        raise ValueError(
            f"Expected 2 alleles at locus 1 (got {alleles_locus1}) and locus 2 (got {alleles_locus2}). "
            "System is bi-allelic at two loci."
        )

    sorted_alleles1 = sorted(list(alleles_locus1))
    sorted_alleles2 = sorted(list(alleles_locus2))

    A1, a1 = sorted_alleles1[0], sorted_alleles1[1] # A1 is lexicographically smaller
    B2, b2 = sorted_alleles2[0], sorted_alleles2[1] # B2 is lexicographically smaller

    # Define the 4 canonical haplotype strings using the inferred, sorted alleles
    # These are the specific forms we need frequencies for.
    # Their uppercase versions will be used for matching against uppercased input keys.
    canon_hap_str_A1B2 = (A1 + B2).upper()
    canon_hap_str_A1b2 = (A1 + b2).upper()
    canon_hap_str_a1B2 = (a1 + B2).upper()
    canon_hap_str_a1b2 = (a1 + b2).upper()

    # Ensure these 4 canonical uppercase strings are distinct. If not, input alleles were not truly bi-allelic after uppercasing (e.g. 'a' and 'A').
    # This should be caught by len(alleles_locusX)==2 if alleles only differ by case e.g. {'a','A'} -> sorted ['A','a']
    # If A1='A',a1='a',B1='B',b1='b', then canon_hap_str_ are "AB", "AB", "AB", "AB". This is what was causing issues.
    # The canonical strings themselves must be distinct based on A1,a1,B2,b2.
    # A1B2, A1b2, a1B2, a1b2 are distinct if A1!=a1 and B2!=b2.
    # Example: A1='A', a1='a', B2='B', b2='b'.
    # canon_A1B2_lookup = "AB".upper() = "AB"
    # canon_A1b2_lookup = "Ab".upper() = "AB"  <-- Problem was here. The lookup key itself became non-unique.

    # Solution: The canonical haplotype strings (using A1,a1,B2,b2) are the reference.
    # We iterate through input hap_freqs, uppercase the input key, and see which canonical form it matches.

    P_A1B2_val, P_A1b2_val, P_a1B2_val, P_a1b2_val = 0.0, 0.0, 0.0, 0.0

    for input_hap_key, freq_val in hap_freqs.items():
        # Determine which canonical haplotype this input_hap_key corresponds to
        # by comparing its characters to the inferred A1,a1,B2,b2
        char1, char2 = input_hap_key[0], input_hap_key[1]

        # Normalize chars from input key to match A1/a1 and B2/b2 category
        # E.g. if A1='A', a1='a', then 'A' maps to A1, 'a' maps to a1.
        #      if A1='g', a1='G', then 'g' maps to A1, 'G' maps to a1.

        norm_char1 = A1 if char1 == A1 or char1 == a1 and A1==char1 else a1 # Assign to A1 or a1
        norm_char2 = B2 if char2 == B2 or char2 == b2 and B2==char2 else b2 # Assign to B2 or b2

        # Reconstruct canonical haplotype from normalized chars
        # This ensures we are comparing with A1B2, A1b2, a1B2, a1b2 consistently
        if norm_char1 == A1 and norm_char2 == B2:
            P_A1B2_val += freq_val
        elif norm_char1 == A1 and norm_char2 == b2:
            P_A1b2_val += freq_val
        elif norm_char1 == a1 and norm_char2 == B2:
            P_a1B2_val += freq_val
        elif norm_char1 == a1 and norm_char2 == b2:
            P_a1b2_val += freq_val
        # else: # Should not happen if input keys are valid combinations of inferred alleles
              # Or could be a key like 'AX' where X was not one of the 2 inferred alleles for locus 2.
              # The initial sum check sum(hap_freqs.values()) includes these.
              # The current_sum_of_canonical_haps check below will detect if such freqs are "lost".

    current_sum_of_canonical_haps = P_A1B2_val + P_A1b2_val + P_a1B2_val + P_a1b2_val
    if not math.isclose(current_sum_of_canonical_haps, 1.0):
         # This error means that the sum of frequencies for the four specific haplotype
         # forms (A1B2, A1b2, a1B2, a1b2), after attempting to map all input keys
         # to these forms, does not equal the total sum of input frequencies (which should be 1.0).
         # This implies some input keys did not map to any of these 4 forms.
         raise ValueError(f"Sum of frequencies for the four canonical haplotype forms ({current_sum_of_canonical_haps}) "
                          f"derived from alleles ({A1},{a1} and {B2},{b2}) must be close to 1.0. "
                          f"Input keys might not all map to these forms or are inconsistent.")

    p_A1 = P_A1B2_val + P_A1b2_val
    p_a1 = P_a1B2_val + P_a1b2_val
    p_B2 = P_A1B2_val + P_a1B2_val
    p_b2 = P_A1b2_val + P_a1b2_val

    if not (math.isclose(p_A1 + p_a1, 1.0) and math.isclose(p_B2 + p_b2, 1.0)):
        # This is an internal sanity check, should not be reached if prior logic is correct.
        raise ValueError(f"Internal error: Allele frequencies pA1={p_A1}, pa1={p_a1} (sum {p_A1+p_a1}) or "
                         f"pB2={p_B2}, pb2={p_b2} (sum {p_B2+p_b2}) do not sum to 1.0 per locus.")

    final_allele_freqs = {
        'A': p_A1, 'a': p_a1,
        'B': p_B2, 'b': p_b2
    }
    return final_allele_freqs


def calculate_d(hap_freqs: dict[str, float]) -> tuple[float, dict[str, float]]:
    allele_freqs_canonical = calculate_allele_frequencies_from_haplotypes(hap_freqs)

    # To calculate D = P(A1B2)P(a1b2) - P(A1b2)P(a1B2), we need the frequencies
    # of the four specific haplotype configurations based on inferred A1,a1,B2,b2.
    # The same logic as in calculate_allele_frequencies_from_haplotypes to get these P_ values.

    alleles_locus1 = set()
    alleles_locus2 = set()
    for hap_key in hap_freqs:
        if not isinstance(hap_key, str) or len(hap_key) != 2:
             raise ValueError(f"Malformed haplotype key for D calc: '{hap_key}'.") # Should be caught by helper
        alleles_locus1.add(hap_key[0])
        alleles_locus2.add(hap_key[1])

    if len(alleles_locus1) != 2 or len(alleles_locus2) != 2: # Redundant if helper passed
        raise ValueError("Could not confirm 2 alleles at each locus for D calculation.")

    s_al1 = sorted(list(alleles_locus1))
    s_al2 = sorted(list(alleles_locus2))
    A1, a1 = s_al1[0], s_al1[1]
    B2, b2 = s_al2[0], s_al2[1]

    # Get haplotype frequencies (P_A1B2, P_A1b2, P_a1B2, P_a1b2) correctly
    P_A1B2_val, P_A1b2_val, P_a1B2_val, P_a1b2_val = 0.0, 0.0, 0.0, 0.0
    for input_hap_key, freq_val in hap_freqs.items():
        char1, char2 = input_hap_key[0], input_hap_key[1]
        norm_char1 = A1 if char1 == A1 or (char1 == a1 and A1 == char1) else a1
        if char1 != A1 and char1 != a1 : continue # Skip if char1 is not one of the inferred locus1 alleles

        norm_char2 = B2 if char2 == B2 or (char2 == b2 and B2 == char2) else b2
        if char2 != B2 and char2 != b2 : continue # Skip if char2 is not one of the inferred locus2 alleles

        if norm_char1 == A1 and norm_char2 == B2: P_A1B2_val += freq_val
        elif norm_char1 == A1 and norm_char2 == b2: P_A1b2_val += freq_val
        elif norm_char1 == a1 and norm_char2 == B2: P_a1B2_val += freq_val
        elif norm_char1 == a1 and norm_char2 == b2: P_a1b2_val += freq_val

    d_value = (P_A1B2_val * P_a1b2_val) - (P_A1b2_val * P_a1B2_val)
    return d_value, allele_freqs_canonical


def calculate_d_prime(d_value: float, allele_freqs: dict[str, float]) -> float:
    if math.isclose(d_value, 0.0):
        return 0.0

    p_A1 = allele_freqs.get('A', -1.0)
    p_a1 = allele_freqs.get('a', -1.0)
    p_B2 = allele_freqs.get('B', -1.0)
    p_b2 = allele_freqs.get('b', -1.0)

    if p_A1 < 0 or p_a1 < 0 or p_B2 < 0 or p_b2 < 0:
        keys_found = sorted(list(allele_freqs.keys()))
        raise ValueError(f"Allele frequencies dictionary is missing one or more canonical keys 'A','a','B','b' or has invalid values. Found: {keys_found}")

    d_max_denominator = 0.0
    if d_value > 0:
        d_max_denominator = min(p_A1 * p_b2, p_a1 * p_B2)
    elif d_value < 0:
        d_max_denominator = min(p_A1 * p_B2, p_a1 * p_b2)

    if math.isclose(d_max_denominator, 0.0):
        return float('nan') if not math.isclose(d_value, 0.0) else 0.0

    d_prime = d_value / d_max_denominator
    return d_prime


def calculate_r_squared(d_value: float, allele_freqs: dict[str, float]) -> float:
    if math.isclose(d_value, 0.0):
        return 0.0

    p_A1 = allele_freqs.get('A', -1.0)
    p_a1 = allele_freqs.get('a', -1.0)
    p_B2 = allele_freqs.get('B', -1.0)
    p_b2 = allele_freqs.get('b', -1.0)

    if p_A1 < 0 or p_a1 < 0 or p_B2 < 0 or p_b2 < 0:
         keys_found = sorted(list(allele_freqs.keys()))
         raise ValueError(f"Allele frequencies dictionary is missing one or more canonical keys 'A','a','B','b' or has invalid values. Found: {keys_found}")

    denominator = p_A1 * p_a1 * p_B2 * p_b2

    if math.isclose(denominator, 0.0):
        return float('nan') if not math.isclose(d_value, 0.0) else 0.0

    r_squared = (d_value**2) / denominator
    return r_squared
