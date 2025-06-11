import scipy.stats
from collections import Counter

def calculate_hwe_chi_squared(observed_genotypes: dict[str, int]) -> tuple[float, float]:
    """
    Calculates the Chi-squared statistic and p-value for Hardy-Weinberg Equilibrium.

    Assumes a bi-allelic system and diploid individuals.
    Genotype keys are primarily assumed to be like "AA", "Aa", "aa".
    Alleles are inferred from the keys of homozygous genotypes (e.g., 'A' from "AA").
    The function tries to be somewhat flexible but works best with clear "AA", "Aa", "aa" inputs.

    Args:
        observed_genotypes: A dictionary where keys are genotype strings
                            (e.g., "AA", "Aa", "aa") and values are their counts.

    Returns:
        A tuple containing:
            - chi_squared_stat (float): The calculated Chi-squared statistic.
            - p_value (float): The p-value.
        Returns (0.0, 1.0) if any expected genotype count is effectively zero (<= 1e-9 to avoid
        issues with floating point representations of zero, especially if p or q is zero),
        or if N=0 (though N=0 raises ValueError).

    Raises:
        ValueError: If the total number of individuals (N) is 0, or if alleles
                    cannot be reliably determined (e.g., missing homozygous genotypes
                    when heterozygotes are also ambiguous or absent).
    """
    N = sum(observed_genotypes.values())
    if N == 0:
        raise ValueError("Total number of individuals (N) must be greater than 0.")

    # Infer alleles and identify canonical genotype keys
    # Alleles are typically single characters, e.g. 'A', 'a'
    chars = set()
    homozygous_genotypes_present = {} # Store actual keys like "AA" or "aa"
    heterozygous_genotypes_present = {} # Store actual keys like "Aa"

    for gt, count in observed_genotypes.items():
        if not gt or len(gt) != 2 : # Basic validation for "AA" like keys
            # Allow flexibility for other key styles if needed in future by adjusting this logic
            # For now, assume 2-char keys for simplicity as per "AA", "Aa", "aa"
            # print(f"Warning: Genotype key '{gt}' is not in the expected 2-character format. It will be ignored for allele inference.")
            continue

        char1, char2 = gt[0], gt[1]
        chars.add(char1)
        chars.add(char2)
        if char1 == char2:
            homozygous_genotypes_present[char1] = gt # e.g. {'A': "AA", 'a': "aa"}
        else:
            # Store sorted tuple of chars as key to group e.g. "Aa" and "aA"
            het_key_tuple = tuple(sorted((char1, char2)))
            current_het_count = heterozygous_genotypes_present.get(het_key_tuple, {}).get('count', 0)
            heterozygous_genotypes_present[het_key_tuple] = {'key': gt, 'count': current_het_count + count}


    sorted_alleles = sorted(list(chars))

    if not sorted_alleles:
        raise ValueError("Could not determine alleles from genotype keys.")

    allele1 = sorted_alleles[0]
    allele2 = None
    if len(sorted_alleles) == 1: # Fixation
        # Create a placeholder for the second allele if it's truly fixed.
        # This means q or p will be 0.
        # The HWE calculation should reflect this.
        # For example, if allele1 is 'A', allele2 could be conceptual 'a' with 0 frequency.
        # The standard chi-squared test might be problematic (df issues if one allele is absent).
        # However, the rule for E=0 should handle this.
        pass # allele2 remains None, p or q will be 0.
    elif len(sorted_alleles) == 2:
        allele2 = sorted_alleles[1]
    else:
        # This implementation assumes a simple bi-allelic system based on "AA", "Aa", "aa"
        raise ValueError(f"Expected two distinct alleles (e.g., 'A', 'a'), but found {len(sorted_alleles)}: {sorted_alleles}.")

    # Define canonical genotype keys for calculation
    key_hom1 = allele1 + allele1 # e.g. "AA"
    obs_hom1 = observed_genotypes.get(key_hom1, 0)

    if allele2:
        key_hom2 = allele2 + allele2 # e.g. "aa"
        obs_hom2 = observed_genotypes.get(key_hom2, 0)

        # For heterozygote, check both forms e.g. "Aa" or "aA"
        key_het_form1 = allele1 + allele2
        key_het_form2 = allele2 + allele1
        obs_het = observed_genotypes.get(key_het_form1, 0)
        if key_het_form1 != key_het_form2: # e.g. if alleles are 'A' and 'g', then "Ag" vs "gA"
            obs_het += observed_genotypes.get(key_het_form2, 0)
            if observed_genotypes.get(key_het_form1,0) > 0 and observed_genotypes.get(key_het_form2,0) > 0:
                 # This is a bit strict, usually one form is chosen by convention in input data.
                 # print(f"Warning: Counts for both heterozygous forms {key_het_form1} and {key_het_form2} found and summed.")
                 pass # Summed them up.
    else: # Fixation case (allele2 is None)
        key_hom2 = "" # No second homozygote
        obs_hom2 = 0
        key_het_form1 = "" # No heterozygote with a conceptual second allele
        obs_het = 0

    # Allele counts and frequencies
    count_allele1 = 2 * obs_hom1 + obs_het
    count_allele2 = 2 * obs_hom2 + obs_het
    total_alleles = count_allele1 + count_allele2

    if total_alleles == 0: # Should be caught by N > 0, but if keys don't match
        return 0.0, 1.0

    p = count_allele1 / total_alleles # Freq of allele1
    q = count_allele2 / total_alleles # Freq of allele2

    # Expected genotype counts under HWE
    exp_hom1 = (p**2) * N
    exp_het  = (2 * p * q) * N
    exp_hom2 = (q**2) * N

    # If any expected count is effectively zero (e.g., due to p=0 or q=0 for fixation)
    # return (0.0, 1.0) as per problem statement.
    # Use a small epsilon for floating point comparisons to zero.
    epsilon = 1e-9
    if exp_hom1 < epsilon or exp_het < epsilon or exp_hom2 < epsilon :
        # More precise check for fixation: if p or q is 0 (or very close to 0)
        if (p < epsilon and q > 1-epsilon) or (q < epsilon and p > 1-epsilon): # Fixation
            # Check if observed matches expected perfectly for fixation
            # obs_hom1 vs N*p^2, obs_het vs N*2pq, obs_hom2 vs N*q^2
            # If p=1, q=0: Exp_hom1=N, Exp_het=0, Exp_hom2=0.
            # If obs_hom1=N, obs_het=0, obs_hom2=0, then chi_sq is indeed 0.
            if (p > 1-epsilon and abs(obs_hom1 - N) < epsilon and obs_het < epsilon and obs_hom2 < epsilon) or \
               (q > 1-epsilon and abs(obs_hom2 - N) < epsilon and obs_het < epsilon and obs_hom1 < epsilon):
                return 0.0, 1.0
        # If not perfect fixation but an E is still zero (e.g. Aa=0 if p=0.5, N=0), the rule applies generally.
        # The problem states "if any expected count is <=0".
        # This handles cases where p=0 or q=0 leading to E=0 for some genotypes.
        # This also handles if N is so small that p^2*N rounds to 0.
        # However, my epsilon logic above is specific to p/q being near 0/1.
        # The general rule is: if any E_i <= 0 (or very small for float), return 0, 1.0
        # This is because chi-sq formula would divide by E_i.
        # If p or q is exactly 0, then some E_i will be exactly 0.
        if exp_hom1 <= epsilon or (allele2 and exp_hom2 <= epsilon) or (allele2 and exp_het <= epsilon):
            # The conditions for allele2 ensure we only check exp_hom2 and exp_het if allele2 exists.
            # If allele2 is None (fixation of allele1), then q=0, so exp_het=0 and exp_hom2=0.
            # These fall under the check.
            return 0.0, 1.0


    # Chi-squared statistic
    observed_counts = [obs_hom1, obs_het, obs_hom2]
    expected_counts = [exp_hom1, exp_het, exp_hom2]

    chi_squared_stat = 0
    for obs, exp in zip(observed_counts, expected_counts):
        # Skip terms where expected is 0 if we didn't catch it above (e.g. if allele2 was None)
        if exp < epsilon: # Should have been caught by the rule above
            if obs < epsilon : # obs=0, exp=0, component is 0
                continue
            else: # obs > 0, exp = 0, chi-sq is infinite. This indicates strong disequilibrium.
                  # This case should ideally be handled by the E<=0 rule => (0.0, 1.0)
                  # which suggests deviation cannot be reliably calculated by this formula.
                  # However, typical chi-sq tests require E_i >= 5.
                  # The rule "if any expected count is <= 0, return (0.0, 1.0)" is a simplification.
                  # Let's adhere strictly to it. If we are here, all E_i > epsilon.
                  pass

        chi_squared_stat += ((obs - exp)**2) / exp

    df = 1 # Degrees of freedom: number of genotypes - 1 - number of independent alleles estimated
           # 3 genotypes - 1 - 1 (since p+q=1, estimating p means q is fixed) = 1 df.
           # This is true if we estimate allele frequencies from the sample.

    p_value = scipy.stats.chi2.sf(chi_squared_stat, df)

    return chi_squared_stat, p_value
