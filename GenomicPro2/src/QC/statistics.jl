"""
Statistical tests and metrics for quality control.
"""

"""
    hardy_weinberg_test(n0::Int, n1::Int, n2::Int) -> Float64

Compute Hardy-Weinberg equilibrium exact test p-value.

Uses the exact test based on SNPHWE algorithm.

# Arguments
- `n0::Int`: Count of AA genotypes (homozygous reference)
- `n1::Int`: Count of Aa genotypes (heterozygous)
- `n2::Int`: Count of aa genotypes (homozygous alternate)

# Returns
P-value for HWE test (higher values indicate better fit to HWE)

# Theory

Under Hardy-Weinberg equilibrium:
- Expected het frequency = 2p(1-p)
- where p = (2*n0 + n1) / (2*n)

The test uses exact enumeration of all possible genotype configurations
with the same allele frequency.

# Example
```julia
# 100 AA, 50 Aa, 25 aa
pval = hardy_weinberg_test(100, 50, 25)
if pval < 1e-6
    println("Significant deviation from HWE")
end
```

# References
Wigginton JE, Cutler DJ, Abecasis GR. 2005. A note on exact tests of
Hardy-Weinberg equilibrium. Am J Hum Genet. 76(5):887-93.
"""
function hardy_weinberg_test(n0::Int, n1::Int, n2::Int)
    # Total number of individuals
    n = n0 + n1 + n2

    if n == 0
        return 1.0  # No data
    end

    # Total number of alleles
    n_alleles = 2 * n

    # Rare allele count (choose smaller)
    n_rare = 2 * n0 + n1  # A alleles
    n_common = 2 * n2 + n1  # a alleles

    # Make rare truly the rarer allele
    if n_rare > n_common
        n_rare, n_common = n_common, n_rare
        n0, n2 = n2, n0
    end

    # Probability of observed heterozygotes
    het_probs = zeros(Float64, n_rare + 1)

    # Compute probabilities for all possible heterozygote counts
    for het in 0:n_rare
        # Number of homozygotes for rare allele
        hom_rare = (n_rare - het) ÷ 2

        # Check if valid configuration
        if (n_rare - het) % 2 != 0
            continue  # Can't have fractional homozygotes
        end

        # Number of homozygotes for common allele
        hom_common = n - het - hom_rare

        if hom_common < 0
            continue
        end

        # Log probability (to avoid overflow)
        # P ∝ n! / (hom_rare! * het! * hom_common!)
        log_prob = (
            log_factorial(n) -
            log_factorial(hom_rare) -
            log_factorial(het) -
            log_factorial(hom_common)
        )

        het_probs[het + 1] = exp(log_prob)
    end

    # Normalize
    total_prob = sum(het_probs)
    if total_prob > 0
        het_probs ./= total_prob
    else
        return 1.0
    end

    # P-value: sum of probabilities <= observed probability
    obs_prob = het_probs[n1 + 1]
    pvalue = sum(p for p in het_probs if p <= obs_prob + 1e-10)

    return min(pvalue, 1.0)
end

"""
    log_factorial(n::Int) -> Float64

Compute log(n!) efficiently.

Uses Stirling's approximation for large n.
"""
function log_factorial(n::Int)
    if n < 0
        throw(ArgumentError("Factorial of negative number"))
    end

    if n <= 1
        return 0.0
    end

    if n < 20
        # Exact for small n
        return log(factorial(big(n)))
    else
        # Stirling's approximation for large n
        # log(n!) ≈ n*log(n) - n + 0.5*log(2π*n)
        return n * log(n) - n + 0.5 * log(2π * n)
    end
end

"""
    call_rate(geno::CompactGenotypes; dim::Int=0) -> Union{Float64, Vector{Float64}}

Compute call rate (1 - missing rate).

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `dim::Int`: Dimension (0=overall, 1=per sample, 2=per marker)

# Returns
- `dim=0`: Overall call rate
- `dim=1`: Call rate per sample
- `dim=2`: Call rate per marker

# Example
```julia
overall_cr = call_rate(geno)
sample_cr = call_rate(geno; dim=1)
marker_cr = call_rate(geno; dim=2)

# Filter samples with < 95% call rate
keep = findall(sample_cr .>= 0.95)
```
"""
function call_rate(geno::CompactGenotypes; dim::Int=0)
    return 1.0 .- missing_rate(geno; dim=dim)
end

"""
    heterozygosity_rate(geno::CompactGenotypes; dim::Int=0) -> Union{Float64, Vector{Float64}}

Compute heterozygosity rate.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `dim::Int`: Dimension (0=overall, 1=per sample, 2=per marker)

# Returns
- `dim=0`: Overall heterozygosity rate
- `dim=1`: Heterozygosity rate per sample
- `dim=2`: Heterozygosity rate per marker

# Example
```julia
overall_het = heterozygosity_rate(geno)
sample_het = heterozygosity_rate(geno; dim=1)

# Identify samples with unusual heterozygosity
mean_het = mean(sample_het)
std_het = std(sample_het)
outliers = findall(abs.(sample_het .- mean_het) .> 3 * std_het)
```
"""
function heterozygosity_rate(geno::CompactGenotypes; dim::Int=0)
    if dim == 0
        # Overall heterozygosity
        n_het = 0
        n_total = 0

        for j in 1:n_markers(geno)
            for i in 1:n_samples(geno)
                if !ismissing(geno, i, j)
                    if geno[i, j] == 1
                        n_het += 1
                    end
                    n_total += 1
                end
            end
        end

        return n_total > 0 ? n_het / n_total : 0.0

    elseif dim == 1
        # Per sample
        het_rates = zeros(Float64, n_samples(geno))

        for i in 1:n_samples(geno)
            n_het = 0
            n_total = 0

            for j in 1:n_markers(geno)
                if !ismissing(geno, i, j)
                    if geno[i, j] == 1
                        n_het += 1
                    end
                    n_total += 1
                end
            end

            het_rates[i] = n_total > 0 ? n_het / n_total : 0.0
        end

        return het_rates

    elseif dim == 2
        # Per marker
        het_rates = zeros(Float64, n_markers(geno))

        for j in 1:n_markers(geno)
            n_het = 0
            n_total = 0

            for i in 1:n_samples(geno)
                if !ismissing(geno, i, j)
                    if geno[i, j] == 1
                        n_het += 1
                    end
                    n_total += 1
                end
            end

            het_rates[j] = n_total > 0 ? n_het / n_total : 0.0
        end

        return het_rates

    else
        throw(ArgumentError("dim must be 0, 1, or 2"))
    end
end

"""
    expected_heterozygosity(geno::CompactGenotypes) -> Vector{Float64}

Compute expected heterozygosity under HWE for each marker.

Expected Het = 2p(1-p) where p is the allele frequency.

# Arguments
- `geno::CompactGenotypes`: Genotype data

# Returns
Vector of expected heterozygosity values per marker

# Example
```julia
obs_het = heterozygosity_rate(geno; dim=2)
exp_het = expected_heterozygosity(geno)

# Markers with excess heterozygosity
excess = findall((obs_het .- exp_het) .> 0.1)
```
"""
function expected_heterozygosity(geno::CompactGenotypes)
    freqs = allele_frequencies(geno)
    return 2 .* freqs .* (1 .- freqs)
end

"""
    inbreeding_coefficient(geno::CompactGenotypes; dim::Int=1) -> Vector{Float64}

Compute inbreeding coefficient (F) for samples.

F = 1 - (Observed Het / Expected Het)

Positive F indicates excess homozygosity (inbreeding).
Negative F indicates excess heterozygosity.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `dim::Int`: Must be 1 (per sample)

# Returns
Vector of F values per sample

# Example
```julia
F = inbreeding_coefficient(geno)

# Identify highly inbred samples
inbred = findall(F .> 0.1)
```
"""
function inbreeding_coefficient(geno::CompactGenotypes; dim::Int=1)
    if dim != 1
        throw(ArgumentError("inbreeding_coefficient only supports dim=1 (per sample)"))
    end

    obs_het = heterozygosity_rate(geno; dim=1)
    exp_het_per_marker = expected_heterozygosity(geno)

    F_values = zeros(Float64, n_samples(geno))

    for i in 1:n_samples(geno)
        # Compute expected heterozygosity for this sample
        # (average over non-missing markers)
        exp_het_sum = 0.0
        n_valid = 0

        for j in 1:n_markers(geno)
            if !ismissing(geno, i, j)
                exp_het_sum += exp_het_per_marker[j]
                n_valid += 1
            end
        end

        if n_valid > 0
            exp_het_sample = exp_het_sum / n_valid
            if exp_het_sample > 0
                F_values[i] = 1.0 - (obs_het[i] / exp_het_sample)
            end
        end
    end

    return F_values
end
