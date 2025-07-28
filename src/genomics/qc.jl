# Quality control module for genomic data

module QualityControl

using Statistics
using StatsBase
using DataFrames
using Distributions
using ProgressMeter
using ..CoreTypes
using ..Constants

export quality_control, filter_markers, filter_samples

function quality_control(geno_data::GenotypeData;
                        maf_threshold::Float64=DEFAULT_MAF_FILTER,
                        call_rate_marker_threshold::Float64=DEFAULT_CALL_RATE,
                        call_rate_sample_threshold::Float64=DEFAULT_CALL_RATE,
                        hwe_pvalue_threshold::Float64=DEFAULT_HWE_PVALUE)

    @info "Starting comprehensive quality control..."

    # --- Marker QC ---
    m = geno_data.n_markers
    marker_pass_qc = trues(m)

    # Calculate marker stats
    maf = zeros(m)
    call_rate_marker = zeros(m)
    hwe_p = ones(m)

    @showprogress "Computing marker statistics" for j in 1:m
        valid_genos = filter(g -> g >= 0, geno_data.genotypes[:, j])
        if isempty(valid_genos)
            marker_pass_qc[j] = false
            continue
        end
        call_rate_marker[j] = length(valid_genos) / geno_data.n_individuals
        p = mean(valid_genos) / 2
        maf[j] = min(p, 1-p)
        if maf[j] > 0
            hwe_p[j] = hardy_weinberg_test(valid_genos)
        end
    end

    marker_pass_qc .&= (maf .>= maf_threshold)
    marker_pass_qc .&= (call_rate_marker .>= call_rate_marker_threshold)
    marker_pass_qc .&= (hwe_p .>= hwe_pvalue_threshold)

    n_markers_removed = m - sum(marker_pass_qc)
    @info "Marker QC: Removed \$n_markers_removed markers."

    geno_data_filtered = filter_markers(geno_data, marker_pass_qc)

    # --- Sample QC ---
    n = geno_data_filtered.n_individuals
    sample_pass_qc = trues(n)

    call_rate_sample = vec(mean(geno_data_filtered.genotypes .>= 0, dims=2))
    sample_pass_qc .&= (call_rate_sample .>= call_rate_sample_threshold)

    n_samples_removed = n - sum(sample_pass_qc)
    @info "Sample QC: Removed \$n_samples_removed samples."

    geno_data_final = filter_samples(geno_data_filtered, sample_pass_qc)

    @info "QC complete. Final dimensions: \$(geno_data_final.n_individuals) individuals, \$(geno_data_final.n_markers) markers."

    return geno_data_final
end

function hardy_weinberg_test(genotypes::Vector)
    n = length(genotypes)
    n_AA = sum(genotypes .== 0)
    n_Aa = sum(genotypes .== 1)
    n_aa = sum(genotypes .== 2)

    p = (2*n_AA + n_Aa) / (2*n)
    q = 1 - p

    if p == 0 || q == 0 return 1.0 end

    e_AA = n * p^2
    e_Aa = n * 2 * p * q
    e_aa = n * q^2

    chi_sq = ((n_AA - e_AA)^2 / e_AA) + ((n_Aa - e_Aa)^2 / e_Aa) + ((n_aa - e_aa)^2 / e_aa)

    return cdf(Chisq(1), chi_sq)
end

function filter_markers(geno_data::GenotypeData, keep::BitVector)
    new_genotypes = geno_data.genotypes[:, keep]
    new_allele_freq = geno_data.allele_freq[keep]
    new_marker_info = geno_data.marker_info[keep, :]

    return GenotypeData(
        new_genotypes, new_allele_freq, geno_data.maf_filter,
        geno_data.call_rate[keep], geno_data.n_individuals, sum(keep),
        new_marker_info, geno_data.sample_info, geno_data.ploidy
    )
end

function filter_samples(geno_data::GenotypeData, keep::BitVector)
    new_genotypes = geno_data.genotypes[keep, :]
    new_sample_info = geno_data.sample_info[keep, :]

    # Allele frequencies should be recalculated after filtering samples
    new_allele_freq = vec(mean(filter(g -> g >= 0, new_genotypes), dims=1)) ./ 2

    return GenotypeData(
        new_genotypes, new_allele_freq, geno_data.maf_filter,
        ones(sum(keep)), # call rates need recalc
        sum(keep), geno_data.n_markers,
        geno_data.marker_info, new_sample_info, geno_data.ploidy
    )
end

end
