# Quality control module for genomic data

module QualityControl

using Statistics
using ..CoreTypes

export quality_control

function quality_control(geno_data::GenotypeData;
                        maf_threshold::Float64=0.01,
                        call_rate_threshold::Float64=0.95)

    # This is a placeholder for a full QC pipeline.
    # A real implementation would filter markers and samples.

    @info "Performing basic quality control..."

    # Example: MAF filter
    initial_markers = geno_data.n_markers

    maf = [min(p, 1-p) for p in geno_data.allele_freq]
    keep_markers = maf .>= maf_threshold

    geno_data_filtered = filter_markers(geno_data, keep_markers)

    @info "Removed \$(initial_markers - geno_data_filtered.n_markers) markers due to MAF < \$maf_threshold"

    return geno_data_filtered
end

function filter_markers(geno_data::GenotypeData, keep::BitVector)

    new_genotypes = geno_data.genotypes[:, keep]
    new_allele_freq = geno_data.allele_freq[keep]
    new_marker_info = geno_data.marker_info[keep, :]

    return GenotypeData(
        new_genotypes,
        new_allele_freq,
        geno_data.maf_filter,
        geno_data.call_rate[keep],
        geno_data.n_individuals,
        sum(keep),
        new_marker_info,
        geno_data.sample_info,
        geno_data.ploidy
    )
end

end
