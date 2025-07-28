# PLINK file format I/O

module PlinkIO

using DataFrames
using ProgressMeter
using SnpArrays
using ..CoreTypes

using ..Constants

export read_plink

"""
    read_plink(bed_path::String)

Read PLINK binary format files (.bed, .bim, .fam) into a GenotypeData object.
Uses the SnpArrays.jl package for efficient reading.
"""
function read_plink(bed_path::String)
    # Check for .bed extension and construct paths for .bim and .fam
    stem = splitext(bed_path)[1]
    bim_path = stem * ".bim"
    fam_path = stem * ".fam"

    if !isfile(bed_path) || !isfile(bim_path) || !isfile(fam_path)
        error("One or more PLINK files (.bed, .bim, .fam) not found for stem: \$stem")
    end

    @info "Reading PLINK files from stem: \$stem"

    # Use SnpArrays to memory-map the .bed file
    snp_array = SnpArray(bed_path)

    # Convert to a standard integer matrix (0, 1, 2, -9 for missing)
    n_individuals, n_markers = size(snp_array)
    genotypes = zeros(Int8, n_individuals, n_markers)

    @showprogress "Converting genotypes..." for i in 1:n_individuals
        for j in 1:n_markers
            val = snp_array[i, j]
            if ismissing(val)
                genotypes[i, j] = MISSING_GENOTYPE
            else
                genotypes[i, j] = val
            end
        end
    end

    # Read .bim (marker info) and .fam (sample info) files
    marker_info = CSV.read(bim_path, DataFrame, header=[:CHR, :ID, :CM, :POS, :A1, :A2])
    sample_info = CSV.read(fam_path, DataFrame, header=[:FID, :IID, :PAT, :MAT, :SEX, :PHENOTYPE])

    # Calculate initial allele frequencies
    allele_freq = vec(mean(filter(g -> g >= 0, genotypes), dims=1)) ./ 2

    # Create GenotypeData object
    geno_data = GenotypeData(
        genotypes,
        allele_freq,
        DEFAULT_MAF_FILTER,
        ones(n_markers), # Placeholder for call rate
        n_individuals,
        n_markers,
        marker_info,
        sample_info,
        PLOIDY_DIPLOID
    )

    @info "Finished reading PLINK data. Found \$n_individuals individuals and \$n_markers markers."

    return geno_data
end
