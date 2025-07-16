# ===== src/utils.jl =====
"""
Utility functions and helpers for the DynamicEpistasisGBLUP package.
"""
module Utils

export update_allele_frequencies!, impute_missing_genotypes!

using CUDA
using SparseArrays
using LinearAlgebra
using ..DynamicEpistasisGBLUP.Types
using ..DynamicEpistasisGBLUP.GPUSupport
using ..DynamicEpistasisGBLUP.GPUKernels

"""
    update_allele_frequencies!(genotypes::GenotypeMatrix)

Updates the `allele_freq` field of a `GenotypeMatrix` based on its current genotype data.
This version assumes missing values have been imputed or are not present.
"""
function update_allele_frequencies!(genotypes::GenotypeMatrix)
    # Efficiently sum each SNP column on the GPU
    sums = sum(genotypes.data, dims=1)

    # Allele frequency = sum_of_alleles / (2 * num_individuals)
    # Note: This assumes no missing data. Imputation should be done before this.
    genotypes.allele_freq = vec(sums) ./ (2 * genotypes.n_individuals)
end

"""
    impute_missing_genotypes!(genotypes::GenotypeMatrix; use_gpu::Bool=true)

Imputes missing genotype values in `genotypes.data` using the mean genotype method.
Missing values for a SNP `j` are replaced with `2 * p_j`, where `p_j` is the
allele frequency for that SNP.

This function modifies `genotypes.data` in-place.

# Arguments
- `genotypes::GenotypeMatrix`: The genotype data to be imputed.
- `use_gpu::Bool`: If true, perform imputation on the GPU.
"""
function impute_missing_genotypes!(genotypes::GenotypeMatrix; use_gpu::Bool=true, allele_freq_source=nothing)
    missing_mask = genotypes.missing_mask
    n_missing = nnz(missing_mask)

    if n_missing == 0
        return # Nothing to impute
    end

    freqs_to_use = allele_freq_source !== nothing ? allele_freq_source : genotypes.allele_freq

    if use_gpu && CUDA.functional()
        # Get row and column indices of missing values from the sparse mask
        rows, cols, _ = findnz(missing_mask)

        backend = get_backend(genotypes.data)
        kernel! = impute_genotypes_kernel!(backend)
        kernel!(genotypes.data, CuArray(rows), CuArray(cols), freqs_to_use, n_missing, ndrange=n_missing)
        synchronize(backend)
    else
        # CPU Fallback
        data_cpu = Array(genotypes.data)
        freq_cpu = allele_freq_source !== nothing ? Array(freqs_to_use) : Array(genotypes.allele_freq)
        missing_mask_cpu = SparseMatrixCSC(missing_mask)
        rows, cols, _ = findnz(missing_mask_cpu)

        for i in 1:n_missing
            row, col = rows[i], cols[i]
            data_cpu[row, col] = 2 * freq_cpu[col]
        end
        genotypes.data = CuArray(data_cpu) # Move back to GPU
    end
end

end # module Utils
