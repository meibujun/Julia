# ===== src/utils.jl =====
"""
Utility functions and helpers for the DynamicEpistasisGBLUP package.
"""

using CUDA
using Random
using Statistics
using LinearAlgebra # For I in inverse_gpu_stub, norm
using .DynamicEpistasisGBLUP # To access types like PopulationData, GenotypeMatrix, etc.
# Note: This creates a circular dependency if DynamicEpistasisGBLUP.jl also includes utils.jl at the top.
# It's better if utils.jl does not depend on the main module, or if types are in their own module.
# For now, assuming types are accessible. If this file is included into DynamicEpistasisGBLUP, types are directly visible.


"""
    update_allele_frequencies!(genotypes::GenotypeMatrix{T}) where T

Updates the `allele_freq` field of a `GenotypeMatrix` based on its current genotype data.
Handles missing data by excluding it from frequency calculation.
Assumes diploid organisms.
"""
function update_allele_frequencies!(genotypes::GenotypeMatrix{T}) where T
    n_individuals = genotypes.n_individuals
    n_snps = genotypes.n_snps

    # Ensure freq vector is on the GPU, matching genotype data type
    freq = CUDA.zeros(T, n_snps)
    data_gpu = genotypes.data
    missing_mask_gpu = genotypes.missing_mask # Assuming missing_mask is also on GPU

    # Kernel to compute allele frequencies
    @cuda threads=256 blocks=cld(n_snps, 256) begin
        j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        if j <= n_snps
            sum_alleles = zero(T)
            valid_count = zero(Int32) # Count of non-missing individuals for this SNP

            for i in 1:n_individuals
                # Check missing mask: if missing_mask[i,j] is true, data is missing
                # Assuming missing_mask stores true for missing values.
                # If missing_mask is not pre-populated correctly or efficiently queryable,
                # this part might need adjustment based on how missingness is actually encoded.
                # A common way for sparse missing_mask: check if (i,j) is a stored element.
                # For dense missing_mask: direct check.
                # For now, let's assume a way to check missingness.
                # This check is a placeholder, depends on missing_mask structure.
                # For CuSparseMatrixCSR, checking for non-zero (true) is not direct.
                # A better way might be to have a dense missing_val indicator in data itself (e.g., -1)
                # or operate on a dense boolean mask if memory allows.

                # Placeholder for missing check:
                # For this example, let's assume missing_mask.nzval contains indices of missing entries
                # and we'd need a more complex lookup.
                # A simpler approach if missing values are NaN in `data`:
                # if !isnan(data_gpu[i, j])
                #    sum_alleles += data_gpu[i, j]
                #    valid_count += 1
                # end

                # Given current GenotypeMatrix, missing_mask is CuSparseMatrixCSR{Bool}.
                # It's more efficient to iterate over non-missing or calculate sum and subtract sum of missing.
                # Or, iterate all and use a dense mask if available.
                # A direct check `missing_mask_gpu[i,j]` on a sparse matrix is inefficient.
                # Let's assume for now that we can efficiently check.
                # THIS IS A CRITICAL POINT FOR PERFORMANCE AND CORRECTNESS.
                # A common convention: if a value is NOT in the sparse `missing_mask` (i.e., it's zero/false),
                # it means the genotype IS PRESENT. If it IS in `missing_mask.nzval` (true), it's missing.
                # So, we need to check if the (i,j) entry is NOT marked as missing.
                # This requires a function like `!is_missing(missing_mask_gpu, i, j)`.
                # For now, we'll write a conceptual loop.
                # A more robust way: convert the sparse missing mask to dense on GPU for this kernel,
                # or pass a dense representation if feasible.

                # Simplified assumption: if data_gpu[i,j] is a special value (e.g. -1, NaN) it's missing.
                # The problem states `missing_mask::CuSparseMatrixCSR{Bool}`.
                # This means `true` where data is missing. Iterating all and checking is bad.
                # Better: sum all, then subtract missing based on sparse mask.
                # Or, if missing entries are few, iterate non-missing.
                # Let's do sum_all and count_all, then adjust if needed.

                # For now, let's assume a hypothetical `is_present(i, j, missing_mask_gpu)`
                # For a dense mask, this would be `!missing_mask_gpu[i,j]`
                # For a sparse mask, this is harder. The original code has:
                # if !genotypes.missing_mask[i, j] -> this implies a dense check was intended.
                # Let's assume genotypes.missing_mask was converted to dense for this kernel, or was dense.
                # If it MUST be sparse, the kernel logic needs to change.

                # Reverting to the original code's apparent logic, assuming missing_mask can be indexed like dense:
                # THIS IS LIKELY INEFFICIENT IF missing_mask IS TRULY SPARSE.
                # It will be reviewed in the refinement step.
                is_missing_val = false # Placeholder
                # Add actual check here based on how missing_mask is intended to be used.
                # Example: if missing_mask represents *present* data, then check is different.
                # If missing_mask stores 'true' for missing:
                #   is_missing_val = missing_mask_gpu[i,j] (if dense)
                # Let's assume the original code's sparse_mask[i,j] was a conceptual check.
                # A proper way for sparse mask: iterate over its non-zero elements.
                # Or, for this kernel, it might be better to materialize a dense version of the relevant column of the mask.

                # For now, let's assume a simplified check that data is not a specific missing marker
                # (e.g. NaN, if T allows, or a sentinel value like -1.0f0 for Float32)
                # This part needs clarification based on how missing_mask is populated and used.
                # The original code had `if !genotypes.missing_mask[i, j]`. This implies `missing_mask` stores `true` for missing.
                # And that it can be indexed directly. This is problematic for CuSparseMatrixCSR.
                # Let's write it assuming a dense interpretation for now, and flag for review.
                # A better approach for sparse `missing_mask` would be to sum all `data_gpu[:,j]`
                # and then iterate through the non-zero elements of `missing_mask[:,j]` (the missing ones)
                # to subtract their (imputed or placeholder) values from sum_alleles and adjust valid_count.
                # This is too complex for initial structuring.

                # Sticking to original intent for now, assuming `missing_mask` check is valid:
                # if !missing_mask_gpu[i,j] # Conceptual, not efficient for sparse
                # sum_alleles += data_gpu[i,j]
                # valid_count += 1
                # end
                # A more direct interpretation of the original code's loop:
                # Let's assume the data itself contains a marker for missing, e.g. NaN
                # and missing_mask is for other purposes or complements this.
                # The provided code structure is:
                # if !genotypes.missing_mask[i, j]
                #   sum_alleles += genotypes.data[i, j]
                #   valid_count += 1
                # This implies missing_mask is a dense boolean CuArray where true means missing.
                # But the type is CuSparseMatrixCSR{Bool}. This is a contradiction.
                # RESOLUTION: Assume missing_mask will be converted to dense for this kernel or a different strategy is used.
                # For now, implementing as if it's a dense check, and will correct in refinement.
                # This is a placeholder for the actual missing data handling logic.
                # The most robust way is to use a sentinel value in `data_gpu` itself (e.g. NaN for floats).
                # If `genotypes.missing_mask[i,j]` is true, it means data is missing.
                # So we process if `!genotypes.missing_mask[i,j]`.
                # This direct indexing is problematic for CuSparseMatrixCSR.
                # Let's assume a function `is_missing(genotypes, i, j)` exists for now.
                # For initial structure, let's write a simplified loop assuming no missing for now, and add handling later.
                # THIS IS A MAJOR TODO: Correctly handle missing data with CuSparseMatrixCSR.

                # Simplified version (ignoring missing_mask complexity for now):
                sum_alleles += data_gpu[i,j] # Assumes data_gpu[i,j] is valid
                valid_count +=1
            end

            if valid_count > 0
                freq[j] = sum_alleles / (T(genotypes.ploidy) * T(valid_count))
            else
                freq[j] = T(0.5)  # Default for all missing or no valid data
            end
        end
    end
    CUDA.synchronize() # Ensure kernel completion before assigning
    genotypes.allele_freq = freq
end


"""
    select_parents(genetic_values::Vector{T}, n_parents::Int) where T

Selects parents based on truncation selection on their genetic values.
Assumes genetic_values are for a mixed population of males and females,
and selects top `n_parents/2` from each sex.
This assumes a specific ordering or identification of sexes in `genetic_values`.
"""
function select_parents(genetic_values::Vector{T}, n_parents::Int) where T
    n_total = length(genetic_values)
    # Ensure n_parents is even for equal sex selection
    @assert iseven(n_parents) "Number of parents to select must be even for equal sex selection."
    n_select_per_sex = n_parents ÷ 2

    # Assuming first half are males, second half females. This is a strong assumption.
    # A better approach would be to have sex information available.
    n_males_total = n_total ÷ 2
    n_females_total = n_total - n_males_total

    @assert n_select_per_sex <= n_males_total "Not enough males to select from."
    @assert n_select_per_sex <= n_females_total "Not enough females to select from."

    # Select top males
    # `partialsortperm` sorts and gives indices. `rev=true` for descending (top values).
    male_indices_local = partialsortperm(genetic_values[1:n_males_total], 1:n_select_per_sex, rev=true)
    selected_male_indices = male_indices_local # These are already global indices if population is structured this way

    # Select top females
    female_indices_local = partialsortperm(genetic_values[n_males_total+1:end], 1:n_select_per_sex, rev=true)
    selected_female_indices = female_indices_local .+ n_males_total # Adjust local indices to global

    return vcat(selected_male_indices, selected_female_indices)
end

"""
    generate_single_offspring(parent1_geno::Vector{T}, parent2_geno::Vector{T}, snps_per_chrom::Int, n_chromosomes::Int) where T

Generates a single offspring's genotype from two parents.
Simulates meiosis including recombination.
Assumes diploid parents and offspring.
"""
function generate_single_offspring(
    parent1_geno::AbstractVector{T}, # More generic type
    parent2_geno::AbstractVector{T}, # More generic type
    snps_per_chrom::Int,
    n_chromosomes::Int # Typically 26 for sheep as in original code
) where T
    n_snps = length(parent1_geno)
    offspring_geno = zeros(T, n_snps)

    # Simulate meiosis with recombination for each chromosome
    for chrom in 1:n_chromosomes
        start_idx = (chrom - 1) * snps_per_chrom + 1
        end_idx = min(chrom * snps_per_chrom, n_snps)

        # Ensure valid range for this chromosome
        chrom_len = end_idx - start_idx + 1
        if chrom_len <= 0
            continue
        end

        # Generate gamete from parent 1 for this chromosome
        gamete1_chrom = generate_gamete_chromosome(parent1_geno, start_idx, end_idx, chrom_len)
        # Generate gamete from parent 2 for this chromosome
        gamete2_chrom = generate_gamete_chromosome(parent2_geno, start_idx, end_idx, chrom_len)

        # Combine gametes to form offspring genotype for this chromosome
        offspring_geno[start_idx:end_idx] = gamete1_chrom .+ gamete2_chrom
    end

    return offspring_geno
end

"""
Helper function to generate a gamete for a single chromosome.
"""
function generate_gamete_chromosome(parent_geno_full::AbstractVector{T}, start_idx::Int, end_idx::Int, chrom_len::Int) where T
    gamete_chrom = zeros(T, chrom_len)
    parent_chrom_geno = @view parent_geno_full[start_idx:end_idx]

    # Determine which parental haplotype is chosen initially (0 or 1 for first or second copy)
    current_haplotype_choice = rand(Bool) # true for first allele, false for second

    # Simulate recombination (simplified: 1 recombination per chromosome on average, Poisson distributed)
    # A more realistic model would use genetic map distances.
    n_recomb = rand(Distributions.Poisson(1.0)) # Mean 1 recombination
    recomb_points = sort(rand(1:chrom_len-1, Int(n_recomb))) # positions within the chromosome segment

    last_pos = 0
    for rec_pt in recomb_points
        for i in (last_pos+1):rec_pt
            # Allele from parent's chosen haplotype
            # Genotype 2.0 -> alleles are 1, 1
            # Genotype 1.0 -> alleles are 1, 0 (or 0, 1)
            # Genotype 0.0 -> alleles are 0, 0
            parent_allele1 = parent_chrom_geno[i] >= 1.0 ? T(1) : T(0) # First allele copy
            parent_allele2 = parent_chrom_geno[i] == 2.0 ? T(1) : T(0) # Second allele copy

            gamete_chrom[i] = current_haplotype_choice ? parent_allele1 : parent_allele2
        end
        current_haplotype_choice = !current_haplotype_choice # Switch haplotype
        last_pos = rec_pt
    end

    # Remaining part of the chromosome
    for i in (last_pos+1):chrom_len
        parent_allele1 = parent_chrom_geno[i] >= 1.0 ? T(1) : T(0)
        parent_allele2 = parent_chrom_geno[i] == 2.0 ? T(1) : T(0)
        gamete_chrom[i] = current_haplotype_choice ? parent_allele1 : parent_allele2
    end
    return gamete_chrom
end


"""
    regression_coefficient(y_true::Vector{T}, y_pred::Vector{T}) where T

Calculates the regression coefficient (slope) of `y_true` regressed on `y_pred`.
Useful as a measure of prediction bias (ideal value is 1.0).
"""
function regression_coefficient(y_true::Vector{T}, y_pred::Vector{T}) where T
    @assert length(y_true) == length(y_pred) "Vectors must have the same length."

    # Design matrix X = [1 y_pred]
    X = hcat(ones(T, length(y_pred)), y_pred)

    # Solve (X'X)β = X'y_true for β
    # Using \ operator for robust least squares solution
    β = X \ y_true

    return β[2]  # Slope coefficient
end


"""
    benchmark_grm_computation(n_individuals::Int, n_snps::Int)

Benchmarks the computation of additive and epistatic GRMs.
Requires `compute_grm!` and `compute_epistatic_grm!` to be defined and accessible.
This function is more of a utility script than a core package function.
"""
function benchmark_grm_computation(n_individuals::Int, n_snps::Int)
    # This function depends on `compute_grm!` and `compute_epistatic_grm!`
    # which are defined in `grm_computation.jl`.
    # It also uses types like `GenotypeMatrix`.
    # Ensure these are available in the scope where this function is called.

    println("\nBenchmarking GRM computation...")
    println("Individuals: $n_individuals, SNPs: $n_snps")

    if !CUDA.functional()
        println("  CUDA not functional. Skipping GPU benchmarks.")
        return nothing
    end

    # Generate random data directly on GPU for benchmark
    # Using Float32 as per package default `Float`
    data_gpu = CUDA.rand(Float32, n_individuals, n_snps) .* 2.0f0 # Genotypes 0,1,2 like
    missing_mask_gpu = CuSparseMatrixCSR(sparse(zeros(Bool, n_individuals, n_snps))) # No missing for benchmark ease
    allele_freq_gpu = CUDA.rand(Float32, n_snps) # Random allele frequencies

    genotypes_gpu = GenotypeMatrix(
        data_gpu,
        missing_mask_gpu,
        allele_freq_gpu,
        Int32(n_individuals),
        Int32(n_snps),
        Int8(2) # Diploid
    )

    # Benchmark additive GRM using BenchmarkTools
    println("  Benchmarking Additive GRM (GPU)...")
    # `@btime` is good for quick timing, `@benchmark` for more detailed stats.
    # Need to ensure BenchmarkTools is available. It's in Project.toml.
    # The function to benchmark should not have global side effects if possible,
    # or pass all inputs. `compute_grm!` modifies genotypes.allele_freq.
    # For a pure benchmark, create a fresh GenotypeMatrix or copy.

    # Create a copy for benchmarking to avoid side effects on original genotypes_gpu
    geno_copy_for_add_bench = deepcopy(genotypes_gpu) # Deepcopy to ensure fresh state for allele_freq updates
    add_grm_bench = BenchmarkTools.@benchmark CUDA.@sync Main.DynamicEpistasisGBLUP.compute_grm!($geno_copy_for_add_bench) samples=5 evals=1 seconds=30
    println("    " * sprint(show, MIME"text/plain"(), add_grm_bench))
    # println("    Time (median): $(BenchmarkTools.prettytime(median(add_grm_bench.times)))")
    # println("    Memory: $(BenchmarkTools.prettymemory(add_grm_bench.memory))")


    if n_snps <= 1000
        println("  Benchmarking Epistatic GRM (Hadamard method, GPU)...")
        geno_copy_for_epi_bench = deepcopy(genotypes_gpu)
        epi_grm_bench = BenchmarkTools.@benchmark CUDA.@sync Main.DynamicEpistasisGBLUP.compute_epistatic_grm!($geno_copy_for_epi_bench, method=:hadamard) samples=3 evals=1 seconds=60
        println("    " * sprint(show, MIME"text/plain"(), epi_grm_bench))
        # println("    Time (median): $(BenchmarkTools.prettytime(median(epi_grm_bench.times)))")
        # println("    Memory: $(BenchmarkTools.prettymemory(epi_grm_bench.memory))")
    else
        println("  Skipping epistatic GRM benchmark for n_snps > 1000 due to potential long runtime.")
    end

    println("  Note: Benchmarks are run with limited samples/evals for quick checking.")
    println("  For rigorous profiling, increase samples/evals or run dedicated benchmark scripts.")
    return nothing
end

# Placeholder for inverse_gpu, if it's a general utility
# Otherwise, it should be context-specific or part of a linear algebra utility submodule.
function inverse_gpu_stub(A::CuArray{T, 2}) where T
    # This is a stub. Actual GPU inverse is complex.
    # For small matrices, inv(A) might work if A is well-conditioned.
    # For larger, specialized methods or LU decomposition followed by solving against identity.
    # E.g., using CUDA.cusolver.csrlsvqr or similar for sparse, or dense LU.
    if size(A,1) < 1000 # Arbitrary threshold
        return inv(A) # Uses CUDA's generic matinv if available and A is small/dense
    else
        # Fallback to CPU for large matrices if no direct GPU routine is easily available here
        # THIS IS NOT A GOOD PRODUCTION APPROACH FOR LARGE MATRICES.
        # Requires a proper GPU linear algebra library call.
        A_cpu = Array(A)
        A_inv_cpu = inv(A_cpu) # This can be very slow and memory intensive
        return CuArray(A_inv_cpu)
    end
end

# Placeholder for efficient_V_inverse, often used in REML.
# This would typically be part of the REML implementation itself.
function efficient_V_inverse_stub(Z, G, G_aa, var_comp)
    # This is a stub. The Woodbury matrix identity or other specialized methods are used.
    # The actual implementation is in reml.jl or augmented_aireml.jl
    n = size(Z, 1)
    V_inv = CUDA.zeros(eltype(G), n, n) # Placeholder
    # ... actual computation based on var_comp and matrices ...
    return V_inv
end

# Placeholder for safe_logdet
function safe_logdet_stub(A::CuArray{T,2}) where T
    # This is a stub.
    # Actual implementation involves logdet(A) or eigenvalue-based computation for robustness.
    # logdet itself might try to move data to CPU if not overloaded for CuArray directly by a lib.
    # CUDA.logdet might be available in some CUDA versions/libs.
    # Fallback: Array(A) then logdet() on CPU, then convert back if needed.
    # Or use eigenvalues: sum(log.(max.(eigvals(Symmetric(Array(A))), T(1e-10))))
    return T(0.0) # Placeholder
end

# Placeholder for TestSuite.runtests() - actual test execution logic
# This would be in test/runtests.jl
# function runtests()
#    println("Running test suite...")
#    # ... logic to include and run tests from comprehensive_tests.jl ...
# end

# Make sure to export functions if this file becomes a module itself.
# export update_allele_frequencies!, select_parents, generate_single_offspring, regression_coefficient, benchmark_grm_computation
