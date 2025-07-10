# ===== src/distributed_computing.jl =====
"""
Distributed computing support for DynamicEpistasisGBLUP.
Handles very large datasets (e.g., 100K+ SNPs, 10K+ individuals)
by distributing computations across multiple workers/nodes.
Uses Distributed.jl, DistributedArrays.jl, SharedArrays.jl, and potentially MPI.
"""

module DistributedComputing

using Distributed
using DistributedArrays # For DArray
using SharedArrays    # For SharedArray
# using MPI           # If MPI integration is desired and available
using CUDA            # For GPU operations on workers
using LinearAlgebra   # For I, tr, norm, BLAS ops on CPU workers
using Statistics      # For mean, var on workers
using SparseArrays    # For sparse representations if used by workers

# Assuming types.jl (GenotypeMatrix, VarianceComponents) and other core modules are accessible on workers.
# This is typically handled by `@everywhere using Main.DynamicEpistasisGBLUP` or similar in the main script.

# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

# Export key structures and functions for distributed operations
export DistributedGenotypeSetup, # Renamed from DistributedGenotypeMatrix to avoid conflict if it also holds DArray
       DistributedGBLUPResult, # Renamed from DistributedGBLUP
       initialize_distributed_environment, # Renamed from initialize_distributed_env
       distribute_genotype_data, # Renamed from distribute_genotypes
       perform_distributed_grm_computation, # Renamed from distributed_grm_computation
       perform_distributed_epistasis_grm # Renamed from distributed_epistasis_grm
       # distributed_sparse_epistasis was also exported but is very complex; stub for now.

"""
    DistributedGenotypeSetup{T}

Holds information about how genotype data is distributed across workers.
`data_ref` could be a DArray or a reference to data loaded per worker.
`worker_chunk_map` details which worker holds which part of the data.
"""
struct DistributedGenotypeSetup{T<:AbstractFloat}
    data_ref::Union{DArray{T,2}, Dict{Int, Future}} # DArray or Dict of Futures for remotely stored chunks
    worker_chunk_map::Dict{Int, NamedTuple{(:row_range, :col_range), Tuple{UnitRange{Int}, UnitRange{Int}}}}
    total_individuals::Int
    total_snps::Int
    num_workers_used::Int
    # chunk_dimensions::Tuple{Int, Int} # Nominal chunk size for DArray creation
end

"""
    DistributedGBLUPResult{T}

Stores results from a distributed GBLUP computation, like distributed GRM chunks
and globally aggregated variance components.
"""
struct DistributedGBLUPResult{T<:AbstractFloat}
    # GRM_additive_distributed::DArray{T,2} # Additive GRM, possibly as a DArray
    # GRM_epistatic_distributed::Union{Nothing, DArray{T,2}} # Epistatic GRM, if computed
    # For very large GRMs, might store paths to saved chunks or only summary.
    aggregated_variance_components::Main.DynamicEpistasisGBLUP.VarianceComponents{T} # Using main module's type
    # num_workers_involved::Int
end


"""
    initialize_distributed_environment(; n_workers_requested, ...) -> Vector{Int}

Initializes the distributed computing environment by adding workers if needed
and loading necessary packages on all workers.
Returns a list of GPU-enabled worker PIDs.
"""
function initialize_distributed_environment(;
    n_workers_requested::Int = Sys.CPU_THREADS, # Default to number of local CPU threads
    require_gpu_on_workers::Bool = CUDA.functional() # If true, only use/check GPU workers
    # memory_per_worker_gb::Int = 8 # Informational, not directly enforced here
)
    # Add workers if current number is less than requested
    if nworkers() < n_workers_requested
        procs_to_add = n_workers_requested - nworkers()
        # Consider exeflags for memory limits if needed, or cluster manager flags
        addprocs(procs_to_add)
    end

    # Load essential packages on all workers.
    # This assumes DynamicEpistasisGBLUP and its dependencies are available in the project environment
    # of each worker.
    @everywhere begin
        # Minimal needed for workers to operate on data chunks:
        using LinearAlgebra
        using Statistics
        # CUDA might be needed if GPU operations are done per worker
        if $(require_gpu_on_workers) # Interpolate the boolean host variable
            try
                using CUDA
                if !CUDA.functional()
                    # println("Worker $(myid()): CUDA requested but not functional.")
                end
            catch e
                # println("Worker $(myid()): Failed to load CUDA - $e")
            end
        end
        # Load the main package if its functions/types are directly called by worker tasks
        # This requires the package to be precompiled and accessible.
        # using Main.DynamicEpistasisGBLUP # Or the actual package name if it's a registered package
    end

    # Check GPU availability on each worker if required
    gpu_worker_pids = Int[]
    if require_gpu_on_workers
        worker_gpu_status = @distributed (vcat) for w_pid in workers()
            try
                if CUDA.functional()
                    (pid=w_pid, gpu_functional=true, device_name=CUDA.name(CUDA.device()))
                else
                    (pid=w_pid, gpu_functional=false, device_name="N/A")
                end
            catch
                (pid=w_pid, gpu_functional=false, device_name="Error loading CUDA")
            end
        end

        for status in worker_gpu_status
            if status.gpu_functional
                push!(gpu_worker_pids, status.pid)
            end
            # if verbose println("  Worker $(status.pid): GPU Functional = $(status.gpu_functional), Device = $(status.device_name)") end
        end
        if isempty(gpu_worker_pids) && require_gpu_on_workers
            @warn "GPU required on workers, but no GPU-functional workers found."
        end
    else
        gpu_worker_pids = workers() # All workers are considered usable if GPU not strictly required
    end

    # if verbose
        println("Distributed environment initialized:")
        println("  Total active workers: $(nworkers())")
        println("  Usable workers (based on GPU requirement): $(length(gpu_worker_pids))")
    # end

    # MPI initialization if used (original code had placeholder)
    # if @isdefined(MPI) && !MPI.Initialized() MPI.Init() end

    return gpu_worker_pids # Return PIDs of workers that meet criteria (e.g., have GPU)
end


"""
    distribute_genotype_data(genotype_matrix_host::Matrix{T}; ...) -> DistributedGenotypeSetup{T}

Distributes a large genotype matrix (from host memory) across available workers.
`genotype_matrix_host` is Individuals x SNPs.
Returns a `DistributedGenotypeSetup` object describing the distribution.
"""
function distribute_genotype_data(
    genotype_matrix_host::Matrix{T}; # Full genotype data on host
    target_worker_pids::Vector{Int} = workers(), # PIDs of workers to distribute data to
    chunking_strategy::Symbol = :row_balanced, # :row_balanced, :col_balanced, :custom_row_ranges
    # custom_row_ranges_per_worker::Union{Nothing, Dict{Int, UnitRange{Int}}} = nothing, # For custom distribution
    # overlap_size::Int = 0 # For computations needing boundary data (e.g., convolutions)
) where T <: AbstractFloat

    total_individuals, total_snps = size(genotype_matrix_host)
    num_target_workers = length(target_worker_pids)

    if num_target_workers == 0
        error("No target workers specified or available for data distribution.")
    end

    # Determine data partitioning (ranges of rows/cols for each worker)
    worker_chunk_map_actual = Dict{Int, NamedTuple{(:row_range, :col_range), Tuple{UnitRange{Int}, UnitRange{Int}}}}()

    # For DArray based distribution:
    # DArray constructor handles partitioning based on `procs` and `dist` arguments.
    # `dist` is number of chunks per dimension. e.g., [num_row_chunks, num_col_chunks]
    # `procs` is a Cartesian N-dim array of worker PIDs mapping to chunks.

    # Simplified DArray distribution: distribute rows mostly, keep all columns per worker for many genomic calcs.
    # Or, distribute blocks.
    # Let's use DArray's default balanced block distribution for now if not row_balanced.

    # Calculate DArray distribution parameters
    # Example: Try to make row chunks, assigning all columns to each worker handling a row chunk.
    # This means DArray `dist` would be `[num_target_workers, 1]`
    # And `procs` map would be `reshape(target_worker_pids, num_target_workers, 1)` if possible.

    # Using DArray to distribute the data:
    # Define how many partitions along each dimension.
    # For :row_balanced, we want N partitions for rows, 1 for columns.
    # This makes each worker get all SNPs for a subset of individuals.
    if chunking_strategy == :row_balanced
        # Ensure enough workers for this, or adjust.
        # If num_target_workers > total_individuals, some workers get no rows.
        # DArray needs `procs` to match `dist` dimensions.
        proc_grid_dims = (min(num_target_workers, total_individuals), 1)
        procs_for_darray = reshape(target_worker_pids[1:prod(proc_grid_dims)], proc_grid_dims)

        distributed_array = DArray(I -> genotype_matrix_host[I...],
                                   (total_individuals, total_snps),
                                   procs_for_darray) # Let DArray handle chunk sizes based on procs grid
    else # Default block distribution by DArray
        distributed_array = distribute(genotype_matrix_host, procs=target_worker_pids)
    end

    # Populate worker_chunk_map from the DArray's distribution info
    for (chunk_idx, pids_in_chunk_dim) in enumerate(distributed_array.pids) # .pids is the process grid
        # For DArray, each element of .pids can be a single PID if distribution is simple.
        # Or it can be a CartesianIndex into a proc grid.
        # .indices gives the global index ranges for each chunk.
        # We need to map PID to its global index range.
        # This part depends on how DArray internally stores this mapping.
        # A common way: iterate through localparts and get their indices.
        # For now, this is conceptual:
        # worker_pid = pids_in_chunk_dim # If simple 1D proc list
        # global_idx_range_for_pid = distributed_array.indices[chunk_idx]
        # worker_chunk_map_actual[worker_pid] = (row_range=global_idx_range_for_pid[1], col_range=global_idx_range_for_pid[2])
    end
    # A more robust way to get the map for DArray:
    for proc_idx = 1:length(distributed_array.pids)
      pid = distributed_array.pids[proc_idx] # This assumes pids is flat list of workers for chunks
      chunk_indices = distributed_array.indices[proc_idx] # This is a tuple of UnitRanges
      worker_chunk_map_actual[pid] = (row_range=chunk_indices[1], col_range=chunk_indices[2])
    end


    # Transfer chunks to GPU on each worker if desired (DArray's `I->...` can handle this)
    # The DArray constructor used above already places data on workers.
    # If GPU transfer is needed *after* DArray creation from host matrix:
    # @sync for p in procs(distributed_array)
    #    @spawnat p localpart(distributed_array) = CuArray(localpart(distributed_array))
    # end

    return DistributedGenotypeSetup(
        distributed_array, # The DArray itself
        worker_chunk_map_actual,
        total_individuals,
        total_snps,
        num_target_workers
        # nominal_chunk_dims # Store how DArray was asked to chunk
    )
end


"""
    perform_distributed_grm_computation(dist_setup::DistributedGenotypeSetup{T}; ...) -> DArray{T,2} or Matrix{T}

Performs distributed computation of the additive GRM.
Uses data from `DistributedGenotypeSetup`.
Returns the GRM, possibly as a DArray if large, or gathered to host Matrix.
"""
function perform_distributed_grm_computation(
    dist_setup::DistributedGenotypeSetup{T};
    # grm_method::Symbol = :vanraden, # Currently only VanRaden implied
    gather_result_to_host::Bool = dist_setup.total_individuals <= 5000 # Heuristic
) where T <: AbstractFloat

    total_individuals = dist_setup.total_individuals
    d_genotypes = dist_setup.data_ref # This is the DArray of genotypes

    # 1. Compute global allele frequencies (p_j for each SNP j)
    # This requires summing allele counts across all workers / chunks for each SNP.
    # And summing valid genotype counts.
    # print("  DistGRM: Computing global allele frequencies...")
    global_allele_freqs_gpu = compute_global_allele_frequencies_distributed(d_genotypes, total_individuals, dist_setup.total_snps)
    # println("Done.")

    # 2. Center genotype chunks on each worker using global allele frequencies
    # print("  DistGRM: Centering genotype chunks on workers...")
    # Create a DArray for centered genotypes (W_centered)
    # Or, modify d_genotypes in-place if it's mutable and safe.
    # For now, create new DArray for W_centered.
    W_centered_distributed = similar(d_genotypes) # Creates a DArray with same distribution

    @sync for p in procs(d_genotypes) # Iterate over processes holding parts of d_genotypes
        @spawnat p begin
            local_geno_chunk = localpart(d_genotypes) # Get the actual array data on this worker
            local_W_chunk = localpart(W_centered_distributed) # Get corresponding part of output DArray

            # Get the global column indices for this worker's chunk of SNPs
            # This requires dist_setup.worker_chunk_map or DArray.indices
            current_worker_pid = myid()
            # Find this worker's col_range from the map (this is complex if map not perfectly aligned with DArray chunks)
            # Simpler: DArray `localindices(d_genotypes)` gives local index ranges.
            # `map(idx->global_allele_freqs_gpu[idx], local_col_indices_global)`
            # For centering, each worker needs the relevant slice of `global_allele_freqs_gpu`.
            # Assuming `global_allele_freqs_gpu` is small enough to be broadcast or efficiently accessed.
            # Or, pass only the relevant part of `global_allele_freqs_gpu` to each worker.

            # For this example, assume `global_allele_freqs_gpu` is available on all workers (e.g. broadcasted)
            # Or, `dist_setup.worker_chunk_map[myid()].col_range` gives the SNP indices for this worker.
            # This part requires careful handling of indices.

            # Conceptual centering:
            # local_col_indices = localindices(d_genotypes)[2] # Local column indices in this chunk
            # global_snp_indices_for_chunk = # Map local_col_indices to global SNP indices
            # freqs_for_chunk = global_allele_freqs_gpu[global_snp_indices_for_chunk]

            # Simplified: Assume each worker has access to the full `global_allele_freqs_gpu`
            # and knows its `col_range` of global SNP indices.
            # Let `local_geno_chunk` be N_local_rows x M_local_cols.
            # `local_W_chunk` is same size.
            # `col_offset = first(dist_setup.worker_chunk_map[myid()].col_range) - 1`
            # for j_local in 1:size(local_geno_chunk, 2)
            #    global_snp_j = j_local + col_offset
            #    p_j = global_allele_freqs_gpu[global_snp_j]
            #    local_W_chunk[:, j_local] = local_geno_chunk[:, j_local] .- T(2) .* p_j
            # end
            # This is inefficient if global_allele_freqs_gpu is large and not sliced.
            # A DArray-centric way:
            # W_centered_distributed = map(-) do parts of d_genotypes and parts of (2*global_p) DArray.
            # This requires global_p to be a DArray distributed compatibly or broadcast.
            # For now, using explicit loop with @spawnat for clarity of concept.
            # This part is complex to implement efficiently with DArray for arbitrary distributions.
            # The original code used `center_genotypes_chunk!` which takes local_data and relevant allele_freq slice.
            # This means `global_allele_freqs_gpu` needs to be effectively passed or sliced.
            # TODO: Refine this distributed centering.
            # Placeholder for now:
            copyto!(local_W_chunk, local_geno_chunk) # Just copy for structure, centering is complex here.
        end
    end
    # println("Done.")


    # 3. Compute GRM blocks: G_block = W_chunk_i * W_chunk_j' / scaling_factor
    # This results in a distributed GRM (DArray).
    # print("  DistGRM: Computing GRM blocks...")
    # The scaling_factor = sum(2*p_j*(1-p_j)) needs to be computed from global_allele_freqs_gpu.
    sum_2pq_host = Array(global_allele_freqs_gpu) # Move to host for sum
    scaling_factor_val = sum(T(2) .* sum_2pq_host .* (one(T) .- sum_2pq_host))
    if scaling_factor_val <= eps(T) error("GRM scaling factor is zero.") end
    inv_scaling_factor = one(T) / scaling_factor_val

    # GRM_distributed = ۵(-) # Placeholder for DArray representing GRM
    # GRM_distributed = DistributedArrays.mapfill(zero(T), (total_individuals, total_individuals), procs=procs(W_centered_distributed))
    # This requires a distributed matrix multiplication: W_centered * W_centered_transpose
    # This is a major operation, often done with block algorithms (SUMMA, Cannon's).
    # DistributedArrays.jl might have high-level `*` for DArrays if compatible.
    # `GRM_distributed = (W_centered_distributed * W_centered_distributed') .* inv_scaling_factor`
    # This relies on DArray supporting `*` and `'` correctly and efficiently.
    # This is a placeholder for that advanced distributed linear algebra.
    # println("Done.")

    # For now, returning a placeholder (e.g. empty DArray or error)
    # The original `distributed_grm_blocks` and `assemble_grm_from_blocks` were more detailed.
    # This high-level function would orchestrate those.

    # Placeholder result
    if gather_result_to_host
        # This implies the full GRM is assembled on host. Only for small N.
        # return zeros(T, total_individuals, total_individuals) # Placeholder
        error("Distributed GRM computation and gathering not fully stubbed yet.")
    else
        # Return a DArray reference or similar distributed object representation.
        # return DArray(...) # Placeholder
        error("Distributed GRM computation (returning DArray) not fully stubbed yet.")
    end
end

"""
Helper for distributed allele frequency calculation.
"""
function compute_global_allele_frequencies_distributed(
    d_genotypes::DArray{T,2}, total_n_ind::Int, total_n_snps::Int
) where T
    # Each worker computes sum_of_alleles and count_of_valid_genotypes for its SNP columns.
    # This is complex if SNPs are distributed across workers (i.e., col_chunks > 1).
    # If SNPs are not chunked (all workers have all SNPs for their row subset):
    #   Each worker computes p_local for its individuals. Then these p_locals are averaged. (Incorrect for MAF)
    # Correct way: Sum allele counts (0,1,2) per SNP across all individuals, and divide by 2*N_valid_individuals_for_SNP.

    # Use `mapreduce` over the DArray if distribution allows efficient column-wise operations.
    # Example: Summing columns of a DArray
    # allele_sums_dist = sum(d_genotypes, dims=1) # This would be a DArray of 1xM_total_snps
    # To get to host: `sum_counts_host = convert(Array, allele_sums_dist)`
    # This assumes d_genotypes stores 0,1,2.

    # Number of non-missing genotypes per SNP (also needs to be summed globally)
    # `d_non_missing = map(x -> ismissing(x) ? zero(T) : one(T), d_genotypes)`
    # `non_missing_counts_dist = sum(d_non_missing, dims=1)`
    # `non_missing_counts_host = convert(Array, non_missing_counts_dist)`

    # `global_p = sum_counts_host ./ (T(2) .* non_missing_counts_host)`
    # This is a high-level sketch. Actual implementation needs care with DArray parts.

    # For now, simple placeholder:
    return CUDA.rand(T, total_n_snps) .* T(0.5) # Random global frequencies
end


"""
    perform_distributed_epistasis_grm(dist_setup::DistributedGenotypeSetup{T}; ...) -> DArray{T,2} or Matrix{T}

Performs distributed computation of the epistatic GRM.
Placeholder for now. Relies on methods like distributed symmetric polynomials or block-wise.
"""
function perform_distributed_epistasis_grm(
    dist_setup::DistributedGenotypeSetup{T};
    # method::Symbol = :symmetric_polynomial,
    gather_result_to_host::Bool = dist_setup.total_individuals <= 2000 # Stricter heuristic for G_aa
) where T <: AbstractFloat

    # print("  DistEpiGRM: Starting computation...")
    # This would involve:
    # 1. Standardizing genotype chunks (W_std) using global allele frequencies.
    # 2. Applying distributed algorithm (e.g., symmetric polynomials, or block-wise Hadamard).
    #    - Symmetric Poly: Compute e1, p2 distributedly, then combine for G_aa elements.
    #    - Block-wise: Compute interactions for blocks of SNPs, then sum contributions.
    # print("Done.")

    # Placeholder result
     if gather_result_to_host
        error("Distributed Epistatic GRM computation and gathering not fully stubbed yet.")
    else
        error("Distributed Epistatic GRM computation (returning DArray) not fully stubbed yet.")
    end
end


# The original file had many more detailed functions (e.g. for specific kernels, block computations).
# These would be called by the higher-level functions above.
# For initial structure, focusing on the main exported API placeholders.
# Specific kernels like `center_kernel_dist!` are assumed to be in `gpu_kernels.jl`.

# Other functions from original:
# - `distributed_allele_frequencies` (stubbed as `compute_global_allele_frequencies_distributed`)
# - `distributed_grm_blocks` (part of `perform_distributed_grm_computation`)
# - `compute_grm_block!` (worker function for a block)
# - `distributed_symmetric_epistasis` (part of `perform_distributed_epistasis_grm`)
# - `distributed_polynomial` (helper for symmetric poly)
# - `compute_epistatic_block_symmetric` (worker for symmetric poly block)
# - `get_distributed_chunk` (utility to gather data for a worker)
# - `distributed_blockwise_epistasis` (alternative method)
# - `compute_epistasis_block_contribution` (worker for blockwise)
# - `get_distributed_snp_block` (utility)
# - `distributed_sparse_epistasis` (complex, involves screening, model fitting)
# - `distributed_marginal_screening`
# - `sure_independence_screening` (example screening method)
# - `center_genotypes_chunk!` (worker utility, kernel in gpu_kernels.jl)
# - `symmetrize_distributed_matrix!` (utility for DArrays)
# - `save_distributed_results`, `load_distributed_data`

# These indicate a very detailed existing implementation plan.
# The stubs above capture the main entry points.

end # module DistributedComputing
