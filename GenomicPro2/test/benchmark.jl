"""
GenomicPro2 Performance Benchmark Suite

Comprehensive benchmarks for all major operations.

Usage:
    julia --project=. test/benchmark.jl
    julia --threads=8 --project=. test/benchmark.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random

# Benchmark utilities
struct BenchmarkResult
    name::String
    mean_time::Float64
    std_time::Float64
    min_time::Float64
    max_time::Float64
    n_runs::Int
    throughput::Union{Float64,Nothing}
    memory_mb::Union{Float64,Nothing}
end

function Base.show(io::IO, br::BenchmarkResult)
    @printf(io, "%-40s: %.3f ± %.3f ms", br.name, br.mean_time * 1000, br.std_time * 1000)
    if !isnothing(br.throughput)
        @printf(io, " (%.2f ops/s)", br.throughput)
    end
    if !isnothing(br.memory_mb)
        @printf(io, " [%.2f MB]", br.memory_mb)
    end
end

"""
    benchmark(f::Function, name::String; n_runs::Int=10, warmup::Int=2) -> BenchmarkResult

Benchmark a function with multiple runs and statistics.
"""
function benchmark(f::Function, name::String; n_runs::Int=10, warmup::Int=2,
                  calc_throughput::Union{Function,Nothing}=nothing)
    # Warmup runs
    for _ in 1:warmup
        f()
    end

    # GC before timing
    GC.gc()

    # Timed runs
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t)
    end

    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)

    throughput = isnothing(calc_throughput) ? nothing : calc_throughput(mean_time)

    return BenchmarkResult(
        name, mean_time, std_time, min_time, max_time, n_runs, throughput, nothing
    )
end

function print_header(text::String)
    println("\n" * "="^80)
    printstyled(text * "\n"; color=:cyan, bold=true)
    println("="^80)
end

function print_section(text::String)
    println("\n" * "-"^80)
    printstyled(text * "\n"; color=:yellow, bold=true)
    println("-"^80)
end

print_header("GenomicPro2 Performance Benchmarks")

println("Julia Version: $(VERSION)")
println("Threads: $(Threads.nthreads())")
println("CPU Info: $(Sys.cpu_info()[1].model)")
println()

Random.seed!(123)

# ============================================================================
# Benchmark 1: Genotype Data Creation and Access
# ============================================================================

print_section("Genotype Data Operations")

# Different dataset sizes
sizes = [
    (100, 1000, "Small"),
    (500, 5000, "Medium"),
    (1000, 10000, "Large")
]

results = BenchmarkResult[]

for (n_samples, n_markers, label) in sizes
    println("\n  Dataset: $label ($n_samples samples × $n_markers markers)")

    geno_data = rand(0:2, n_samples, n_markers)
    sample_ids = [string("S", i) for i in 1:n_samples]
    marker_ids = [string("M", i) for i in 1:n_markers]

    # Benchmark creation
    result = benchmark("  Create CompactGenotypes ($label)", n_runs=5) do
        CompactGenotypes(geno_data, sample_ids, marker_ids)
    end
    println(result)
    push!(results, result)

    # Create once for access benchmarks
    geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

    # Benchmark access
    result = benchmark("  Random access ($label)", n_runs=20) do
        for _ in 1:1000
            i = rand(1:n_samples)
            j = rand(1:n_markers)
            get_genotype(geno, i, j)
        end
    end
    println(result)
    push!(results, result)

    # Benchmark to_matrix
    result = benchmark("  to_matrix ($label)", n_runs=5) do
        to_matrix(geno; impute=true)
    end
    println(result)
    push!(results, result)
end

# ============================================================================
# Benchmark 2: GRM Computation
# ============================================================================

print_section("GRM Computation")

# Create test genotype data
n_samples = 1000
n_markers = 10000

geno_data = rand(0:2, n_samples, n_markers)
sample_ids = [string("S", i) for i in 1:n_samples]
marker_ids = [string("M", i) for i in 1:n_markers]
geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

println("\n  Dataset: $n_samples samples × $n_markers markers")

# VanRaden method
result = benchmark("  GRM VanRaden (single-thread)", n_runs=3) do
    compute_grm(geno; method=:vanraden, min_maf=0.0)
end
println(result)
push!(results, result)

# Parallel GRM
result = benchmark("  GRM VanRaden (parallel)", n_runs=3) do
    compute_grm_parallel(geno; method=:vanraden, min_maf=0.0, use_threads=true)
end
println(result)
push!(results, result)

# ============================================================================
# Benchmark 3: GBLUP Model
# ============================================================================

print_section("GBLUP Model Fitting")

# Generate phenotypes
y = randn(n_samples)
pheno = PhenotypeData(sample_ids, ["Trait1"], reshape(y, n_samples, 1))

G = compute_grm(geno; method=:vanraden, min_maf=0.0)

# Cholesky solver
result = benchmark("  GBLUP fit (Cholesky)", n_runs=5) do
    model = GBLUPModel(method=:cholesky, estimate_variances=false)
    fit!(model, geno, pheno; G=G, verbose=false)
end
println(result)
push!(results, result)

# PCG solver
result = benchmark("  GBLUP fit (PCG)", n_runs=5) do
    model = GBLUPModel(method=:pcg, estimate_variances=false)
    fit!(model, geno, pheno; G=G, verbose=false)
end
println(result)
push!(results, result)

# Prediction
model = GBLUPModel(method=:cholesky, estimate_variances=false)
fit!(model, geno, pheno; G=G, verbose=false)

result = benchmark("  GBLUP predict", n_runs=10) do
    predict(model, geno)
end
println(result)
push!(results, result)

# ============================================================================
# Benchmark 4: Quality Control
# ============================================================================

print_section("Quality Control Operations")

# Reset with original data
geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

result = benchmark("  MAF calculation", n_runs=10) do
    minor_allele_frequency(geno)
end
println(result)
push!(results, result)

result = benchmark("  Missing rate calculation", n_runs=10) do
    missing_rate(geno; dim=2)
end
println(result)
push!(results, result)

result = benchmark("  Quality control (full)", n_runs=3) do
    quality_control(geno; min_maf=0.01, max_missing_per_marker=0.1,
                    max_missing_per_sample=0.1, hwe_pvalue=1e-6, verbose=false)
end
println(result)
push!(results, result)

# ============================================================================
# Benchmark 5: LD Pruning
# ============================================================================

print_section("LD Pruning")

# Smaller dataset for LD pruning
n_samples_ld = 500
n_markers_ld = 2000

geno_data_ld = rand(0:2, n_samples_ld, n_markers_ld)
sample_ids_ld = [string("S", i) for i in 1:n_samples_ld]
marker_ids_ld = [string("M", i) for i in 1:n_markers_ld]
chromosomes = vcat([fill(string(c), div(n_markers_ld, 5)) for c in 1:5]...)
positions = repeat(collect(1:div(n_markers_ld, 5)) .* 10000, 5)
geno_ld = CompactGenotypes(geno_data_ld, sample_ids_ld, marker_ids_ld;
                           chromosome=chromosomes, position=positions)

result = benchmark("  LD r² computation (pairwise)", n_runs=5) do
    for i in 1:100
        compute_ld_r2(geno_ld, rand(1:100), rand(101:200))
    end
end
println(result)
push!(results, result)

result = benchmark("  LD pruning (window-based)", n_runs=3) do
    ld_prune_window(geno_ld; window_size=50, r2_threshold=0.8, verbose=false)
end
println(result)
push!(results, result)

# ============================================================================
# Benchmark 6: BayesR (Short Run)
# ============================================================================

print_section("BayesR Model (Short MCMC)")

# Smaller dataset for BayesR
n_samples_bayes = 200
n_markers_bayes = 500

geno_data_bayes = rand(0:2, n_samples_bayes, n_markers_bayes)
sample_ids_bayes = [string("S", i) for i in 1:n_samples_bayes]
marker_ids_bayes = [string("M", i) for i in 1:n_markers_bayes]
geno_bayes = CompactGenotypes(geno_data_bayes, sample_ids_bayes, marker_ids_bayes)

y_bayes = randn(n_samples_bayes)
pheno_bayes = PhenotypeData(sample_ids_bayes, ["Trait1"], reshape(y_bayes, n_samples_bayes, 1))

result = benchmark("  BayesR fit (1000 iterations)", n_runs=2) do
    model = BayesRModel(n_iter=1000, burn_in=500, thin=5, verbose=false, seed=123)
    fit!(model, geno_bayes, pheno_bayes)
end
println(result)
push!(results, result)

# ============================================================================
# Benchmark 7: File I/O
# ============================================================================

print_section("File I/O Operations")

# Create temporary test file
temp_dir = mktempdir()

# PLINK write
plink_file = joinpath(temp_dir, "benchmark")
result = benchmark("  PLINK write", n_runs=3) do
    write_plink(plink_file, geno; verbose=false)
end
println(result)
push!(results, result)

# PLINK read
result = benchmark("  PLINK read", n_runs=5) do
    read_plink(plink_file; verbose=false)
end
println(result)
push!(results, result)

# VCF write
vcf_file = joinpath(temp_dir, "benchmark.vcf")
result = benchmark("  VCF write", n_runs=3) do
    write_vcf(vcf_file, geno; verbose=false)
end
println(result)
push!(results, result)

# VCF read
result = benchmark("  VCF read", n_runs=3) do
    read_vcf(vcf_file; verbose=false)
end
println(result)
push!(results, result)

# Cleanup
rm(temp_dir; recursive=true)

# ============================================================================
# Summary
# ============================================================================

print_header("Benchmark Summary")

println("\nTop 5 Fastest Operations:")
println("-"^80)
sorted = sort(results, by=r->r.mean_time)
for (i, result) in enumerate(sorted[1:min(5, length(sorted))])
    @printf("%d. ", i)
    println(result)
end

println("\nTop 5 Slowest Operations:")
println("-"^80)
sorted = sort(results, by=r->r.mean_time, rev=true)
for (i, result) in enumerate(sorted[1:min(5, length(sorted))])
    @printf("%d. ", i)
    println(result)
end

println("\n" * "="^80)
printstyled("Benchmark Suite Complete!\n"; color=:green, bold=true)
println("="^80)
