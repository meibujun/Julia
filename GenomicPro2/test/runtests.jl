"""
GenomicPro2 Comprehensive Test Suite

Runs all tests for GenomicPro2 with detailed reporting and timing information.

Usage:
    julia --project=. test/runtests.jl
    julia --threads=4 --project=. test/runtests.jl
"""

using Test
using GenomicPro2
using LinearAlgebra
using Statistics
using Printf

# Enable colored output
const USE_COLOR = get(ENV, "TERM", "") != "dumb"

function print_colored(io::IO, text::String, color::Symbol)
    if USE_COLOR
        printstyled(io, text; color=color, bold=true)
    else
        print(io, text)
    end
end

function print_header(text::String)
    println("\n" * "="^80)
    print_colored(stdout, text * "\n", :cyan)
    println("="^80)
end

function print_test_summary(name::String, time::Float64, passed::Bool)
    status = passed ? "✓ PASS" : "✗ FAIL"
    color = passed ? :green : :red
    @printf("  %-40s ", name)
    print_colored(stdout, status, color)
    @printf(" (%.3fs)\n", time)
end

# Track test results
test_results = Dict{String,NamedTuple}()

print_header("GenomicPro2 Test Suite")
println("Julia Version: $(VERSION)")
println("Threads: $(Threads.nthreads())")
println("Project: $(dirname(dirname(@__FILE__)))")
println()

overall_start = time()

@testset verbose=true "GenomicPro2 Complete Test Suite" begin

    # Core tests
    @testset "Core Functionality" begin
        start_time = time()
        try
            include("test_core.jl")
            test_results["Core"] = (passed=true, time=time()-start_time)
        catch e
            test_results["Core"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # Data structures tests
    @testset "Data Structures" begin
        start_time = time()
        try
            include("test_genotypes.jl")
            test_results["Genotypes"] = (passed=true, time=time()-start_time)
        catch e
            test_results["Genotypes"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # I/O tests
    @testset "File I/O" begin
        start_time = time()
        try
            include("test_io.jl")
            test_results["I/O"] = (passed=true, time=time()-start_time)
        catch e
            test_results["I/O"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # VCF tests
    @testset "VCF Format" begin
        start_time = time()
        try
            include("test_vcf.jl")
            test_results["VCF"] = (passed=true, time=time()-start_time)
        catch e
            test_results["VCF"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # Model tests
    @testset "Statistical Models" begin
        start_time = time()
        try
            include("test_models.jl")
            test_results["Models"] = (passed=true, time=time()-start_time)
        catch e
            test_results["Models"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # Cross-validation tests
    @testset "Cross-Validation" begin
        start_time = time()
        try
            include("test_crossvalidation.jl")
            test_results["CrossValidation"] = (passed=true, time=time()-start_time)
        catch e
            test_results["CrossValidation"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # BayesR tests
    @testset "BayesR Model" begin
        start_time = time()
        try
            include("test_bayesr.jl")
            test_results["BayesR"] = (passed=true, time=time()-start_time)
        catch e
            test_results["BayesR"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # BayesCπ tests
    @testset "BayesCπ Model" begin
        start_time = time()
        try
            include("test_bayescpi.jl")
            test_results["BayesCπ"] = (passed=true, time=time()-start_time)
        catch e
            test_results["BayesCπ"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # RKHS tests
    @testset "RKHS Model" begin
        start_time = time()
        try
            include("test_rkhs.jl")
            test_results["RKHS"] = (passed=true, time=time()-start_time)
        catch e
            test_results["RKHS"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # Quality control tests
    @testset "Quality Control" begin
        start_time = time()
        try
            include("test_qc.jl")
            test_results["QC"] = (passed=true, time=time()-start_time)
        catch e
            test_results["QC"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end

    # LD Pruning tests
    @testset "LD Pruning" begin
        start_time = time()
        try
            include("test_ld_pruning.jl")
            test_results["LD Pruning"] = (passed=true, time=time()-start_time)
        catch e
            test_results["LD Pruning"] = (passed=false, time=time()-start_time, error=e)
            rethrow(e)
        end
    end
end

overall_time = time() - overall_start

# Print summary
print_header("Test Summary")

total_tests = length(test_results)
passed_tests = count(r -> r.passed, values(test_results))
failed_tests = total_tests - passed_tests

println("\nTest Results:")
println("-"^80)
@printf("%-40s %10s %10s\n", "Module", "Status", "Time (s)")
println("-"^80)

for (name, result) in sort(collect(test_results), by=x->x[1])
    print_test_summary(name, result.time, result.passed)
end

println("-"^80)
@printf("%-40s %10s %10.3f\n", "TOTAL", "$passed_tests/$total_tests", overall_time)
println("-"^80)

# Success rate
success_rate = total_tests > 0 ? 100.0 * passed_tests / total_tests : 0.0
@printf("\nSuccess Rate: %.1f%%\n", success_rate)

if failed_tests == 0
    print_colored(stdout, "\n✓ ALL TESTS PASSED!\n", :green)
else
    print_colored(stdout, "\n✗ $failed_tests TEST(S) FAILED\n", :red)
end

println("\n" * "="^80)
println("Total test time: $(round(overall_time, digits=2))s")
println("="^80)
