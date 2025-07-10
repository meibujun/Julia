using DynamicEpistasisGBLUP # Use the main module name defined in Project.toml and src/
using Test
using Random # For setting seed in tests
using CUDA # For checking CUDA functionality

# Set a global seed for reproducibility of tests that involve randomness
Random.seed!(12345)

@testset "DynamicEpistasisGBLUP Package Tests" begin

    # Check if CUDA is functional, skip GPU tests if not
    cuda_functional_for_tests = CUDA.functional() && length(CUDA.devices()) > 0
    if !cuda_functional_for_tests
        @warn "CUDA is not functional or no CUDA devices available. Skipping GPU-specific tests."
    end

    # Test individual components
    # These could be separate files included here, e.g., test_types.jl, test_simulation.jl, etc.
    # For now, including the comprehensive_tests.jl directly.

    println("Running Core Functionality Tests...")
    # These would ideally be in a file like test_core.jl
    @testset "Core Components (Types, Basic Utils)" begin
        # Example: Test type constructors if they have logic
        @test begin
            vc = DynamicEpistasisGBLUP.VarianceComponents{Float32}(
                σ²_a=0.3f0, σ²_aa=0.1f0, σ²_e=0.6f0, σ²_p=1.0f0, h²=0.3f0, H²=0.4f0
            )
            vc.σ²_a == 0.3f0
        end

        # Add more basic unit tests for types and simple utilities here.
    end

    # Include the main comprehensive test suite
    # The `comprehensive_tests.jl` file will contain the TestSuite module.
    # We need to make sure it can access `cuda_functional_for_tests`.
    # One way is to pass it, or TestSuite module can check itself.
    # For simplicity, TestSuite might re-check CUDA.functional().

    # If comprehensive_tests.jl defines a module (e.g., TestSuiteModule) with a runtests function:
    # include("comprehensive_tests.jl")
    # TestSuiteModule.run_all_tests(cuda_functional_for_tests) # Pass the flag

    # Or, if comprehensive_tests.jl just contains @testset blocks:
    println("Including Comprehensive Test Suite...")
    include("comprehensive_tests.jl")

end
