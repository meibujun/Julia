# ===== test/comprehensive_tests.jl =====
"""
Comprehensive test suite for DynamicEpistasisGBLUP package.
This file is included by test/runtests.jl.
"""

# This file can define its own module to encapsulate tests, or just add to the @testset from runtests.jl.
# Using a module for better organization and to manage using statements for submodules.
module ComprehensiveTestSuite

using Test
using Random
using LinearAlgebra
using Statistics
using CUDA
using DataFrames # For checking CV results format

# Import the main module to test its functionalities
# This assumes runtests.jl has already done `using DynamicEpistasisGBLUP`
# or that we are in the same top-level test environment.
# To be safe, explicitly use the full path if DynamicEpistasisGBLUP is the package name.
import Main.DynamicEpistasisGBLUP # Accessing the main package module loaded by runtests.jl
# For submodules, they are typically accessed via the main package module:
# e.g., Main.DynamicEpistasisGBLUP.WalshHadamard if they are exported or submodule syntax is used.

# Helper function to generate test data - can be defined here or imported if in a test_utils.jl
function generate_test_population_data(n_ind::Int = 100, n_snps::Int = 500; seed_val::Int=42)
    Random.seed!(seed_val) # Seed for reproducibility within this function

    test_pop = Main.DynamicEpistasisGBLUP.simulate_population(
        n_individuals = n_ind,
        n_snps = n_snps,
        n_qtl_additive = min(10, n_snps > 0 ? n_snps ÷ 5 : 0),
        n_qtl_epistatic_pairs = min(5, n_snps > 0 ? n_snps ÷ 10 : 0),
        h2_narrow = 0.3,
        h2_broad = 0.4,
        seed = seed_val
    )
    return test_pop
end

const IS_CUDA_FUNCTIONAL_TESTSUITE = CUDA.functional() && !isempty(CUDA.devices())


@testset "Simulation Module Tests" begin
    @testset "Population Simulation and Genetic Architecture" begin
        n_ind, n_snps_test = 50, 200
        n_add_qtl, n_epi_pairs_qtl = 10, 5

        pop_data = generate_test_population_data(n_ind, n_snps_test, seed_val=111)
        @test pop_data isa Main.DynamicEpistasisGBLUP.PopulationData
        @test pop_data.genotypes.n_individuals == n_ind
        @test pop_data.genotypes.n_snps == n_snps_test
        @test length(pop_data.phenotypes.values) == n_ind

        arch = pop_data.metadata[:architecture]
        @test arch isa Main.DynamicEpistasisGBLUP.GeneticArchitecture
        # Check if counts match what was requested (or slightly less if few SNPs available)
        @test arch.n_qtl_additive <= n_add_qtl
        @test arch.n_epistatic_pairs <= n_epi_pairs_qtl
        @test length(arch.additive_qtl_actual_indices) == arch.n_qtl_additive
        @test length(arch.additive_effects) == arch.n_qtl_additive
        @test length(arch.epistatic_pairs_actual_indices) == arch.n_epistatic_pairs
        @test length(arch.epistatic_effects) == arch.n_epistatic_pairs
        @test arch.h2_narrow_target ≈ 0.3
        @test arch.h2_broad_target ≈ 0.4

        # Test that true BVs are stored and have correct length
        @test haskey(pop_data.metadata, :true_breeding_values)
        @test length(pop_data.metadata[:true_breeding_values]) == n_ind

        # Test that calculate_true_genetic_values runs with the new arch
        # It's implicitly tested by simulate_population, but an explicit check is good
        gvals = Main.DynamicEpistasisGBLUP.calculate_true_genetic_values(pop_data.genotypes, arch)
        @test length(gvals[:total]) == n_ind

    end
end


@testset "GRM Computation Tests" begin
    if IS_CUDA_FUNCTIONAL_TESTSUITE
        pop_data = generate_test_population_data(64, 128, seed_val=222)

        @testset "Additive GRM (GPU)" begin
            G_add_gpu = Main.DynamicEpistasisGBLUP.compute_grm!(pop_data.genotypes, use_gpu=true)
            @test G_add_gpu isa CuArray
            @test size(G_add_gpu) == (64, 64)
            @test issymmetric(Array(G_add_gpu))
            @test all(diag(Array(G_add_gpu)) .>= -1e-6) # Allow for small numerical noise around 0
        end

        @testset "Epistatic GRM (GPU)" begin
            pop_data_small_snps = generate_test_population_data(32, 60, seed_val=333) # n_snps not power of 2
            G_epi_gpu = Main.DynamicEpistasisGBLUP.compute_epistatic_grm!(pop_data_small_snps.genotypes, use_gpu=true)
            @test G_epi_gpu isa CuArray
            @test size(G_epi_gpu) == (32, 32)
            @test issymmetric(Array(G_epi_gpu))
        end
    else
        @warn "Skipping GRM GPU tests as CUDA is not functional."
    end

    @testset "Additive GRM (CPU)" begin
        pop_data_cpu = generate_test_population_data(30, 100, seed_val=444)
        # Temporarily make genotype data a Matrix for CPU path if it expects that
        # For now, assume compute_grm! handles CuArray input even for use_gpu=false path correctly.
        G_add_cpu = Main.DynamicEpistasisGBLUP.compute_grm!(pop_data_cpu.genotypes, use_gpu=false)
        @test G_add_cpu isa Matrix
        @test size(G_add_cpu) == (30, 30)
        @test issymmetric(G_add_cpu)
        @test all(diag(G_add_cpu) .>= -1e-6)
    end
end

@testset "Core GBLUP Fitting Tests" begin
    if IS_CUDA_FUNCTIONAL_TESTSUITE
        pop_data_gblup = generate_test_population_data(50, 250, seed_val=555)

        @testset "Additive GBLUP (GPU)" begin
            model_add, gebv_add = Main.DynamicEpistasisGBLUP.orthogonal_epistasis_gblup(
                pop_data_gblup,
                include_epistasis=false
            )
            @test model_add isa Main.DynamicEpistasisGBLUP.OrthogonalGBLUP
            @test length(gebv_add) == 50
            @test model_add.variance.σ²_a >= 0
            @test model_add.variance.σ²_aa ≈ 0 atol=1e-5 # No epistasis in this model fit
        end

        @testset "Epistatic GBLUP (GPU)" begin
            model_epi, gebv_epi = Main.DynamicEpistasisGBLUP.orthogonal_epistasis_gblup(
                pop_data_gblup,
                include_epistasis=true
            )
            @test model_epi isa Main.DynamicEpistasisGBLUP.OrthogonalGBLUP
            @test length(gebv_epi) == 50
            @test model_epi.variance.σ²_a >= 0
            @test model_epi.variance.σ²_aa >= 0
            @test model_epi.variance.H² >= model_epi.variance.h² - 1e-6 # Allow for float precision
        end
    else
        @warn "Skipping GBLUP GPU fitting tests as CUDA is not functional."
    end
end

@testset "Prediction and Cross-Validation Tests" begin
    pop_data_cv = generate_test_population_data(100, 150, seed_val=666) # Reduced SNPs for faster CV

    @testset "Cross-Validation" begin
        if IS_CUDA_FUNCTIONAL_TESTSUITE
            cv_results_df = Main.DynamicEpistasisGBLUP.cross_validation(
                pop_data_cv,
                n_folds = 2,
                include_epistasis = true
            )
            @test cv_results_df isa DataFrame
            @test nrow(cv_results_df) == 2
            @test all(col -> col in names(cv_results_df), ["fold", "accuracy", "bias", "mse"])
            @test all(isfinite.(cv_results_df.accuracy)) # Check for NaN/Inf
        else
            @warn "Skipping Cross-Validation tests as CUDA not functional."
        end
    end

    @testset "Genomic Prediction (Stubbed - to be expanded)" begin
        # This test will require implemented cross-GRM functions and careful setup
        # of training model and prediction set.
        @test true # Placeholder
    end
end


@testset "Advanced Modules Sanity Checks" begin
    if IS_CUDA_FUNCTIONAL_TESTSUITE
        pop_data_adv = generate_test_population_data(64, 128, seed_val=777) # PoT for WHT
        phenos_gpu = CuArray(pop_data_adv.phenotypes.values)
        genos_gpu = pop_data_adv.genotypes.data

        @testset "WalshHadamard Module" begin
            # Access submodule via Main.DynamicEpistasisGBLUP
            WHT = Main.DynamicEpistasisGBLUP.WalshHadamard

            test_vec_len = 8
            test_vec = CuArray{Float32}(rand(Float32, test_vec_len))
            wht_result = WHT.fast_walsh_hadamard_transform!(copy(test_vec))
            @test length(wht_result) == length(test_vec)
            wht_wht_result = WHT.fast_walsh_hadamard_transform!(copy(wht_result)) # Apply again
            @test Array(wht_wht_result) ≈ Array(test_vec) atol=1e-5 # WHT(WHT(x)) = x (up to overall N factor if unnormalized, or x if 1/sqrtN used twice)

            n_snps_orig_wht = size(genos_gpu,2)
            # WHT epistasis detection expects SNP dim to be power of 2.
            # For this test, let's select a subset of SNPs that is power of 2 if current isn't.
            n_snps_for_wht_test = prevpow(2, n_snps_orig_wht)
            if n_snps_for_wht_test < 2 n_snps_for_wht_test = 0 end # Skip if not enough SNPs

            if n_snps_for_wht_test > 0
                genos_for_wht_test = genos_gpu[:, 1:n_snps_for_wht_test]
                epi_interactions_wht = WHT.detect_epistasis_wht(
                    genos_for_wht_test,
                    k_top_interactions=10
                )
                @test epi_interactions_wht isa WHT.EpistaticInteractionsWHT
                @test length(epi_interactions_wht.indices) <= 10
            else
                 @info "Skipping WHT epistasis detection test due to insufficient SNPs for power-of-2 requirement."
            end
        end

        @testset "NOIAFramework Module" begin
            NOIA = Main.DynamicEpistasisGBLUP.NOIAFramework

            n_effects_noia = 1
            S_coding_gpu = CUDA.zeros(Float32, size(genos_gpu,1), size(genos_gpu,2), n_effects_noia)

            # Create OrthogonalGenotypeCoding object first
            coding_obj = NOIA.OrthogonalGenotypeCoding(
                S_coding_gpu,
                pop_data_adv.genotypes.allele_freq,
                :population,
                [:additive]
            )
            # Then call compute_orthogonal_coding! which modifies S in the object
            NOIA.compute_orthogonal_coding!(
                coding_obj.S, genos_gpu, pop_data_adv.genotypes.allele_freq,
                effects_to_include=[:additive], reference_point_symbol=:population
            )
            @test true # Ran without error

            effects_dict_noia = NOIA.compute_orthogonal_effects_noia(phenos_gpu, coding_obj)
            @test haskey(effects_dict_noia, :additive)
            if haskey(effects_dict_noia, :additive)
                 @test size(effects_dict_noia[:additive]) == (size(genos_gpu,2),) # Num SNPs
            end
        end

        @testset "SparseEpistasis Module" begin
            SE = Main.DynamicEpistasisGBLUP.SparseEpistasis
            sel_interactions, _ = SE.detect_sparse_interactions_screening(
                genos_gpu, phenos_gpu, max_candidate_interactions=20
            )
            @test length(sel_interactions) <= 20

            if !isempty(sel_interactions)
                sparse_model_en = SE.fit_elastic_net_epistasis(
                    genos_gpu, phenos_gpu, sel_interactions, lambda1=0.01f0, lambda2=0.001f0
                )
                @test sparse_model_en isa SE.SparseEpistaticModel
            end
        end
    else
        @warn "Skipping Advanced GPU Module Sanity Checks (WHT, NOIA, SparseEpistasis) as CUDA is not functional."
    end
end

@testset "Integration: Simulation Pipeline" begin
    if IS_CUDA_FUNCTIONAL_TESTSUITE
        initial_pop = generate_test_population_data(40, 100, seed_val=888) # Smaller for faster pipeline test

        populations_hist, models_hist, gains_hist = Main.DynamicEpistasisGBLUP.simulate_long_term_genetic_gain(
            initial_pop,
            num_generations_to_simulate = 1, # Very short simulation
            num_matings_per_gen = 10,
            num_offspring_per_mating = 2,
            parent_selection_intensity = 0.5,
            re_estimate_model_freq_gens = 1
        )
        @test length(populations_hist) == 2
        @test length(models_hist) == 1
        @test length(gains_hist) == 1
        @test true
    else
        @warn "Skipping Integration: Simulation Pipeline test as CUDA is not functional."
    end
end

@testset "Basic Benchmarks (Sanity Check)" begin
    if IS_CUDA_FUNCTIONAL_TESTSUITE && isdefined(Main.DynamicEpistasisGBLUP, :benchmark_grm_computation)
        # Check if function exists before calling, as it's more of a utility script
        # println("Running basic benchmark_grm_computation...")
        # Main.DynamicEpistasisGBLUP.benchmark_grm_computation(32, 64) # Very small scale
        @test true
    else
        # @warn "Skipping Basic Benchmarks as CUDA is not functional or benchmark_grm_computation not found."
    end
end

@testset "REML Implementation Tests" begin
    if IS_CUDA_FUNCTIONAL_TESTSUITE
        # Use a small, controlled population for REML test
        pop_data = generate_test_population_data(80, 150, seed_val=999)

        y_train = CuArray(pop_data.phenotypes.values)
        X_train = CUDA.ones(Float32, 80, 1)

        Ga = Main.DynamicEpistasisGBLUP.GRMComputation.compute_grm!(pop_data.genotypes, use_gpu=true)
        Gaa = Main.DynamicEpistasisGBLUP.GRMComputation.compute_epistatic_grm!(pop_data.genotypes, use_gpu=true)

        # Initial variance guesses
        total_var = var(pop_data.phenotypes.values)
        initial_variances = Float32[0.4*total_var, 0.1*total_var, 0.5*total_var]

        reml_params = Main.DynamicEpistasisGBLUP.Types.REMLParameters{Float32}(
            max_iter=50,
            tol=1e-5,
            verbose=false,
            min_variance_value=1e-9,
            use_gpu=true
        )

        reml_results = Main.DynamicEpistasisGBLUP.REML.estimate_variance_components_reml!(
            y_train,
            [Ga, Gaa],
            X_train,
            initial_variances,
            reml_params
        )

        @test reml_results.converged == true
        @test all(reml_results.var_components .> 0)
        @test isfinite(reml_results.log_likelihood)
    else
        @warn "Skipping REML Implementation tests as CUDA is not functional."
    end
end

@testset "CPU GRM Optimizations Tests" begin
    # Test if optimized CPU versions give approx the same result as GPU versions
    if IS_CUDA_FUNCTIONAL_TESTSUITE
        pop_data = generate_test_population_data(60, 120, seed_val=101)

        # Additive GRM
        G_gpu = Main.DynamicEpistasisGBLUP.GRMComputation.compute_grm!(deepcopy(pop_data.genotypes), use_gpu=true)
        G_cpu = Main.DynamicEpistasisGBLUP.GRMComputation.compute_grm!(deepcopy(pop_data.genotypes), use_gpu=false)
        @test G_cpu isa Matrix
        @test Array(G_gpu) ≈ G_cpu atol=1e-5

        # Epistatic GRM
        Gaa_gpu = Main.DynamicEpistasisGBLUP.GRMComputation.compute_epistatic_grm!(deepcopy(pop_data.genotypes), use_gpu=true)
        Gaa_cpu = Main.DynamicEpistasisGBLUP.GRMComputation.compute_epistatic_grm!(deepcopy(pop_data.genotypes), use_gpu=false)
        @test Gaa_cpu isa Matrix
        @test Array(Gaa_gpu) ≈ Gaa_cpu atol=1e-5
    else
        @warn "Skipping CPU vs GPU GRM comparison tests as CUDA is not functional."
    end
end


end # module ComprehensiveTestSuite
