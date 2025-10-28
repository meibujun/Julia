# test/data_tests.jl

using Test
using DataFrames
using CSV
using GenomicPrediction

# Create dummy data for testing
const sample_geno_path = "sample_data/genotypes.csv"
const sample_pheno_path = "sample_data/phenotypes.csv"
const sample_cov_path = "sample_data/covariates.csv"
const wrong_pheno_path = "sample_data/wrong_phenotypes.csv"

mkdir("sample_data")
CSV.write(sample_geno_path, DataFrame(ID=1:5, m1=rand(0:2, 5), m2=rand(0:2, 5)))
CSV.write(sample_pheno_path, DataFrame(ID=1:5, trait1=rand(5)))
CSV.write(sample_cov_path, DataFrame(ID=1:5, cov1=rand(5)))
CSV.write(wrong_pheno_path, DataFrame(ID=[1,2,6], trait1=[1.0, 2.0, 3.0]))


@testset "DataProcessing.jl" begin

    @testset "load_csv" begin
        # Test basic loading
        data = load_csv(sample_geno_path, sample_pheno_path)
        @test data isa GenomicData
        @test size(data.genotypes) == (5, 3)
        @test size(data.phenotypes) == (5, 2)
        @test isnothing(data.covariates)

        # Test loading with covariates
        data_with_cov = load_csv(sample_geno_path, sample_pheno_path, cov_path=sample_cov_path)
        @test data_with_cov isa GenomicData
        @test !isnothing(data_with_cov.covariates)
        @test size(data_with_cov.covariates) == (5, 2)

        # Test data alignment
        data_unaligned = load_csv(sample_geno_path, wrong_pheno_path)
        @test size(data_unaligned.genotypes, 1) == 2
        @test size(data_unaligned.phenotypes, 1) == 2
    end

    @testset "calculate_grm" begin
        G = rand(0:2, 10, 5)
        grm = calculate_grm(G, scale=false)
        @test size(grm) == (10, 10)
        @test grm[1,1] > 0

        grm_scaled = calculate_grm(G, scale=true)
        @test size(grm_scaled) == (10, 10)
        @test isapprox(tr(grm_scaled), 10.0)
    end

    @testset "filter_markers" begin
        geno = DataFrame(ID=1:5, m1=[0,0,0,0,0], m2=[0,1,0,1,2], m3=[0,0,1,1,1])
        filtered_geno = filter_markers(geno, maf_threshold=0.1)
        @test "m1" ∉ names(filtered_geno) # MAF = 0
        @test "m2" ∈ names(filtered_geno) # MAF = 0.4
        @test "m3" ∈ names(filtered_geno) # MAF = 0.4
    end

    @testset "impute_mean" begin
        geno = DataFrame(ID=1:4, m1=[0, 1, missing, 2], m2=[1,1,1,1])
        imputed_geno = impute_mean(geno)
        @test !any(ismissing, imputed_geno.m1)
        @test imputed_geno.m1[3] == 1 # mean of (0,1,2) is 1
        @test imputed_geno.m2 == [1,1,1,1] # No change
    end
end

# Cleanup
rm("sample_data", recursive=true)
