using Test
using GenomicPro2

@testset "I/O Module Tests" begin
    @testset "PhenotypeData" begin
        # Create test phenotype data
        sample_ids = ["S$i" for i in 1:10]
        trait_names = ["Trait1", "Trait2"]
        values = randn(10, 2)

        pheno = PhenotypeData(sample_ids, trait_names, values)

        @test n_samples(pheno) == 10
        @test n_traits(pheno) == 2
        @test sample_ids(pheno) == sample_ids

        # Test indexing
        @test pheno[1, 1] == values[1, 1]
        @test length(pheno[:, 1]) == 10
    end

    @testset "PhenotypeData with covariates" begin
        sample_ids = ["S$i" for i in 1:10]
        trait_names = ["Yield"]
        values = randn(10, 1)
        covariates = Dict("Sex" => rand(["M", "F"], 10),
                         "Location" => rand(["L1", "L2", "L3"], 10))

        pheno = PhenotypeData(sample_ids, trait_names, values;
                             covariates=covariates)

        @test !isempty(pheno.covariates)
        @test haskey(pheno.covariates, "Sex")
        @test length(pheno.covariates["Sex"]) == 10
    end

    @testset "PhenotypeData dimension mismatch" begin
        sample_ids = ["S$i" for i in 1:10]
        trait_names = ["Trait1"]
        values = randn(5, 1)  # Wrong size

        @test_throws DimensionMismatchError PhenotypeData(sample_ids, trait_names, values)
    end

    @testset "Write and read phenotypes (CSV)" begin
        # Create test file
        mktemp() do path, io
            # Write test data
            println(io, "ID,Trait1,Trait2,Sex")
            for i in 1:10
                println(io, "S$i,$(rand()),$(rand()),$(rand(["M", "F"]))")
            end
            close(io)

            # Read back
            pheno = read_phenotypes(path;
                                   id_col="ID",
                                   trait_cols=["Trait1", "Trait2"],
                                   covariate_cols=["Sex"])

            @test n_samples(pheno) == 10
            @test n_traits(pheno) == 2
            @test haskey(pheno.covariates, "Sex")
        end
    end

    @testset "Merge genotype and phenotype" begin
        # Create test data
        geno_data = rand(0:2, 15, 100)
        geno_ids = ["S$i" for i in 1:15]
        marker_ids = ["M$i" for i in 1:100]
        geno = CompactGenotypes(geno_data, geno_ids, marker_ids)

        pheno_ids = ["S$i" for i in 5:20]  # Partial overlap
        pheno_values = randn(16, 1)
        pheno = PhenotypeData(pheno_ids, ["Yield"], pheno_values)

        # Merge
        geno_merged, pheno_merged, common_ids = merge_genotype_phenotype(geno, pheno)

        # Should have samples 5-15 (11 total)
        @test n_samples(geno_merged) == 11
        @test n_samples(pheno_merged) == 11
        @test length(common_ids) == 11
        @test all(id -> id in common_ids, sample_ids(geno_merged))
        @test all(id -> id in common_ids, sample_ids(pheno_merged))
    end

    @testset "Merge with no overlap" begin
        geno_data = rand(0:2, 10, 100)
        geno = CompactGenotypes(geno_data,
                               ["S$i" for i in 1:10],
                               ["M$i" for i in 1:100])

        pheno = PhenotypeData(["S$i" for i in 11:20],
                             ["Yield"],
                             randn(10, 1))

        @test_throws DataValidationError merge_genotype_phenotype(geno, pheno)
    end
end

# Note: PLINK format tests would require actual .bed/.bim/.fam files
# These are integration tests that would be run separately
@testset "PLINK format (structure only)" begin
    @testset "PlinkFiles construction" begin
        files = PlinkFiles("test")
        @test files.bed == "test.bed"
        @test files.bim == "test.bim"
        @test files.fam == "test.fam"
    end
end

println("✓ All I/O tests passed")
