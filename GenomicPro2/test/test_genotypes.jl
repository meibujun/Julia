"""
Tests for genotype data structures.
"""

using GenomicPro2
using GenomicPro2.Data: encode_genotypes, decode_genotypes, to_matrix, memory_usage
using Test
using Statistics

function test_genotypes()
    @testset "Encoding/Decoding" begin
        # Small test matrix
        data = [
            0 1 2 0
            1 2 0 1
            2 0 1 2
        ]

        encoded, missing_mask = encode_genotypes(data)

        @test length(encoded) == cld(3 * 4, 4)  # 3 samples × 4 markers / 4 per byte
        @test size(missing_mask) == (3, 4)
        @test !any(missing_mask)  # No missing values
    end

    @testset "Encoding with missing" begin
        data = [
            0 1 missing
            1 2 0
        ]

        encoded, missing_mask = encode_genotypes(data)

        @test missing_mask[1, 3] == true
        @test !missing_mask[1, 1]
        @test !missing_mask[2, 1]
    end

    @testset "CompactGenotypes construction" begin
        n_samples = 100
        n_markers = 1000

        # Generate random genotypes
        data = rand(0:2, n_samples, n_markers)
        sample_ids = ["sample_$i" for i in 1:n_samples]
        marker_ids = ["marker_$j" for j in 1:n_markers]

        geno = CompactGenotypes(data, sample_ids, marker_ids)

        @test n_samples(geno) == n_samples
        @test n_markers(geno) == n_markers
        @test length(sample_ids(geno)) == n_samples
        @test length(marker_ids(geno)) == n_markers
        @test size(geno) == (n_samples, n_markers)
    end

    @testset "Genotype access" begin
        data = [
            0 1 2
            1 2 0
            2 0 1
        ]

        sample_ids = ["s1", "s2", "s3"]
        marker_ids = ["m1", "m2", "m3"]

        geno = CompactGenotypes(data, sample_ids, marker_ids)

        # Test individual access
        @test geno[1, 1] == 0
        @test geno[1, 2] == 1
        @test geno[1, 3] == 2
        @test geno[2, 1] == 1
        @test geno[3, 3] == 1

        # Test with missing
        data_missing = [
            0 missing 2
            1 2 0
        ]

        geno_missing = CompactGenotypes(
            data_missing,
            ["s1", "s2"],
            ["m1", "m2", "m3"]
        )

        @test geno_missing[1, 1] == 0
        @test ismissing(geno_missing[1, 2])
        @test geno_missing[2, 2] == 2
    end

    @testset "Allele frequencies" begin
        # Known frequencies
        data = [
            0 0 1  # Freq = 0/6, 0/6, 2/6
            0 0 1
            0 0 2
        ]

        geno = CompactGenotypes(
            data,
            ["s1", "s2", "s3"],
            ["m1", "m2", "m3"]
        )

        freqs = allele_frequencies(geno)

        @test freqs[1] ≈ 0.0
        @test freqs[2] ≈ 0.0
        @test freqs[3] ≈ 4/6  # (1+1+2)/(2*3) = 4/6

        # With missing data
        data_missing = [
            0 missing
            1 2
            2 0
        ]

        geno_missing = CompactGenotypes(
            data_missing,
            ["s1", "s2", "s3"],
            ["m1", "m2"]
        )

        freqs_missing = allele_frequencies(geno_missing)

        # m1: (0+1+2)/(2*3) = 3/6 = 0.5
        @test freqs_missing[1] ≈ 0.5

        # m2: (2+0)/(2*2) = 2/4 = 0.5 (only 2 valid)
        @test freqs_missing[2] ≈ 0.5
    end

    @testset "Missing rate" begin
        data = [
            0 1 missing
            1 missing 0
            2 0 1
        ]

        geno = CompactGenotypes(
            data,
            ["s1", "s2", "s3"],
            ["m1", "m2", "m3"]
        )

        # Overall missing rate: 2/9
        @test missing_rate(geno; dim=0) ≈ 2/9

        # Per sample: [1/3, 1/3, 0/3]
        per_sample = missing_rate(geno; dim=1)
        @test length(per_sample) == 3
        @test per_sample[1] ≈ 1/3
        @test per_sample[2] ≈ 1/3
        @test per_sample[3] ≈ 0.0

        # Per marker: [0/3, 1/3, 1/3]
        per_marker = missing_rate(geno; dim=2)
        @test length(per_marker) == 3
        @test per_marker[1] ≈ 0.0
        @test per_marker[2] ≈ 1/3
        @test per_marker[3] ≈ 1/3
    end

    @testset "Validation" begin
        # Valid genotypes
        data = rand(0:2, 50, 100)
        geno = CompactGenotypes(
            data,
            ["s$i" for i in 1:50],
            ["m$j" for j in 1:100]
        )

        result = validate(geno)
        @test result.valid

        # Duplicate sample IDs (should fail)
        @test_throws Exception CompactGenotypes(
            data,
            ["s1", "s1", ["s$i" for i in 3:50]...],  # Duplicate!
            ["m$j" for j in 1:100]
        )
    end

    @testset "to_matrix" begin
        data = [
            0 1 2
            1 2 0
        ]

        geno = CompactGenotypes(
            data,
            ["s1", "s2"],
            ["m1", "m2", "m3"]
        )

        # Without imputation
        mat = to_matrix(geno)
        @test mat == Float64.(data)

        # With missing and imputation
        data_missing = [
            0 missing 2
            1 2 0
        ]

        geno_missing = CompactGenotypes(
            data_missing,
            ["s1", "s2"],
            ["m1", "m2", "m3"]
        )

        mat_imputed = to_matrix(geno_missing; impute=true)

        @test mat_imputed[1, 1] == 0.0
        @test !ismissing(mat_imputed[1, 2])  # Imputed
        @test mat_imputed[1, 3] == 2.0
    end

    @testset "Memory usage" begin
        data = rand(0:2, 1000, 5000)
        geno = CompactGenotypes(
            data,
            ["s$i" for i in 1:1000],
            ["m$j" for j in 1:5000]
        )

        mem = memory_usage(geno)

        @test mem.total > 0
        @test mem.data > 0
        @test mem.missing_mask > 0
        @test mem.savings > 0.9  # Should save > 90%

        # Verify calculation
        expected_naive = 1000 * 5000 * 8  # Float64
        @test mem.naive == expected_naive
        @test mem.savings ≈ 1 - mem.total / expected_naive
    end

    @testset "Round-trip encoding" begin
        # Encode and decode should give same result
        data = rand(0:2, 50, 100)

        sample_ids = ["s$i" for i in 1:50]
        marker_ids = ["m$j" for j in 1:100]

        geno = CompactGenotypes(data, sample_ids, marker_ids)
        decoded = decode_genotypes(geno)

        @test decoded == data
    end

    @testset "Large dataset" begin
        # Test with realistic size
        n_samples = 5000
        n_markers = 50000

        data = rand(0:2, n_samples, n_markers)
        sample_ids = ["sample_$i" for i in 1:n_samples]
        marker_ids = ["marker_$j" for j in 1:n_markers]

        # Should not throw
        @test_nowarn geno = CompactGenotypes(data, sample_ids, marker_ids)

        geno = CompactGenotypes(data, sample_ids, marker_ids)

        # Memory savings should be significant
        mem = memory_usage(geno)
        @test mem.savings > 0.95

        # Can access data
        @test geno[1, 1] in [0, 1, 2]
        @test length(allele_frequencies(geno)) == n_markers
    end
end
