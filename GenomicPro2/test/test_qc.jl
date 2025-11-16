using Test
using GenomicPro2
using Statistics

@testset "QC Module Tests" begin
    # Create test data with known properties
    n_samples = 100
    n_markers = 500

    # Generate genotype data
    geno_data = rand(0:2, n_samples, n_markers)

    # Add some missing values (5%)
    for _ in 1:round(Int, 0.05 * n_samples * n_markers)
        i, j = rand(1:n_samples), rand(1:n_markers)
        geno_data[i, j] = missing
    end

    sample_ids = ["S$i" for i in 1:n_samples]
    marker_ids = ["M$i" for i in 1:n_markers]
    geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

    @testset "Hardy-Weinberg Test" begin
        # Test exact HWE test
        @testset "Perfect HWE" begin
            # Under HWE with p=0.5: expect 25% AA, 50% Aa, 25% aa
            pval = hardy_weinberg_test(25, 50, 25)
            @test pval > 0.05  # Should not reject HWE
        end

        @testset "Deviation from HWE" begin
            # Excess homozygosity
            pval = hardy_weinberg_test(45, 10, 45)
            @test pval < 0.05  # Should reject HWE
        end

        @testset "No heterozygotes" begin
            pval = hardy_weinberg_test(50, 0, 50)
            @test pval < 0.01
        end

        @testset "Edge cases" begin
            # No data
            pval = hardy_weinberg_test(0, 0, 0)
            @test pval == 1.0

            # Single genotype
            pval = hardy_weinberg_test(100, 0, 0)
            @test pval >= 0.0 && pval <= 1.0
        end
    end

    @testset "Call Rate" begin
        # Overall call rate
        cr_overall = call_rate(geno)
        @test 0 <= cr_overall <= 1
        @test cr_overall ≈ 1 - missing_rate(geno)

        # Per sample
        cr_sample = call_rate(geno; dim=1)
        @test length(cr_sample) == n_samples
        @test all(0 .<= cr_sample .<= 1)

        # Per marker
        cr_marker = call_rate(geno; dim=2)
        @test length(cr_marker) == n_markers
        @test all(0 .<= cr_marker .<= 1)
    end

    @testset "Heterozygosity Rate" begin
        # Overall heterozygosity
        het_overall = heterozygosity_rate(geno)
        @test 0 <= het_overall <= 1

        # Per sample
        het_sample = heterozygosity_rate(geno; dim=1)
        @test length(het_sample) == n_samples
        @test all(0 .<= het_sample .<= 1)

        # Per marker
        het_marker = heterozygosity_rate(geno; dim=2)
        @test length(het_marker) == n_markers
        @test all(0 .<= het_marker .<= 1)
    end

    @testset "Expected Heterozygosity" begin
        exp_het = expected_heterozygosity(geno)

        @test length(exp_het) == n_markers
        @test all(0 .<= exp_het .<= 0.5)  # Max at p=0.5

        # Check calculation for known frequency
        # For p=0.5, expect 2*0.5*0.5 = 0.5
        freqs = allele_frequencies(geno)
        for i in 1:n_markers
            expected = 2 * freqs[i] * (1 - freqs[i])
            @test exp_het[i] ≈ expected
        end
    end

    @testset "Inbreeding Coefficient" begin
        F = inbreeding_coefficient(geno)

        @test length(F) == n_samples
        # F can be negative (excess heterozygosity) or positive (excess homozygosity)
        @test all(-1 .<= F .<= 1)
    end

    @testset "MAF Filter" begin
        keep_markers = filter_maf(geno, 0.05)

        @test length(keep_markers) <= n_markers
        @test all(1 .<= keep_markers .<= n_markers)

        # Verify all kept markers pass threshold
        maf = minor_allele_frequency(geno)
        @test all(maf[keep_markers] .>= 0.05)
    end

    @testset "Missing Rate Filters" begin
        @testset "Filter markers" begin
            keep_markers = filter_missing_markers(geno, 0.2)

            @test length(keep_markers) <= n_markers
            @test all(1 .<= keep_markers .<= n_markers)

            # Verify
            marker_missing = missing_rate(geno; dim=2)
            @test all(marker_missing[keep_markers] .<= 0.2)
        end

        @testset "Filter samples" begin
            keep_samples = filter_missing_samples(geno, 0.2)

            @test length(keep_samples) <= n_samples
            @test all(1 .<= keep_samples .<= n_samples)

            # Verify
            sample_missing = missing_rate(geno; dim=1)
            @test all(sample_missing[keep_samples] .<= 0.2)
        end
    end

    @testset "HWE Filter" begin
        keep_markers = filter_hwe(geno, 1e-6)

        @test length(keep_markers) <= n_markers
        @test all(1 .<= keep_markers .<= n_markers)
        # Most random data should pass HWE
        @test length(keep_markers) > 0.8 * n_markers
    end

    @testset "Quality Control Pipeline" begin
        # Standard QC
        geno_qc = quality_control(geno;
            min_maf = 0.01,
            max_missing_per_marker = 0.1,
            max_missing_per_sample = 0.1,
            apply_hwe = true,
            verbose = false
        )

        @test n_samples(geno_qc) <= n_samples(geno)
        @test n_markers(geno_qc) <= n_markers(geno)
        @test n_samples(geno_qc) >= 10
        @test n_markers(geno_qc) >= 100

        # Verify filters were applied
        @test missing_rate(geno_qc) <= missing_rate(geno)

        maf_qc = minor_allele_frequency(geno_qc)
        @test all(maf_qc .>= 0.01)
    end

    @testset "Strict QC" begin
        geno_strict = quality_control(geno;
            min_maf = 0.05,
            max_missing_per_marker = 0.02,
            max_missing_per_sample = 0.05,
            apply_hwe = false,
            verbose = false
        )

        @test n_samples(geno_strict) <= n_samples(geno)
        @test n_markers(geno_strict) <= n_markers(geno)
    end

    @testset "Duplicate Detection" begin
        # Create data with a duplicate
        n_dup = 50
        m_dup = 200

        data1 = rand(0:2, n_dup, m_dup)
        sample_ids_dup = ["S$i" for i in 1:n_dup]
        marker_ids_dup = ["M$i" for i in 1:m_dup]

        geno_dup = CompactGenotypes(data1, sample_ids_dup, marker_ids_dup)

        # No duplicates in random data
        dups = identify_duplicates(geno_dup; threshold=0.99)
        @test length(dups) == 0

        # Test correlation computation
        cor_mat = compute_sample_correlation(geno_dup)
        @test size(cor_mat) == (n_dup, n_dup)
        @test issymmetric(cor_mat)
        @test all(diag(cor_mat) .≈ 1.0)
    end

    @testset "QC Report" begin
        report = qc_report(geno; check_hwe=false, check_duplicates=false)

        @test report isa QCReport
        @test report.n_samples == n_samples
        @test report.n_markers == n_markers
        @test 0 <= report.overall_missing_rate <= 1

        # Check all stats are present
        @test haskey(report.sample_missing_stats, :mean)
        @test haskey(report.marker_missing_stats, :mean)
        @test haskey(report.maf_stats, :mean)
        @test haskey(report.heterozygosity_stats, :overall)

        # Test display
        io = IOBuffer()
        show(io, report)
        output = String(take!(io))
        @test contains(output, "Quality Control Report")
        @test contains(output, "Dataset Overview")
    end

    @testset "QC Report with HWE" begin
        # Small dataset for faster HWE testing
        small_data = rand(0:2, 20, 50)
        small_geno = CompactGenotypes(small_data,
                                      ["S$i" for i in 1:20],
                                      ["M$i" for i in 1:50])

        report = qc_report(small_geno; check_hwe=true, check_duplicates=true)

        @test report.hwe_stats !== nothing
        @test haskey(report.hwe_stats, :mean_pvalue)
        @test haskey(report.hwe_stats, :n_fail_1e6)
    end

    @testset "QCFilters struct" begin
        # Default construction
        filters = QCFilters()
        @test filters.min_maf == 0.01
        @test filters.max_missing_per_marker == 0.1

        # Custom construction
        filters_custom = QCFilters(
            min_maf = 0.05,
            max_missing_per_marker = 0.02
        )
        @test filters_custom.min_maf == 0.05
        @test filters_custom.max_missing_per_marker == 0.02

        # Invalid parameters
        @test_throws ArgumentError QCFilters(min_maf = -0.1)
        @test_throws ArgumentError QCFilters(min_maf = 0.6)
        @test_throws ArgumentError QCFilters(max_missing_per_marker = 1.5)
    end
end

println("✓ All QC tests passed")
