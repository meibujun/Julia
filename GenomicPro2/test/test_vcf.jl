using Test
using GenomicPro2
using Random

@testset "VCF I/O Tests" begin
    # Create temporary test directory
    test_dir = mktempdir()

    @testset "Genotype Parsing" begin
        @testset "Diploid genotypes" begin
            # Bi-allelic
            @test parse_genotype("0/0") == 0
            @test parse_genotype("0|0") == 0
            @test parse_genotype("0/1") == 1
            @test parse_genotype("1/0") == 1
            @test parse_genotype("0|1") == 1
            @test parse_genotype("1|0") == 1
            @test parse_genotype("1/1") == 2
            @test parse_genotype("1|1") == 2
        end

        @testset "Missing genotypes" begin
            @test ismissing(parse_genotype("."))
            @test ismissing(parse_genotype("./."))
            @test ismissing(parse_genotype(".|."))
            @test ismissing(parse_genotype("./0"))
            @test ismissing(parse_genotype("0/."))
        end

        @testset "Multi-allelic genotypes" begin
            # Multi-allelic: count non-reference alleles
            @test parse_genotype("1/2") == 2
            @test parse_genotype("2/2") == 2
            @test parse_genotype("0/2") == 1
            @test parse_genotype("2/0") == 1
        end

        @testset "Haploid genotypes" begin
            @test parse_genotype("0") == 0
            @test parse_genotype("1") == 1
            @test parse_genotype("2") == 2
        end
    end

    @testset "VCF Writing and Reading" begin
        # Create test genotype data
        Random.seed!(123)

        n_samples = 50
        n_markers = 100

        geno_data = rand(0:2, n_samples, n_markers)
        sample_ids = [string("Sample", i) for i in 1:n_samples]
        marker_ids = [string("SNP", i) for i in 1:n_markers]

        # Create chromosome and position info
        chromosomes = vcat(fill("1", 50), fill("2", 50))
        positions = vcat(collect(1:50) .* 10000, collect(1:50) .* 10000)
        ref_alleles = fill("A", n_markers)
        alt_alleles = fill("G", n_markers)

        geno = CompactGenotypes(
            geno_data,
            sample_ids,
            marker_ids;
            chromosome = chromosomes,
            position = positions,
            ref_allele = ref_alleles,
            alt_allele = alt_alleles
        )

        @testset "Write and Read VCF" begin
            vcf_file = joinpath(test_dir, "test.vcf")

            # Write VCF
            write_vcf(vcf_file, geno; verbose = false)

            @test isfile(vcf_file)

            # Read VCF
            geno_read = read_vcf(vcf_file; verbose = false)

            @test geno_read.n_samples == n_samples
            @test geno_read.n_markers == n_markers
            @test geno_read.sample_ids == sample_ids
            @test geno_read.marker_ids == marker_ids

            # Check genotypes match
            for i in 1:n_samples
                for j in 1:n_markers
                    @test get_genotype(geno, i, j) == get_genotype(geno_read, i, j)
                end
            end

            # Check metadata
            @test geno_read.chromosome == chromosomes
            @test geno_read.position == positions
            @test geno_read.ref_allele == ref_alleles
            @test geno_read.alt_allele == alt_alleles
        end

        @testset "Write VCF with options" begin
            vcf_file = joinpath(test_dir, "test_options.vcf")

            write_vcf(
                vcf_file,
                geno;
                file_format = "VCFv4.3",
                source = "TestSuite",
                reference = "GRCh38",
                verbose = false
            )

            @test isfile(vcf_file)

            # Check header content
            header_lines = readlines(vcf_file)[1:5]
            @test any(contains.(header_lines, "VCFv4.3"))
            @test any(contains.(header_lines, "TestSuite"))
            @test any(contains.(header_lines, "GRCh38"))
        end
    end

    @testset "VCF Header Parsing" begin
        # Create a minimal VCF file
        vcf_file = joinpath(test_dir, "test_header.vcf")

        open(vcf_file, "w") do io
            println(io, "##fileformat=VCFv4.2")
            println(io, "##contig=<ID=1,length=249250621>")
            println(io, "##contig=<ID=2,length=242193529>")
            println(io, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">")
            println(io, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")
            println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2\tSample3")
            println(io, "1\t10000\trs001\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1")
        end

        io = open(vcf_file, "r")
        header = parse_vcf_header(io)
        close(io)

        @test header.file_format == "VCFv4.2"
        @test header.n_samples == 3
        @test header.sample_ids == ["Sample1", "Sample2", "Sample3"]
        @test haskey(header.contig_info, "1")
        @test haskey(header.contig_info, "2")
        @test header.contig_info["1"] == 249250621
        @test haskey(header.info_fields, "AF")
        @test haskey(header.format_fields, "GT")
    end

    @testset "VCF Reading with Filters" begin
        # Create test VCF with multiple chromosomes and quality scores
        vcf_file = joinpath(test_dir, "test_filter.vcf")

        open(vcf_file, "w") do io
            println(io, "##fileformat=VCFv4.2")
            println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2")
            # Chr1, high quality
            println(io, "1\t1000\tv1\tA\tG\t100\tPASS\t.\tGT\t0/0\t0/1")
            println(io, "1\t2000\tv2\tC\tT\t90\tPASS\t.\tGT\t0/1\t1/1")
            # Chr1, low quality
            println(io, "1\t3000\tv3\tG\tA\t10\tLowQual\t.\tGT\t0/0\t0/0")
            # Chr2, high quality
            println(io, "2\t1000\tv4\tT\tC\t95\tPASS\t.\tGT\t1/1\t0/1")
            println(io, "2\t2000\tv5\tA\tT\t85\tPASS\t.\tGT\t0/1\t0/0")
        end

        @testset "Region filter" begin
            geno = read_vcf(vcf_file; regions = ["1"], verbose = false)
            @test geno.n_markers == 3  # All chr1 variants
            @test all(geno.chromosome .== "1")
        end

        @testset "Quality filter" begin
            geno = read_vcf(vcf_file; min_qual = 50.0, verbose = false)
            @test geno.n_markers == 4  # Variants with QUAL >= 50
        end

        @testset "PASS only filter" begin
            geno = read_vcf(vcf_file; pass_only = true, verbose = false)
            @test geno.n_markers == 4  # Only PASS variants
        end

        @testset "Combined filters" begin
            geno = read_vcf(
                vcf_file;
                regions = ["1"],
                min_qual = 50.0,
                pass_only = true,
                verbose = false
            )
            @test geno.n_markers == 2  # Chr1, QUAL >= 50, PASS
        end

        @testset "Max variants limit" begin
            geno = read_vcf(vcf_file; max_variants = 2, verbose = false)
            @test geno.n_markers == 2
        end
    end

    @testset "VCF Sample Selection" begin
        vcf_file = joinpath(test_dir, "test_samples.vcf")

        open(vcf_file, "w") do io
            println(io, "##fileformat=VCFv4.2")
            println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSampleA\tSampleB\tSampleC\tSampleD")
            println(io, "1\t1000\tv1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/1")
            println(io, "1\t2000\tv2\tC\tT\t.\tPASS\t.\tGT\t0/1\t1/1\t0/0\t0/1")
        end

        @testset "Select specific samples" begin
            geno = read_vcf(vcf_file; samples = ["SampleA", "SampleC"], verbose = false)
            @test geno.n_samples == 2
            @test geno.sample_ids == ["SampleA", "SampleC"]
            @test geno.n_markers == 2
        end

        @testset "Missing sample warning" begin
            # Should warn but still read available samples
            geno = @test_logs (:warn,) match_mode=:any read_vcf(
                vcf_file;
                samples = ["SampleA", "SampleX"],
                verbose = false
            )
            @test geno.n_samples == 1
            @test geno.sample_ids == ["SampleA"]
        end
    end

    @testset "Multi-allelic Handling" begin
        vcf_file = joinpath(test_dir, "test_multiallelic.vcf")

        open(vcf_file, "w") do io
            println(io, "##fileformat=VCFv4.2")
            println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1")
            # Bi-allelic
            println(io, "1\t1000\tv1\tA\tG\t.\tPASS\t.\tGT\t0/0")
            # Multi-allelic
            println(io, "1\t2000\tv2\tC\tT,G\t.\tPASS\t.\tGT\t1/2")
            println(io, "1\t3000\tv3\tG\tA,C,T\t.\tPASS\t.\tGT\t0/1")
        end

        @testset "Skip multi-allelic (default)" begin
            geno = read_vcf(vcf_file; biallelic_only = true, verbose = false)
            @test geno.n_markers == 1  # Only v1
        end

        @testset "Include multi-allelic" begin
            geno = read_vcf(vcf_file; biallelic_only = false, verbose = false)
            @test geno.n_markers == 3  # All variants
        end
    end

    @testset "Missing Data Handling" begin
        vcf_file = joinpath(test_dir, "test_missing.vcf")

        open(vcf_file, "w") do io
            println(io, "##fileformat=VCFv4.2")
            println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3")
            println(io, "1\t1000\tv1\tA\tG\t.\tPASS\t.\tGT\t0/0\t./.\t1/1")
            println(io, "1\t2000\tv2\tC\tT\t.\tPASS\t.\tGT\t0/1\t0/1\t0/1")
            println(io, "1\t3000\tv3\tG\tA\t.\tPASS\t.\tGT\t./.\t./.\t./.")
        end

        geno = read_vcf(vcf_file; verbose = false)

        @test geno.n_markers == 3
        @test geno.n_samples == 3

        # Variant 1: one missing genotype (should be imputed)
        # S2 was ./., should be imputed to median of S1(0) and S3(2) = 1
        @test get_genotype(geno, 1, 1) == 0
        @test get_genotype(geno, 2, 1) in [0, 1, 2]  # Imputed
        @test get_genotype(geno, 3, 1) == 2

        # Variant 2: no missing
        @test get_genotype(geno, 1, 2) == 1
        @test get_genotype(geno, 2, 2) == 1
        @test get_genotype(geno, 3, 2) == 1

        # Variant 3: all missing (should be imputed to 0)
        @test get_genotype(geno, 1, 3) == 0
        @test get_genotype(geno, 2, 3) == 0
        @test get_genotype(geno, 3, 3) == 0
    end

    @testset "VCFHeader Display" begin
        vcf_file = joinpath(test_dir, "test_display.vcf")

        open(vcf_file, "w") do io
            println(io, "##fileformat=VCFv4.2")
            println(io, "##contig=<ID=1,length=249250621>")
            println(io, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">")
            println(io, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")
            println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2")
            println(io, "1\t1000\tv1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1")
        end

        io_vcf = open(vcf_file, "r")
        header = parse_vcf_header(io_vcf)
        close(io_vcf)

        io = IOBuffer()
        show(io, header)
        output = String(take!(io))

        @test contains(output, "VCF Header")
        @test contains(output, "VCFv4.2")
        @test contains(output, "Samples: 2")
        @test contains(output, "Contigs: 1")
    end

    @testset "Round-trip Conversion" begin
        # Create original data
        Random.seed!(456)

        n_samples = 30
        n_markers = 50

        geno_original = CompactGenotypes(
            rand(0:2, n_samples, n_markers),
            [string("ID", i) for i in 1:n_samples],
            [string("Var", i) for i in 1:n_markers];
            chromosome = [string(div(i - 1, 10) + 1) for i in 1:n_markers],
            position = collect(1:n_markers) .* 5000,
            ref_allele = fill("A", n_markers),
            alt_allele = fill("T", n_markers)
        )

        # Write to VCF
        vcf_file = joinpath(test_dir, "test_roundtrip.vcf")
        write_vcf(vcf_file, geno_original; verbose = false)

        # Read back
        geno_read = read_vcf(vcf_file; verbose = false)

        # Compare
        @test geno_read.n_samples == geno_original.n_samples
        @test geno_read.n_markers == geno_original.n_markers
        @test geno_read.sample_ids == geno_original.sample_ids
        @test geno_read.marker_ids == geno_original.marker_ids
        @test geno_read.chromosome == geno_original.chromosome
        @test geno_read.position == geno_original.position

        # Check all genotypes match
        for i in 1:n_samples
            for j in 1:n_markers
                @test get_genotype(geno_read, i, j) == get_genotype(geno_original, i, j)
            end
        end
    end

    @testset "Empty and Edge Cases" begin
        @testset "No genotype data" begin
            vcf_file = joinpath(test_dir, "test_no_gt.vcf")

            open(vcf_file, "w") do io
                println(io, "##fileformat=VCFv4.2")
                println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
                println(io, "1\t1000\tv1\tA\tG\t.\tPASS\t.")
            end

            geno = read_vcf(vcf_file; verbose = false)
            @test geno.n_markers == 0  # No genotype data available
        end

        @testset "No variants pass filters" begin
            vcf_file = joinpath(test_dir, "test_no_pass.vcf")

            open(vcf_file, "w") do io
                println(io, "##fileformat=VCFv4.2")
                println(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1")
                println(io, "1\t1000\tv1\tA\tG\t10\tLowQual\t.\tGT\t0/0")
            end

            geno = read_vcf(vcf_file; min_qual = 50.0, verbose = false)
            @test geno.n_markers == 0
        end
    end

    # Cleanup
    rm(test_dir; recursive = true)
end

println("✓ All VCF I/O tests passed")
