using GenomicPro3
using Test
using DataFrames
using Statistics
using CUDA

@testset "Stats Module" begin
    
    # Create synthetic data
    n_samples = 100
    n_snps = 100
    
    # Genotypes: Random 0, 1, 2
    # We need to pack them manually or use a helper (which we don't have yet for writing)
    # So let's mock the internal data structure directly.
    # n_packed = ceil(100/4) = 25
    data = rand(UInt8, 25, n_snps)
    
    geno = CompactGenotypes(data, n_samples, n_snps, ["S$i" for i in 1:n_samples], ["SNP$j" for j in 1:n_snps])
    
    # Phenotypes: y = SNP1 * 0.5 + noise
    # We need to extract SNP1 to create y
    snp1 = get_snp(geno, 1)
    # Handle missing (NaN) -> 0
    snp1[isnan.(snp1)] .= 0.0
    
    y = snp1 .* 0.5 .+ randn(n_samples) .* 0.1
    
    df = DataFrame(ID = ["S$i" for i in 1:n_samples], Y = y)
    pheno = PhenotypeData(df, ["Y"], String[], "ID")
    
    @testset "GWAS CPU" begin
        res = perform_gwas(geno, pheno, trait="Y", method=:linear) # Should default to CPU if forced or auto
        # But perform_gwas auto-dispatches. Let's call cpu directly to be sure for this test
        res_cpu = GenomicPro3.Stats.GWAS.gwas_linear_cpu(geno, pheno, "Y")
        
        @test nrow(res_cpu) == n_snps
        # SNP1 should be significant
        row1 = res_cpu[res_cpu.SNP .== "SNP1", :]
        @test row1.P[1] < 1e-5
        @test isapprox(row1.Beta[1], 0.5, atol=0.1)
    end
    
    if GenomicPro3.HPC.has_gpu()
        @testset "GWAS GPU" begin
            res_gpu = GenomicPro3.Stats.GWAS.gwas_linear_gpu(geno, pheno, "Y")
            
            @test nrow(res_gpu) == n_snps
            row1 = res_gpu[res_gpu.SNP .== "SNP1", :]
            @test row1.P[1] < 1e-5
            @test isapprox(row1.Beta[1], 0.5, atol=0.1)
            
            # Compare CPU and GPU
            res_cpu = GenomicPro3.Stats.GWAS.gwas_linear_cpu(geno, pheno, "Y")
            @test isapprox(res_gpu.Beta, res_cpu.Beta, atol=1e-4, nans=true)
        end
    else
        @info "Skipping GWAS GPU tests (No GPU detected)"
    end

end
