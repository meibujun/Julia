using GenomicPro3
using Test
using Flux
using CUDA

@testset "AI Module" begin
    
    n_samples = 50
    n_snps = 2000 # Enough for 2 chunks of 1000
    
    # Mock Genotypes
    data = rand(UInt8, ceil(Int, n_samples/4), n_snps)
    geno = CompactGenotypes(data, n_samples, n_snps, ["S$i" for i in 1:n_samples], ["SNP$j" for j in 1:n_snps])
    
    # Mock Phenotypes
    y = randn(Float32, n_samples)
    
    @testset "Transformer Initialization" begin
        model = GenomicTransformer(n_snps, d_model=16, n_heads=2, n_layers=1)
        @test model isa GenomicTransformer
    end
    
    @testset "Transformer Training" begin
        model = GenomicTransformer(n_snps, d_model=16, n_heads=2, n_layers=1)
        
        # Run 1 epoch
        # Note: This might fail if GPU is not present and code forces GPU.
        # The current implementation of train_transformer! uses CuArray explicitly.
        # We should check for GPU.
        
        if GenomicPro3.HPC.has_gpu()
            # This will error if no GPU because train_transformer! uses CuArray
            # We should probably make train_transformer! device agnostic or check inside.
            # For now, we test only if GPU exists.
            train_transformer!(model, geno, y, epochs=1, batch_size=10)
            @test true # If we got here, it didn't crash
        else
            @info "Skipping Transformer Training test (No GPU)"
        end
    end

end
