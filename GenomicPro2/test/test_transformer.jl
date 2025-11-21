using Test
using GenomicPro2
using GenomicPro2.Models
using GenomicPro2.Data
using GenomicPro2.Core
using Flux
using Statistics
using Random

@testset "Transformer Genomic Model" begin
    
    # Create dummy data
    n_samples = 50
    n_snps = 100
    
    # Random genotypes (0, 1, 2)
    X = rand(0:2, n_samples, n_snps)
    
    # Create CompactGenotypes
    sample_ids_vec = ["S$i" for i in 1:n_samples]
    marker_ids_vec = ["M$j" for j in 1:n_snps]
    geno = CompactGenotypes(X, sample_ids_vec, marker_ids_vec)
    
    # Random phenotypes
    y = randn(n_samples)
    pheno = PhenotypeData(y, sample_ids_vec, ["Trait1"])
    
    @testset "Initialization" begin
        model = TransformerGenomicModel(
            n_snps = n_snps,
            d_model = 16,
            n_heads = 2,
            n_layers = 1
        )
        
        @test model isa TransformerGenomicModel
        @test length(model.blocks) == 1
    end
    
    @testset "Forward Pass" begin
        model = TransformerGenomicModel(
            n_snps = n_snps,
            d_model = 16,
            n_heads = 2,
            n_layers = 1
        )
        
        # Manually prepare input for forward pass test
        # (1, seq_len, batch_size)
        input = rand(Float32, 1, n_snps, 5)
        output = model(input)
        
        @test size(output) == (1, 5)
    end
    
    @testset "Training" begin
        model = TransformerGenomicModel(
            n_snps = n_snps,
            d_model = 16,
            n_heads = 2,
            n_layers = 1
        )
        
        # Train for a few epochs
        trained_model = train_transformer!(
            model, 
            geno, 
            pheno, 
            epochs=2, 
            batch_size=10, 
            verbose=false
        )
        
        @test trained_model isa TransformerGenomicModel
        
        # Predict
        preds = predict_transformer(trained_model, geno)
        @test length(preds) == n_samples
        @test eltype(preds) == Float32
    end
end
