# examples/08_multi_omics_integration.jl

using GenomicPro
using DataFrames
using Random

"""
# Example 08: Multi-Omics Integration for Complex Trait Prediction

This example demonstrates how to use the multi-omics capabilities of `GenomicPro.jl`
to integrate genomic (SNP) and transcriptomic (gene expression) data for improved
prediction of complex traits.

## Workflow Overview
1.  **Data Simulation**: Generate synthetic SNP, gene expression, and phenotype data.
2.  **Data Loading**: Load the simulated data into `GenomicPro.jl` data structures.
3.  **Model Definition**: Define a multi-omics model with modality-specific encoders
    (CNN for SNPs, VAE for gene expression) and a cross-attention fusion layer.
4.  **Model Training**: Train the model on the integrated multi-omics data.
5.  **Result Interpretation**: (Placeholder for) interpreting the trained model.
"""

# --- 1. Data Simulation ---
println("Step 1: Simulating multi-omics data...")

const N_SAMPLES = 500
const N_SNPS = 1000
const N_GENES = 100

# Simulate SNP data
snp_data = rand([0, 1, 2], N_SAMPLES, N_SNPS)

# Simulate gene expression data
expression_data = randn(N_SAMPLES, N_GENES)

# Simulate phenotype data (with some correlation to both omics)
phenotypes = vec(sum(snp_data[:, 1:10], dims=2) * 0.1 + sum(expression_data[:, 1:5], dims=2) * 0.2 + randn(N_SAMPLES))

# Create DataFrames
snp_df = DataFrame(snp_data, :auto)
snp_df[!, :ID] = ["ID_$(i)" for i in 1:N_SAMPLES]

expr_df = DataFrame(expression_data, :auto)
expr_df[!, :ID] = ["ID_$(i)" for i in 1:N_SAMPLES]
expr_df[!, :GeneID] = ["Gene_$(i)" for i in 1:N_GENES]


pheno_df = DataFrame(ID = snp_df.ID, Phenotype = phenotypes)

# --- 2. Data Loading ---
println("\nStep 2: Loading data into GenomicPro structures...")

# Create Genotype and ExpressionData objects
genotypes = TwoBitGenotypes(Matrix(snp_df[!, Not(:ID)]))
expression = ExpressionData(expr_df, :ID, :GeneID)
phenotype_data = PhenotypeData(pheno_df, :ID, [:Phenotype])

# Create a MultiOmicsData container
multi_omics_data = MultiOmicsData(Dict(
    :genotypes => genotypes,
    :expression => expression
))

println("Multi-omics data container created.")

# --- 3. Model Definition ---
println("\nStep 3: Defining the multi-omics model...")

# Define the model architecture
latent_dim = 64
n_attention_heads = 4

snp_encoder = build_snp_encoder(N_SNPS, latent_dim)
rnaseq_encoder = build_rnaseq_vae(N_GENES, latent_dim)
fusion_layer = CrossAttentionFusion(latent_dim, n_heads=n_attention_heads)
predictor = Chain(
    Dense(latent_dim * 2, 128, relu),
    Dense(128, 1)
)

multi_omics_model = (
    snp_encoder = snp_encoder,
    expression_encoder = rnaseq_encoder,
    fusion_layer = fusion_layer,
    predictor = predictor
)

println("Multi-omics model defined.")

# --- 4. Model Training ---
println("\nStep 4: Training the multi-omics model...")

# Split data into training and validation sets
train_frac = 0.8
n_train = Int(floor(train_frac * N_SAMPLES))
train_indices = 1:n_train
val_indices = (n_train + 1):N_SAMPLES

y_train = pheno_df.Phenotype[train_indices]
y_val = pheno_df.Phenotype[val_indices]

# This is a simplified training call. A real implementation would require
# subsetting the MultiOmicsData object.
# For now, we'll just pass the full object.
ps, st = train_multiomics_model(multi_omics_model, multi_omics_data, y_train,
                                multi_omics_data, y_val,
                                10, 32, 3)

println("Model training complete.")

# --- 5. Result Interpretation ---
println("\nStep 5: Interpreting the results (Placeholder)...")
println("In a real analysis, you would now use the trained model to make predictions,")
println("evaluate its performance, and interpret the learned attention weights to")
println("understand how the different omics data types contribute to the prediction.")

println("\nMulti-omics integration example finished.")
