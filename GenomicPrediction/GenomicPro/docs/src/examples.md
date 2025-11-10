# Examples

This page provides examples of how to use `GenomicPro.jl` for various analyses.

## GBLUP Analysis

```julia
using GenomicPro

# Load data
geno = read_genotypes("genotypes.vcf")
pheno = read_phenotypes("phenotypes.csv", :ID, [:MilkYield])

# Compute GRM
G = compute_grm(geno)
y = pheno.table[:, :MilkYield]

# Estimate variance components
vc = estimate_variance_components(G, y)
λ = vc.residual_variance / vc.genetic_variance

# Solve for breeding values
results = solve_gblup(G, y, λ)
println("Breeding values: ", results.breeding_values)
```

## SSGBLUP Analysis

```julia
using GenomicPro

# Load data
geno = read_genotypes("genotypes.vcf")
pheno = read_phenotypes("phenotypes.csv", :ID, [:MilkYield])
ped = read_pedigree("pedigree.csv", :ID, :Sire, :Dam)

# ... (match IDs and get genotyped_indices)

# Compute GRM
G = compute_grm(geno)
y = pheno.table[:, :MilkYield]

# Estimate variance components
vc = estimate_variance_components(G, y[genotyped_indices])
λ = vc.residual_variance / vc.genetic_variance

# Solve for breeding values
results = solve_ssgblup(G, ped, genotyped_indices, y, λ)
println("Breeding values: ", results.breeding_values)
```

## Multi-Omics Integration

The following is a complete, runnable example for multi-omics integration. For more details, see the script in the `examples/` directory.

```julia
# examples/08_multi_omics_integration.jl

using GenomicPro
using DataFrames
using Random

# --- 1. Data Simulation ---
const N_SAMPLES = 500
const N_SNPS = 1000
const N_GENES = 100

snp_data = rand([0, 1, 2], N_SAMPLES, N_SNPS)
expression_data = randn(N_SAMPLES, N_GENES)
phenotypes = vec(sum(snp_data[:, 1:10], dims=2) * 0.1 + sum(expression_data[:, 1:5], dims=2) * 0.2 + randn(N_SAMPLES))

snp_df = DataFrame(snp_data, :auto)
snp_df[!, :ID] = ["ID_$(i)" for i in 1:N_SAMPLES]

expr_df = DataFrame(expression_data, :auto)
expr_df[!, :ID] = ["ID_$(i)" for i in 1:N_SAMPLES]
expr_df[!, :GeneID] = ["Gene_$(i)" for i in 1:N_GENES]

pheno_df = DataFrame(ID = snp_df.ID, Phenotype = phenotypes)

# --- 2. Data Loading ---
genotypes = TwoBitGenotypes(Matrix(snp_df[!, Not(:ID)]))
expression = ExpressionData(expr_df, :ID, :GeneID)
phenotype_data = PhenotypeData(pheno_df, :ID, [:Phenotype])

multi_omics_data = MultiOmicsData(Dict(
    :genotypes => genotypes,
    :expression => expression
))

# --- 3. Model Definition ---
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

# --- 4. Model Training ---
train_frac = 0.8
n_train = Int(floor(train_frac * N_SAMPLES))
train_indices = 1:n_train
val_indices = (n_train + 1):N_SAMPLES

y_train = pheno_df.Phenotype[train_indices]
y_val = pheno_df.Phenotype[val_indices]

ps, st = train_multiomics_model(multi_omics_model, multi_omics_data, y_train,
                                multi_omics_data, y_val,
                                10, 32, 3)
```
