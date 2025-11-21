"""
# Transformer Genomic Prediction Model

A Transformer-based model for genomic prediction that treats SNP sequences as tokens.
This allows the model to capture long-range dependencies and complex epistatic interactions
between markers across the genome.

## Model Architecture
- **Embedding Layer**: Projects SNP values (0, 1, 2) into a high-dimensional vector space.
- **Positional Encoding**: Adds information about the relative position of SNPs.
- **Transformer Blocks**: Multiple layers of Multi-Head Self-Attention and Feed-Forward Networks.
- **Global Pooling**: Aggregates information from all SNP tokens.
- **Output Head**: Predicts the phenotype (trait value).

## Usage
```julia
using GenomicPro2.Models

# Initialize model
model = TransformerGenomicModel(
    n_snps = 10000,
    d_model = 64,
    n_heads = 4,
    n_layers = 2
)

# Train
train_transformer!(model, genotypes, phenotypes)

# Predict
predictions = predict_transformer(model, new_genotypes)
```
"""
module TransformerGenomic

using Flux
using Flux: @functor
using LinearAlgebra
using Statistics
using Random
using ..Core: GenotypeData, PhenotypeData
using ..Data: CompactGenotypes

export TransformerGenomicModel, train_transformer!, predict_transformer

# ============================================================================
# Model Components
# ============================================================================

"""
    PositionalEncoding(d_model::Int, max_len::Int=50000)

Adds sinusoidal positional information to the input embeddings.
"""
struct PositionalEncoding
    pe::Matrix{Float32}
end

@functor PositionalEncoding

function PositionalEncoding(d_model::Int, max_len::Int=50000)
    pe = zeros(Float32, d_model, max_len)
    for pos in 1:max_len
        for i in 0:2:(d_model-1)
            pe[i+1, pos] = sin(pos / (10000^(i/d_model)))
            if i+1 < d_model
                pe[i+2, pos] = cos(pos / (10000^(i/d_model)))
            end
        end
    end
    return PositionalEncoding(pe)
end

function (p::PositionalEncoding)(x::AbstractArray)
    # x shape: (d_model, seq_len, batch_size)
    seq_len = size(x, 2)
    # Add positional encoding (broadcast over batch dimension)
    return x .+ p.pe[:, 1:seq_len]
end

"""
    TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int, dropout::Float64=0.1)

Standard Transformer Encoder Block:
MultiHeadAttention -> Add & Norm -> FeedForward -> Add & Norm
"""
struct TransformerBlock
    attention::MultiHeadAttention
    norm1::LayerNorm
    ff::Chain
    norm2::LayerNorm
    dropout::Dropout
end

@functor TransformerBlock

function TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int, dropout::Float64=0.1)
    return TransformerBlock(
        MultiHeadAttention(d_model, n_heads, dropout=dropout),
        LayerNorm(d_model),
        Chain(
            Dense(d_model, d_ff, relu),
            Dropout(dropout),
            Dense(d_ff, d_model),
            Dropout(dropout)
        ),
        LayerNorm(d_model),
        Dropout(dropout)
    )
end

function (b::TransformerBlock)(x::AbstractArray)
    # x shape: (d_model, seq_len, batch_size)
    
    # Self-Attention block with residual connection
    # Note: Flux's MultiHeadAttention expects inputs as (d_model, seq_len, batch_size)
    attn_out = b.attention(x, x, x)
    x1 = b.norm1(x .+ b.dropout(attn_out))
    
    # Feed-forward block with residual connection
    ff_out = b.ff(x1)
    x2 = b.norm2(x1 .+ ff_out)
    
    return x2
end

"""
    TransformerGenomicModel

Main Transformer model for genomic prediction.
"""
struct TransformerGenomicModel
    embedding::Dense  # Linear projection of SNP values
    pos_encoding::PositionalEncoding
    blocks::Chain
    final_pool::GlobalMeanPool
    output_head::Chain
    
    # Metadata
    config::Dict{Symbol, Any}
end

@functor TransformerGenomicModel

"""
    TransformerGenomicModel(; 
        n_snps::Int, 
        d_model::Int=64, 
        n_heads::Int=4, 
        n_layers::Int=2, 
        d_ff::Int=256,
        dropout::Float64=0.1
    )

Constructor for the Transformer model.
"""
function TransformerGenomicModel(; 
    n_snps::Int, 
    d_model::Int=64, 
    n_heads::Int=4, 
    n_layers::Int=2, 
    d_ff::Int=256,
    dropout::Float64=0.1
)
    # Config for saving/loading
    config = Dict(
        :n_snps => n_snps,
        :d_model => d_model,
        :n_heads => n_heads,
        :n_layers => n_layers,
        :d_ff => d_ff,
        :dropout => dropout
    )

    return TransformerGenomicModel(
        Dense(1, d_model), # Project scalar SNP value to d_model vector
        PositionalEncoding(d_model, n_snps),
        Chain([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in 1:n_layers]...),
        GlobalMeanPool(),
        Chain(
            Dense(d_model, d_model, relu),
            Dropout(dropout),
            Dense(d_model, 1)
        ),
        config
    )
end

function (m::TransformerGenomicModel)(x::AbstractArray)
    # Input x: (1, seq_len, batch_size) - raw SNP values
    
    # 1. Embedding & Positional Encoding
    # Reshape for Dense layer: (1, seq_len * batch_size)
    batch_size = size(x, 3)
    seq_len = size(x, 2)
    
    x_flat = reshape(x, 1, :)
    x_emb = m.embedding(x_flat) # (d_model, seq_len * batch_size)
    
    # Reshape back to (d_model, seq_len, batch_size)
    x_emb = reshape(x_emb, :, seq_len, batch_size)
    
    x_pos = m.pos_encoding(x_emb)
    
    # 2. Transformer Blocks
    x_trans = m.blocks(x_pos)
    
    # 3. Global Pooling (Average over sequence length)
    # GlobalMeanPool expects (width, height, channels, batch) for images
    # or (features, seq_len, batch) for sequences? 
    # Flux's GlobalMeanPool is typically for CNNs. Let's implement manual mean.
    x_pool = dropdims(mean(x_trans, dims=2), dims=2) # (d_model, batch_size)
    
    # 4. Output Head
    out = m.output_head(x_pool) # (1, batch_size)
    
    return out
end

# ============================================================================
# Training & Prediction
# ============================================================================

"""
    train_transformer!(model, genotypes, phenotypes; 
                       epochs=50, batch_size=32, lr=0.001, val_split=0.2)

Train the Transformer model.
"""
function train_transformer!(
    model::TransformerGenomicModel,
    genotypes::CompactGenotypes,
    phenotypes::PhenotypeData;
    epochs::Int=50,
    batch_size::Int=32,
    lr::Float64=0.001,
    val_split::Float64=0.2,
    verbose::Bool=true
)
    # Prepare data
    # Convert CompactGenotypes to Float32 matrix for Flux
    # Note: For very large datasets, we should use a custom data loader to avoid
    # loading everything into memory at once. For now, we assume it fits.
    
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)
    
    if verbose
        @info "Preparing data for Transformer..."
    end
    
    # Extract data (this might be heavy, consider batch generator for optimization)
    X_raw = Float32.(to_matrix(genotypes; impute=true))
    # Reshape to (1, seq_len, batch_size) for the model
    X = permutedims(reshape(X_raw, n_samples, n_snps, 1), (3, 2, 1))
    
    y = Float32.(phenotypes.values)
    # Normalize target
    y_mean, y_std = mean(y), std(y)
    y_norm = (y .- y_mean) ./ y_std
    y_norm = reshape(y_norm, 1, :) # (1, batch_size)
    
    # Split train/val
    n_val = floor(Int, n_samples * val_split)
    n_train = n_samples - n_val
    
    indices = randperm(n_samples)
    train_idx = indices[1:n_train]
    val_idx = indices[n_train+1:end]
    
    X_train = X[:, :, train_idx]
    y_train = y_norm[:, train_idx]
    
    X_val = X[:, :, val_idx]
    y_val = y_norm[:, val_idx]
    
    # Optimizer
    opt = Flux.setup(Adam(lr), model)
    
    # Training loop
    if verbose
        @info "Starting training..."
        @info "  Model: $(model.config[:n_layers]) layers, $(model.config[:n_heads]) heads"
        @info "  Train samples: $n_train"
        @info "  Val samples: $n_val"
    end
    
    train_loader = Flux.DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
    
    best_val_loss = Inf
    
    for epoch in 1:epochs
        losses = Float32[]
        
        for (x_batch, y_batch) in train_loader
            val, grads = Flux.withgradient(model) do m
                pred = m(x_batch)
                Flux.mse(pred, y_batch)
            end
            
            Flux.update!(opt, model, grads[1])
            push!(losses, val)
        end
        
        # Validation
        val_pred = model(X_val)
        val_loss = Flux.mse(val_pred, y_val)
        
        if val_loss < best_val_loss
            best_val_loss = val_loss
            # Save best model state if needed
        end
        
        if verbose && (epoch % 5 == 0 || epoch == 1)
            train_loss = mean(losses)
            @info "Epoch $epoch: Train Loss = $(round(train_loss, digits=5)), Val Loss = $(round(val_loss, digits=5))"
        end
    end
    
    if verbose
        @info "Training complete. Best Val Loss: $best_val_loss"
    end
    
    return model
end

"""
    predict_transformer(model, genotypes)

Predict phenotypes using the trained Transformer model.
"""
function predict_transformer(model::TransformerGenomicModel, genotypes::CompactGenotypes)
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)
    
    # Prepare input
    X_raw = Float32.(to_matrix(genotypes; impute=true))
    X = permutedims(reshape(X_raw, n_samples, n_snps, 1), (3, 2, 1))
    
    # Predict
    # Process in batches to avoid OOM
    batch_size = 32
    preds = Float32[]
    
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch = X[:, :, i:end_idx]
        p = model(batch)
        append!(preds, vec(p))
    end
    
    return preds
end

end # module TransformerGenomic
