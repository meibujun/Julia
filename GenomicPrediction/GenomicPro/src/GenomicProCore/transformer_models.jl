# src/GenomicProPredict/transformer_models.jl

"""
    GenomicTransformerModel <: AbstractDeepLearningModel

Transformer architecture with self-attention for genomic prediction.

Transformer models revolutionized sequence modeling through the self-attention mechanism,
which enables the network to weigh the importance of different genomic positions when
making predictions. Unlike convolutional networks that operate on fixed local windows,
attention mechanisms can identify relevant marker interactions across the entire genome,
capturing long-range dependencies between distant loci that jointly influence phenotypes.
This global receptive field proves particularly valuable for traits involving trans-acting
regulatory elements or epistatic interactions between unlinked chromosomes.

# Self-Attention Mechanism

The core innovation of transformers lies in the attention mechanism, which computes
relevance scores between all pairs of positions in the input sequence. For genomic
prediction, this enables the model to determine which markers are most informative for
predicting an individual's phenotype, dynamically adjusting the importance weights based
on the specific genotype pattern being evaluated.

The attention operation computes three representations from the input markers: queries,
keys, and values. These are obtained through learned linear projections of the genotype
matrix. The attention scores between positions i and j are calculated by measuring the
similarity between query i and key j, typically using scaled dot product. These scores
are normalized through a softmax operation to obtain attention weights that sum to one
across all positions. Finally, the output for position i is computed as a weighted sum
of all values, where weights correspond to the attention scores.

Mathematically, the attention mechanism is expressed as:

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

where Q, K, and V represent query, key, and value matrices respectively, and d_k is
the dimension of the key vectors. The scaling factor prevents dot products from becoming
excessively large in high dimensions, which would cause the softmax function to have
vanishingly small gradients.

# Multi-Head Attention

Rather than computing a single attention function, multi-head attention applies multiple
attention operations in parallel, each with different learned linear projections. This
allows the model to attend to information from different representation subspaces at
different positions simultaneously. For genomic data, different attention heads might
focus on distinct types of genetic interactions: one head could capture cis-regulatory
relationships between nearby markers, while another identifies trans-acting effects
across chromosomes.

The outputs of all attention heads are concatenated and linearly transformed to produce
the final multi-head attention output. This mechanism has proven remarkably effective
across diverse sequence modeling tasks, from natural language processing to protein
structure prediction, and now extends to genomic prediction where identifying relevant
marker interactions is paramount.

# Positional Encoding

Transformers lack inherent awareness of sequence order since the attention mechanism
treats all positions symmetrically. For genomic data, preserving information about
marker positions along chromosomes is crucial, as nearby markers exhibit linkage
disequilibrium and functional genomic elements occupy specific chromosomal locations.
Positional encodings address this limitation by augmenting the input representations
with position-specific signals that enable the model to distinguish between markers
based on their genomic coordinates.

Common encoding strategies include sinusoidal functions of varying frequencies that
create unique patterns for each position, or learned embeddings that are optimized
during training to capture relevant positional relationships. For genomic applications,
incorporating chromosome identity as an additional encoding dimension helps the model
recognize intra-chromosomal versus inter-chromosomal interactions.

# Architecture Components

## Encoder Layers
The transformer encoder consists of stacked layers, each containing a multi-head
self-attention sublayer followed by a position-wise feedforward network. Residual
connections bypass each sublayer, facilitating gradient flow during training, while
layer normalization stabilizes learning dynamics. This architecture enables the model
to build increasingly abstract representations of genomic patterns through successive
transformations.

## Feedforward Networks
Following each attention sublayer, a position-wise feedforward network applies the
same fully connected layers independently to each position. This component introduces
additional nonlinearity and capacity to the model, enabling it to learn complex
transformations of the attention-weighted genomic representations.

## Pooling and Prediction Head
After processing through all encoder layers, the sequence of marker representations
must be aggregated into a single prediction. Common strategies include averaging all
position representations to obtain a global genomic profile, selecting the representation
of a special classification token prepended to the input sequence, or applying attention
pooling that learns optimal aggregation weights. The final prediction head consists of
one or more dense layers mapping the pooled representation to the phenotype value.

# Model Configuration

## Attention Parameters
- `n_attention_heads::Int`: Number of parallel attention mechanisms, typically 4 to 16
- `attention_dim::Int`: Dimensionality of attention representations, commonly 256 to 512
- `n_encoder_layers::Int`: Depth of transformer encoder, ranging from 2 to 12 layers

## Feedforward Network
- `feedforward_dim::Int`: Hidden dimension in position-wise networks, often 4× attention dimension
- `activation::Symbol`: Nonlinearity for feedforward networks, typically ReLU or GELU

## Regularization
- `dropout_rate::Float64`: Dropout probability applied to attention and feedforward sublayers
- `attention_dropout::Float64`: Specialized dropout for attention weights

## Training Configuration
- `learning_rate::Float64`: Initial learning rate for optimizer
- `warmup_steps::Int`: Linear learning rate warmup for stable training
- `batch_size::Int`: Training batch size
- `n_epochs::Int`: Maximum training epochs

# Biological Interpretation Through Attention

The learned attention weights provide interpretable insights into genetic architecture.
Visualizing attention patterns reveals which genomic regions the model considers most
relevant when making predictions for specific individuals. Markers receiving high attention
weights across many individuals indicate loci with consistent effects on the trait. In
contrast, context-dependent attention patterns suggest epistatic interactions where a
marker's importance depends on the genetic background at other loci.

Attention weights can be aggregated across the test population to generate genome-wide
importance scores analogous to GWAS p-values, identifying genomic regions harboring
causal variants. The multi-head attention structure provides additional granularity,
with different heads potentially corresponding to distinct biological mechanisms or
genetic pathways influencing the trait.

# Computational Requirements

Transformer models are computationally intensive due to the quadratic complexity of
self-attention with respect to sequence length. For genomic datasets with tens of
thousands of markers, the attention mechanism must compute and store an attention
matrix of size (number of markers) squared, which quickly becomes prohibitive. Several
strategies mitigate this challenge, including processing chromosomes independently to
reduce effective sequence length, applying sparse attention patterns that restrict
attention to local neighborhoods plus a small number of global positions, and utilizing
efficient attention implementations that reduce memory footprint through kernel fusion
and gradient checkpointing.

Training a transformer with 6 encoder layers, 8 attention heads, and dimension 512 on
a dataset of 10,000 individuals with 50,000 markers typically requires 12 to 16 GB of
GPU memory and completes within 2 to 4 hours on modern hardware. Inference is substantially
faster, generating predictions for thousands of test individuals in minutes.

# Examples
```julia
# Configure transformer for genomic prediction
model = GenomicTransformerModel(
    n_attention_heads = 8,
    attention_dim = 512,
    n_encoder_layers = 6,
    feedforward_dim = 2048,
    dropout_rate = 0.1,
    attention_dropout = 0.1,
    learning_rate = 0.0001,
    warmup_steps = 1000,
    batch_size = 128,
    n_epochs = 100
)

# Train transformer model
history = train_transformer!(model,
                            genotypes_train, phenotypes_train,
                            genotypes_val, phenotypes_val,
                            verbose = true,
                            use_gpu = true)

# Generate predictions
predictions = predict_transformer(model, genotypes_test)
accuracy = cor(predictions, phenotypes_test)

println("Transformer prediction accuracy: ", round(accuracy, digits=4))

# Extract and visualize attention patterns
attention_weights = extract_attention_weights(model, genotypes_test[1:10, :])
plot_attention_heatmap(attention_weights, "Transformer Attention Patterns")

# Identify important genomic regions
marker_importance = compute_global_attention_scores(model, genotypes_test)
top_markers = sortperm(marker_importance, rev=true)[1:100]

println("Top 100 markers by attention:")
for (rank, marker_idx) in enumerate(top_markers[1:10])
    println("  Rank $rank: Marker $marker_idx, Score: $(round(marker_importance[marker_idx], digits=4))")
end
```

# Performance Characteristics

Transformers demonstrate particular strengths for traits with complex genetic architectures
involving long-range interactions. Comparative studies show that attention mechanisms
provide modest improvements over CNNs for highly polygenic traits with predominantly
local effects, but substantial gains for traits involving distant epistatic interactions
or trans-regulatory effects. The interpretability advantage of attention weights offers
additional value beyond raw prediction accuracy, enabling biological discovery and
hypothesis generation.

# References
- Vaswani et al. (2017) NIPS 30:5998-6008 (Original transformer architecture)
- Jumper et al. (2021) Nature 596:583-589 (AlphaFold2, transformers for biology)
- Tay et al. (2022) JMLR 23:1-249 (Efficient transformers survey)

# See Also
- [`train_transformer!`](@ref): Transformer training procedure
- [`extract_attention_weights`](@ref): Access learned attention patterns
- [`visualize_genomic_attention`](@ref): Interpret attention mechanisms
"""
struct GenomicTransformerModel <: AbstractDeepLearningModel
    n_attention_heads::Int
    attention_dim::Int
    n_encoder_layers::Int
    feedforward_dim::Int
    activation::Symbol
    dropout_rate::Float64
    attention_dropout::Float64
    learning_rate::Float64
    warmup_steps::Int
    batch_size::Int
    n_epochs::Int
    early_stopping_patience::Int

    parameters::Dict{Symbol, Any}

    function GenomicTransformerModel(;
                                    n_attention_heads::Int = 8,
                                    attention_dim::Int = 512,
                                    n_encoder_layers::Int = 6,
                                    feedforward_dim::Int = 2048,
                                    activation::Symbol = :gelu,
                                    dropout_rate::Float64 = 0.1,
                                    attention_dropout::Float64 = 0.1,
                                    learning_rate::Float64 = 0.0001,
                                    warmup_steps::Int = 1000,
                                    batch_size::Int = 128,
                                    n_epochs::Int = 100,
                                    early_stopping_patience::Int = 10)

        @assert attention_dim % n_attention_heads == 0 "attention_dim must be divisible by n_attention_heads"
        @assert n_attention_heads > 0 "Number of attention heads must be positive"
        @assert n_encoder_layers > 0 "Number of encoder layers must be positive"

        parameters = Dict{Symbol, Any}()

        new(n_attention_heads, attention_dim, n_encoder_layers, feedforward_dim,
            activation, dropout_rate, attention_dropout, learning_rate, warmup_steps,
            batch_size, n_epochs, early_stopping_patience, parameters)
    end
end