# src/GenomicProMultiOmics/encoders.jl

using Lux

"""
    build_snp_encoder(input_dim::Int, latent_dim::Int)

Builds a 1D CNN encoder for SNP data.
"""
function build_snp_encoder(input_dim::Int, latent_dim::Int)
    return Chain(
        ReshapeLayer((1, input_dim, 1)),
        Conv((1, 5), 1 => 32, relu),
        MaxPool((1, 2)),
        Conv((1, 5), 32 => 64, relu),
        MaxPool((1, 2)),
        FlattenLayer(),
        Dense(64 * (input_dim ÷ 4), latent_dim)
    )
end

"""
    build_rnaseq_vae(input_dim::Int, latent_dim::Int)

Builds a Variational Autoencoder for RNA-seq data.
"""
function build_rnaseq_vae(input_dim::Int, latent_dim::Int)
    encoder = Chain(
        Dense(input_dim, 512, relu),
        Dense(512, 256, relu)
    )

    decoder = Chain(
        Dense(latent_dim, 256, relu),
        Dense(256, 512, relu),
        Dense(512, input_dim)
    )

    return VAE(encoder, decoder, latent_dim)
end

struct VAE{E, D}
    encoder::E
    decoder::D
    latent_dim::Int
end
