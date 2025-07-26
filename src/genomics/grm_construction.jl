# Genomic Relationship Matrix (GRM) construction module

module GRMConstruction

using LinearAlgebra
using Statistics
using SparseArrays
using Distributed
using SharedArrays
using ProgressMeter
using ..CoreTypes
using ..Constants
using ..Encoding
using ..Validation

export compute_grm, compute_grm_set, build_sparse_grm,
       compute_weighted_grm, compute_adaptive_grm,
       GRMMethod, VanRadenMethod, YangMethod

# GRM computation methods
abstract type GRMMethod end
struct VanRadenMethod <: GRMMethod end
struct YangMethod <: GRMMethod end

function compute_grm(X::AbstractMatrix{<:Real},
                    allele_freq::AbstractVector{<:Real};
                    method::GRMMethod=VanRadenMethod(),
                    scale::Bool=true,
                    center::Bool=true,
                    ridge::Real=RIDGE_LAMBDA)

    n, m = size(X)

    W = X .- 2 .* allele_freq'

    if isa(method, VanRadenMethod)
        G = compute_vanraden_grm(W, allele_freq, scale)
    elseif isa(method, YangMethod)
        G = compute_yang_grm(W, allele_freq)
    else
        error("Unknown GRM method")
    end

    G = 0.5 * (G + G')
    G[diagind(G)] .+= ridge

    validate_grm(G)

    return G
end

function compute_vanraden_grm(W::AbstractMatrix{<:Real},
                             allele_freq::AbstractVector{<:Real},
                             scale::Bool)
    n, m = size(W)
    scale_factor = scale ? sum(2 .* allele_freq .* (1 .- allele_freq)) : m
    G = (W * W') / scale_factor
    return G
end

function compute_yang_grm(W::AbstractMatrix{<:Real},
                         allele_freq::AbstractVector{<:Real})
    n, m = size(W)
    G = zeros(n, n)

    for j in 1:m
        p = allele_freq[j]
        if p > EPSILON && p < 1 - EPSILON
            var_j = 2 * p * (1 - p)
            w_j = view(W, :, j)
            G .+= (w_j * w_j') / var_j
        end
    end

    G ./= m
    return G
end

function compute_epistatic_grms(G::AbstractMatrix{<:Real},
                               D::AbstractMatrix{<:Real})
    n = size(G, 1)
    G_AA = G .* G
    G_AD = G .* D
    G_DD = D .* D

    G_AA .*= n / tr(G_AA)
    G_AD .*= n / tr(G_AD)
    G_DD .*= n / tr(G_DD)

    return G_AA, G_AD, G_DD
end

function compute_grm_set(geno_data::GenotypeData;
                        include_dominance::Bool=true,
                        include_epistasis::Bool=true)

    update_allele_frequencies!(geno_data)
    X, Z = noia_encode(geno_data.genotypes, geno_data.allele_freq, ploidy=geno_data.ploidy)

    G = compute_grm(X, geno_data.allele_freq)

    n = geno_data.n_individuals
    D = include_dominance ? compute_grm(Z, geno_data.allele_freq) : Matrix(1.0I, n, n)

    if include_epistasis
        G_AA, G_AD, G_DD = compute_epistatic_grms(G, D)
    else
        G_AA, G_AD, G_DD = (Matrix(1.0I, n, n) for _ in 1:3)
    end

    return GRMSet(G, D, G_AA, G_AD, G_DD, nothing, NamedTuple(), 0, geno_data.allele_freq)
end
