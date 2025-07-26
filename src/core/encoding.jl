# NOIA-based orthogonal genotype encoding

module Encoding

using LinearAlgebra
using Statistics
using ..Constants
using ..CoreTypes

export noia_encode, update_allele_frequencies!

function noia_encode(genotypes::AbstractMatrix{T},
                    allele_freq::AbstractVector{<:Real};
                    ploidy::Int=2) where T<:Real

    n, m = size(genotypes)
    @assert length(allele_freq) == m "Allele frequency dimension mismatch"

    X = zeros(Float64, n, m)
    Z = zeros(Float64, n, m)

    if ploidy == 2
        Threads.@threads for j in 1:m
            p = allele_freq[j]
            q = 1.0 - p

            if p < EPSILON || q < EPSILON
                continue
            end

            scale_add = sqrt(2 * p * q)
            scale_dom = sqrt(p * q)

            @inbounds for i in 1:n
                g = genotypes[i, j]
                if g < 0
                    X[i, j] = 0.0
                    Z[i, j] = 0.0
                    continue
                end
                X[i, j] = (g - 2p) / scale_add
                if g == 1
                    Z[i, j] = (1 - 2p*q) / scale_dom
                else
                    Z[i, j] = -2p*q / scale_dom
                end
            end
        end
    else
        # Polyploid support can be added here
    end

    return X, Z
end

function update_allele_frequencies!(geno_data::GenotypeData)
    Threads.@threads for j in 1:geno_data.n_markers
        valid_genotypes = filter(g -> g >= 0, geno_data.genotypes[:, j])
        if !isempty(valid_genotypes)
            geno_data.allele_freq[j] = mean(valid_genotypes) / geno_data.ploidy
        end
    end
    return geno_data
end

end
