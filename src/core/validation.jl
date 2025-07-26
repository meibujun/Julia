# Input validation and data integrity checks

module Validation

using Statistics
using LinearAlgebra
using ..Constants
using ..CoreTypes

export validate_grm

function validate_grm(G::AbstractMatrix; check_pd::Bool=true)

    n = size(G, 1)
    if size(G, 2) != n
        throw(ArgumentError("GRM must be square"))
    end

    if !issymmetric(G)
        max_asym = maximum(abs.(G - G'))
        if max_asym > EPSILON
            throw(ArgumentError("GRM is not symmetric (max asymmetry: \$max_asym)"))
        end
    end

    if check_pd
        eigenvals = eigvals(Symmetric(G))
        min_eigenval = minimum(eigenvals)

        if min_eigenval < -EPSILON
            throw(ArgumentError("GRM is not positive semi-definite (min eigenvalue: \$min_eigenval)"))
        end
    end

    return true
end

end
