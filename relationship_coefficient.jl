function relationship_matrix(pedigree::Dict{String,Tuple{Union{String,Nothing},Union{String,Nothing}}})
    individuals = sort(collect(keys(pedigree)))
    idx = Dict(ind => i for (i, ind) in enumerate(individuals))
    n = length(individuals)
    A = zeros(Float64, n, n)

    for i in 1:n
        ind = individuals[i]
        sire, dam = pedigree[ind]
        if sire === nothing && dam === nothing
            A[i, i] = 1.0
        else
            val = 1.0
            if sire !== nothing && dam !== nothing
                val += 0.5 * A[idx[sire], idx[dam]]
            end
            A[i, i] = val
        end

        for j in 1:i-1
            s = sire === nothing ? 0.0 : A[idx[sire], j]
            d = dam === nothing ? 0.0 : A[idx[dam], j]
            A[i, j] = 0.5 * (s + d)
            A[j, i] = A[i, j]
        end
    end

    return A, idx
end

function relationship_coefficient(pedigree, x::String, y::String)
    A, idx = relationship_matrix(pedigree)
    return A[idx[x], idx[y]]
end

if abspath(PROGRAM_FILE) == @__FILE__
    pedigree = Dict(
        "F" => (nothing, nothing),
        "M1" => (nothing, nothing),
        "M2" => (nothing, nothing),
        "B" => ("F", "M1"),
        "C" => ("F", "M2")
    )

    A, idx = relationship_matrix(pedigree)
    println("Relationship matrix:")
    println(A)
    println("Relationship coefficient between B and C: ", A[idx["B"], idx["C"]])
end
