# AnimalBreeding.jl/src/selection/mating_plan.jl

using DataFrames
using Random

"""
    design_mating_plan(sires, dams, relationship_matrix, sire_ebvs, dam_ebvs; max_inbreeding=0.0625)

Designs a mating plan to minimize inbreeding and maximize genetic gain.

This implementation uses a simple greedy approach to select the best sire for each dam.

# Arguments
- `sires`: A vector of sire indices.
- `dams`: A vector of dam indices.
- `relationship_matrix`: The relationship matrix.
- `sire_ebvs`: A vector of sire EBVs.
- `dam_ebvs`: A vector of dam EBVs.

# Keyword Arguments
- `max_inbreeding`: The maximum allowed inbreeding coefficient (default `0.0625`).

# Returns
A `DataFrame` with the mating plan.
"""
function design_mating_plan(sires::Vector{Int},
                            dams::Vector{Int},
                            relationship_matrix::AbstractMatrix,
                            sire_ebvs::Vector,
                            dam_ebvs::Vector;
                            max_inbreeding::Float64=0.0625)

    mating_plan = DataFrame(sire = Int[], dam = Int[], inbreeding = Float64[], expected_gain = Float64[])

    for (dam_idx, dam) in enumerate(dams)
        best_sire = 0
        max_gain = -Inf

        for (sire_idx, sire) in enumerate(sires)
            inbreeding = relationship_matrix[sire, dam]
            if inbreeding <= max_inbreeding
                expected_gain = (sire_ebvs[sire_idx] + dam_ebvs[dam_idx]) / 2
                if expected_gain > max_gain
                    max_gain = expected_gain
                    best_sire = sire
                end
            end
        end

        if best_sire != 0
            inbreeding = relationship_matrix[best_sire, dam]
            push!(mating_plan, (best_sire, dam, inbreeding, max_gain))
        end
    end

    return mating_plan
end

export design_mating_plan
