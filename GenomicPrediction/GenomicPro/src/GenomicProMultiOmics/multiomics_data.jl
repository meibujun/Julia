# src/GenomicProMultiOmics/multiomics_data.jl

"""
    MultiOmicsData <: AbstractMultiOmicsData

A container for holding multiple omics data types.
"""
struct MultiOmicsData <: AbstractMultiOmicsData
    data::Dict{Symbol, AbstractGenomicData}

    function MultiOmicsData(data::Dict{Symbol, AbstractGenomicData})
        # Validation can be added here to ensure data consistency
        new(data)
    end
end
