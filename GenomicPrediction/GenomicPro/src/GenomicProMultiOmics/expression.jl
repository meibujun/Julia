# src/GenomicProMultiOmics/expression.jl

using DataFrames

"""
    ExpressionData <: AbstractMultiOmicsData

Concrete implementation for storing and managing gene expression data.
"""
struct ExpressionData <: AbstractMultiOmicsData
    table::DataFrame
    sample_id_col::Symbol
    gene_id_col::Symbol

    function ExpressionData(table::DataFrame, sample_id_col::Symbol, gene_id_col::Symbol)
        @assert sample_id_col in names(table) "Sample ID column not found."
        @assert gene_id_col in names(table) "Gene ID column not found."
        new(table, sample_id_col, gene_id_col)
    end
end
