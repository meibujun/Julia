# KernelModels.jl
module KernelModels
using LinearAlgebra, DataFrames, SparseArrays, Statistics
using ..DataProcessing
using ..CoreAlgorithm: _standardize_genotypes
import ..AbstractModel

@doc raw"""
    ssGBLUPModel(lambda::Float64) <: AbstractModel
"""
mutable struct ssGBLUPModel <: AbstractModel
    lambda::Float64
    effects::Vector{Float64}
    intercept::Float64
    all_individuals::Vector{Int}
    ssGBLUPModel(lambda::Float64) = new(lambda, [], 0.0, [])
end

function build_A_inv(pedigree::DataFrame)
    n = size(pedigree, 1)
    id_map = Dict(pedigree.ID[i] => i for i in 1:n)
    A_inv = spzeros(Float64, n, n)

    # Corrected Henderson's rules implementation
    for i in 1:n
        sire = get(id_map, pedigree.Sire[i], 0)
        dam = get(id_map, pedigree.Dam[i], 0)

        if sire != 0 && dam != 0 # Both parents known
            A_inv[i, i] += 2.0
            A_inv[sire, sire] += 0.5
            A_inv[dam, dam] += 0.5
            A_inv[i, sire] -= 1.0
            A_inv[sire, i] -= 1.0
            A_inv[i, dam] -= 1.0
            A_inv[dam, i] -= 1.0
            A_inv[sire, dam] += 0.5
            A_inv[dam, sire] += 0.5
        elseif sire != 0 || dam != 0 # One parent known
            p = sire != 0 ? sire : dam
            A_inv[i, i] += 4.0 / 3.0
            A_inv[i, p] -= 2.0 / 3.0
            A_inv[p, i] -= 2.0 / 3.0
            A_inv[p, p] += 1.0 / 3.0
        else # Both parents unknown
            A_inv[i, i] += 1.0
        end
    end
    return A_inv
end

function fit!(model::ssGBLUPModel, data::GenomicData)
    println("开始 ssGBLUP 模型训练...")

    y_df = data.phenotypes

    all_ped_ids = data.pedigree[!, :ID]
    model.all_individuals = all_ped_ids
    id_map = Dict(id => i for (i, id) in enumerate(all_ped_ids))

    genotyped_ids = data.genotypes[!, :ID]
    genotyped_indices = [id_map[id] for id in genotyped_ids]

    # Efficiently calculate G and its inverse
    G_raw = Matrix(data.genotypes[!, 2:end])
    G = calculate_grm(G_raw)
    G_inv = inv(G + I * 1e-6)

    # Build the sparse A inverse matrix
    A_inv = build_A_inv(data.pedigree)
    A22_inv = A_inv[genotyped_indices, genotyped_indices]

    # Construct H inverse directly using sparse matrices
    H_inv = copy(A_inv)
    # The modification should be done carefully for sparse matrices
    H_inv[genotyped_indices, genotyped_indices] += G_inv - A22_inv

    n_total = length(all_ped_ids)

    # Phenotype incidence matrix setup
    pheno_indices = [id_map[id] for id in y_df.ID]
    n_pheno = length(pheno_indices)

    X = ones(n_pheno, 1)
    Z = spzeros(n_pheno, n_total)
    for (i, p_idx) in enumerate(pheno_indices)
        Z[i, p_idx] = 1.0
    end

    y = y_df.y

    # Mixed Model Equations (MME)
    C11 = X'X
    C12 = X'Z
    C21 = Z'X
    C22 = Z'Z + H_inv * model.lambda

    LHS = [C11 C12; C21 C22]
    RHS = [X'y; Z'y]

    solutions = LHS \ RHS

    model.intercept = solutions[1]
    model.effects = solutions[2:end]

    println("ssGBLUP 训练完成。")
    return nothing
end

function predict(model::ssGBLUPModel, new_ids::Vector{Int})
    id_map = Dict(id => i for (i, id) in enumerate(model.all_individuals))
    indices = [id_map[id] for id in new_ids]
    return model.intercept .+ model.effects[indices]
end
end
