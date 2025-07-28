# Mating optimization module

module MatingOptimization

using LinearAlgebra
using Statistics
using JuMP
using GLPK
using ..CoreTypes
using ..OGBLUPModel

export optimal_mating, MatingPlan, MinimizeInbreeding, MaximizeGeneticGain

abstract type MatingStrategy end
struct MinimizeInbreeding <: MatingStrategy end
struct MaximizeGeneticGain <: MatingStrategy end

struct MatingPlan
    matings::Vector{Tuple{Int, Int}}
    expected_inbreeding::Float64
    expected_gain::Float64
end

function optimal_mating(model::OGBLUP, population::Population;
                  strategy::MatingStrategy=MaximizeGeneticGain(),
                  n_matings::Int=100,
                  sex::Vector{Symbol}) # :M or :F

    @info "Optimizing mating plan with strategy: \$(typeof(strategy))"

    males = findall(sex .== :M)
    females = findall(sex .== :F)

    if isa(strategy, MaximizeGeneticGain)
        plan = maximize_gain_plan(model, population, males, females, n_matings)
    elseif isa(strategy, MinimizeInbreeding)
        plan = minimize_inbreeding_plan(model, population, males, females, n_matings)
    else
        error("Unknown mating strategy")
    end

    return plan
end

function maximize_gain_plan(model, population, males, females, n_matings)
    gebv = predict_gebv(model, components=:additive)

    # Simple strategy: mate best males with best females
    top_males = sortperm(vec(gebv[males, :]), rev=true)
    top_females = sortperm(vec(gebv[females, :]), rev=true)

    matings = [(males[top_males[i]], females[top_females[i]]) for i in 1:min(length(top_males), length(top_females), n_matings)]

    return MatingPlan(matings, 0.0, 0.0) # Placeholder for proper evaluation
end

function minimize_inbreeding_plan(model, population, males, females, n_matings)
    G = model.grms.G
    n_males = length(males)
    n_females = length(females)

    opt_model = Model(GLPK.Optimizer)

    @variable(opt_model, x[1:n_males, 1:n_females], Bin)

    # Objective: minimize sum of relationships of mated pairs
    @objective(opt_model, Min, sum(G[males[i], females[j]] * x[i,j] for i=1:n_males, j=1:n_females))

    # Constraint: total number of matings
    @constraint(opt_model, sum(x) == n_matings)

    # Constraint: each individual used at most once (simplified)
    for i in 1:n_males
        @constraint(opt_model, sum(x[i, :]) <= 1)
    end
    for j in 1:n_females
        @constraint(opt_model, sum(x[:, j]) <= 1)
    end

    optimize!(opt_model)

    matings = Tuple{Int, Int}[]
    if termination_status(opt_model) == MOI.OPTIMAL
        x_val = value.(x)
        for i in 1:n_males
            for j in 1:n_females
                if x_val[i,j] > 0.5
                    push!(matings, (males[i], females[j]))
                end
            end
        end
    end

    return MatingPlan(matings, 0.0, 0.0)
end

end
