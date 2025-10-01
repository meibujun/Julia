module Interfaces

using ArgParse
using DataFrames
using JSON3
using CSV

import ..DataManager: DataRepository, load_phenotypes, load_pedigree, load_genotypes, load_environment,
    integrate_data!, validate_data
import ..ModelSpec: define_model
import ..MixedModels: run_evaluation
import ..Bayesian: run_bayesian_evaluation
import ..MachineLearning: cross_validate

export run_cli

function run_cli(args = ARGS)
    isempty(args) && return _print_usage()
    command = first(args)
    rest = args[2:end]
    if command == "validate"
        _run_validate(rest)
    elseif command == "evaluate"
        _run_evaluate(rest)
    elseif command == "bayes"
        _run_bayes(rest)
    elseif command == "ml"
        _run_ml(rest)
    else
        println("Unknown command $(command)")
        _print_usage()
    end
end

function _run_validate(args)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--phenotype"
            help = "Path to phenotype CSV"
            arg_type = String
        "--pedigree"
            help = "Path to pedigree CSV"
            arg_type = String
        "--genotype"
            help = "Path to genotype CSV"
            arg_type = String
        "--environment"
            help = "Path to environment CSV"
            arg_type = String
    end
    parsed = parse_args(args, settings; as_symbols = true)
    repo = DataRepository()
    if haskey(parsed, :phenotype)
        repo.phenotypes = load_phenotypes(parsed[:phenotype])
    end
    if haskey(parsed, :pedigree)
        repo.pedigrees = load_pedigree(parsed[:pedigree])
    end
    if haskey(parsed, :genotype)
        repo.genotypes = load_genotypes(parsed[:genotype])
    end
    if haskey(parsed, :environment)
        repo.environments = load_environment(parsed[:environment])
    end
    integrate_data!(repo)
    report = validate_data(repo)
    println(JSON3.write(report; indent = 2))
end

function _run_evaluate(args)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--phenotype"
            arg_type = String
            required = true
        "--pedigree"
            arg_type = String
        "--genotype"
            arg_type = String
        "--trait"
            arg_type = String
            required = true
        "--method"
            arg_type = String
            default = "BLUP"
        "--fixed"
            nargs = ArgParse.+
            help = "Fixed effect column names"
        "--random"
            nargs = ArgParse.+
            help = "Random effect factor names"
        "--h2"
            arg_type = Float64
            default = 0.3
    end
    parsed = parse_args(args, settings; as_symbols = true)
    repo = DataRepository()
    repo.phenotypes = load_phenotypes(parsed[:phenotype])
    if haskey(parsed, :pedigree)
        repo.pedigrees = load_pedigree(parsed[:pedigree])
    end
    if haskey(parsed, :genotype)
        repo.genotypes = load_genotypes(parsed[:genotype])
    end
    integrate_data!(repo)
    traits = [parsed[:trait]]
    fixed = haskey(parsed, :fixed) ? Symbol.(parsed[:fixed]) : Symbol[]
    random = haskey(parsed, :random) ? [(r, :additive) for r in parsed[:random]] : []
    model = define_model(traits = Symbol.(traits), fixed = fixed, random = random)
    result = run_evaluation(model, repo; trait = Symbol(parsed[:trait]), method = Symbol(parsed[:method]), h2 = parsed[:h2])
    println(result)
end

function _run_bayes(args)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--phenotype"
            arg_type = String
            required = true
        "--genotype"
            arg_type = String
            required = true
        "--trait"
            arg_type = String
            required = true
        "--method"
            arg_type = String
            default = "BayesC"
        "--n_iter"
            arg_type = Int
            default = 4000
        "--burn_in"
            arg_type = Int
            default = 1000
        "--thin"
            arg_type = Int
            default = 5
    end
    parsed = parse_args(args, settings; as_symbols = true)
    repo = DataRepository()
    repo.phenotypes = load_phenotypes(parsed[:phenotype])
    repo.genotypes = load_genotypes(parsed[:genotype])
    integrate_data!(repo)
    model = define_model(traits = [Symbol(parsed[:trait])], random = [("animal", :additive)])
    result = run_bayesian_evaluation(model, repo; trait = Symbol(parsed[:trait]), method = Symbol(parsed[:method]),
        n_iter = parsed[:n_iter], burn_in = parsed[:burn_in], thin = parsed[:thin])
    println("Posterior mean intercept: $(result.intercept)")
    println("First 5 marker effects: $(result.marker_effects[1:min(end, 5)])")
end

function _run_ml(args)
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--features"
            arg_type = String
            required = true
        "--labels"
            arg_type = String
            required = true
        "--method"
            arg_type = String
            default = "RandomForest"
        "--folds"
            arg_type = Int
            default = 5
    end
    parsed = parse_args(args, settings; as_symbols = true)
    features = DataFrame(CSV.File(parsed[:features]))
    labels = Vector{Float64}(CSV.File(parsed[:labels])[:, 1])
    stats = cross_validate(features, labels, Symbol(parsed[:method]); n_folds = parsed[:folds])
    println(JSON3.write(stats; indent = 2))
end

function _print_usage()
    println("AnimalBreeding CLI")
    println("Commands: validate, evaluate, bayes, ml")
end

end
