module AnimalBreeding

using LinearAlgebra
using SparseArrays
using Statistics
using Logging

include("data/DataManager.jl")
include("model/ModelSpec.jl")
include("evaluation/MixedModels.jl")
include("bayesian/Bayesian.jl")
include("ml/MachineLearning.jl")
include("ui/Interfaces.jl")
include("perf/Performance.jl")

using .DataManager
using .ModelSpec
using .MixedModels
using .Bayesian
using .MachineLearning
using .Interfaces
using .Performance

export DataRepository, load_phenotypes, load_pedigree, load_genotypes, load_environment,
       load_multiomics!, integrate_data!, validate_data, compute_relationship_matrix, clear_cache!,
       ModelSpec, RandomEffectSpec, TraitSpec, define_model, describe,
       run_evaluation, GeneticEvalResult,
       run_bayesian_evaluation, BayesResult, mcmc_diagnostics,
       train_ml_model, MLModel, predict, cross_validate,
       run_cli, configure_performance

end
