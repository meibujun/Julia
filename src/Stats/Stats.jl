module Stats

using Reexport

include("GWAS.jl")
# include("REML.jl")
# include("Bayesian.jl")

@reexport using .GWAS

end # module Stats
