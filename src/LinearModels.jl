module LinearModels

include("LinearModels/Chapter1_Foundations.jl")
include("LinearModels/Chapter2_Estimation.jl")
include("LinearModels/Chapter3_Inference.jl")
include("LinearModels/Chapter4_MixedModels.jl")
include("LinearModels/Chapter5_Diagnostics.jl")
include("LinearModels/Chapter6_ModelSelection.jl")
include("LinearModels/Chapter7_Prediction.jl")
include("LinearModels/Chapter8_VarianceComponents.jl")
include("LinearModels/Chapter9_Bayesian.jl")
include("LinearModels/Chapter10_Computation.jl")

using .LinearModelsChapter1
using .LinearModelsChapter2
using .LinearModelsChapter3
using .LinearModelsChapter4
using .LinearModelsChapter5
using .LinearModelsChapter6
using .LinearModelsChapter7
using .LinearModelsChapter8
using .LinearModelsChapter9
using .LinearModelsChapter10

end # module
