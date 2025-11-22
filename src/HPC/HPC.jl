module HPC

using Reexport

include("CUDABackend.jl")
# include("MultiThreading.jl")

@reexport using .CUDABackend

end # module HPC
