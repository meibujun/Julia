module GenomicPro3

using Reexport

# Export Core Modules
include("GenomicCore/GenomicCore.jl")
@reexport using .GenomicCore

include("IO/IO.jl")
@reexport using .IO

include("HPC/HPC.jl")
@reexport using .HPC

include("Stats/Stats.jl")
@reexport using .Stats

include("Methods/Methods.jl")
@reexport using .Methods

include("AI/AI.jl")
@reexport using .AI

end # module GenomicPro3
