module Methods

using Reexport

include("GBLUP.jl")
# include("ssGBLUP.jl")
# include("MultiTrait.jl")

@reexport using .GBLUP

end # module Methods
