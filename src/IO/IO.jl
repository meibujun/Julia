module IO

using Reexport

include("PlinkReader.jl")
# include("VcfReader.jl") # To be implemented
# include("HDF5Backend.jl") # To be implemented

@reexport using .PlinkReader

end # module IO
