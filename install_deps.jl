using Pkg
Pkg.activate(".")
Pkg.add(["DataFrames", "CSV", "Flux", "CUDA", "Distributions", "Reexport", "LinearAlgebra", "Statistics", "Mmap", "Test", "Printf", "Random", "SparseArrays", "DelimitedFiles"])
Pkg.instantiate()
