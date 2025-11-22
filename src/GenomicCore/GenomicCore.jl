module GenomicCore

# using Reexport

println("Loading GenomicCore...")
include("GenomicTypes.jl")
println("Included GenomicTypes")
println("Names in GenomicCore: ", names(@__MODULE__, all=true))
include("Genotypes.jl")
println("Included Genotypes")
include("Pedigree.jl")
include("Phenotypes.jl")

using .GenomicTypes
using .Genotypes
using .Pedigree
using .Phenotypes

export GenomicTypes, Genotypes, Pedigree, Phenotypes

# Re-export common types
using .GenomicTypes: AbstractGenomicModel, AbstractGenotypeData
export AbstractGenomicModel, AbstractGenotypeData

using .Genotypes: CompactGenotypes, get_snp, maf
export CompactGenotypes, get_snp, maf

using .Phenotypes: PhenotypeData, get_trait, get_covariates, standardize!
export PhenotypeData, get_trait, get_covariates, standardize!

using .Pedigree: PedigreeData
export PedigreeData

end # module Core
