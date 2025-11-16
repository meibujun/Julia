"""
# IO Module

File input/output operations for genomic data.

Supported formats:
- PLINK (.bed/.bim/.fam)
- Phenotype CSV files
- Simple text-based genotype files

## Examples

```julia
# Read PLINK files
geno = read_plink("data.bed")

# Read phenotypes
pheno = read_phenotypes("phenotypes.csv")

# Write genotypes
write_plink("output.bed", geno)
```
"""
module IO

using ..Core
using ..Data
using LinearAlgebra
using Printf

# File format readers
include("plink.jl")
include("phenotypes.jl")

# Export public API
export read_plink, write_plink
export read_phenotypes, write_phenotypes
export PlinkFiles, PhenotypeData

end # module IO
