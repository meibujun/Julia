# PedigreeTools

PedigreeTools is a Julia package for high-performance manipulation and validation of livestock pedigree data. 

## Features
- Load pedigree data from CSV, TSV/TXT, or Excel files
- Automatic column-name standardization and cleaning of auxiliary columns (sex, birthdate, breed)
- Bidirectional mapping between original IDs and integers with optional threaded or process-based parallelism
- Restoration of original IDs after processing
- Pedigree validation (cycle detection, isolated nodes, missing ancestors)
- Lineage extraction for multi-generation tracing
- Basic pedigree graph visualization using GraphPlot

## Quick Start
```julia
using PedigreeTools
using DataFrames

# Load data
ped = load_pedigree("pedigree.csv")

# Convert IDs
ped_int, mapper = convert_pedigree!(ped)

# Validate
info = validate_pedigree(ped_int)

# Restore
restored = restore_pedigree(ped_int, mapper)
```

Run the test suite with:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```
