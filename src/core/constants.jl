# Core constants and configuration for OrthogonalGenomics.jl

module Constants

# Numerical constants
const EPSILON = 1e-10
const RIDGE_LAMBDA = 1e-6
const MAX_ITERATIONS = 1000
const CONVERGENCE_TOL = 1e-8

# Genetic constants
const PLOIDY_DIPLOID = 2
const MISSING_GENOTYPE = -9
const HETEROZYGOUS = 1

# Model defaults
const DEFAULT_MAF_FILTER = 0.01
const DEFAULT_CALL_RATE = 0.95
const DEFAULT_HWE_PVALUE = 1e-6

# Parallel processing
const MIN_MARKERS_FOR_PARALLEL = 10000
const MIN_INDIVIDUALS_FOR_PARALLEL = 1000

# Memory management
const CHUNK_SIZE = 10000
const MAX_MEMORY_GB = 16.0

# File formats
const SUPPORTED_FORMATS = [:plink, :vcf, :csv, :hdf5]
const COMPRESSION_FORMATS = [:gzip, :bzip2, :xz]

export EPSILON, RIDGE_LAMBDA, MAX_ITERATIONS, CONVERGENCE_TOL,
       PLOIDY_DIPLOID, MISSING_GENOTYPE, HETEROZYGOUS,
       DEFAULT_MAF_FILTER, DEFAULT_CALL_RATE, DEFAULT_HWE_PVALUE,
       MIN_MARKERS_FOR_PARALLEL, MIN_INDIVIDUALS_FOR_PARALLEL,
       CHUNK_SIZE, MAX_MEMORY_GB,
       SUPPORTED_FORMATS, COMPRESSION_FORMATS

end
