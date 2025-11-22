module Genotypes

import ..GenomicTypes
using LinearAlgebra
using Statistics
using Mmap

export CompactGenotypes, create_view
export get_snp, get_sample, maf, call_rate, missing_rate

"""
    CompactGenotypes <: GenomicTypes.AbstractGenotypeData

Efficient 2-bit storage for genotypes, aligned with PLINK .bed format (SNP-Major).
Data is stored as a Matrix{UInt8}.
Rows: Packed Samples (ceil(n_samples / 4))
Cols: SNPs (n_snps)

Encoding (Standard PLINK):
00 (0) -> Homozygous for Allele 1 (Ref) -> 0.0
01 (1) -> Missing -> NaN
10 (2) -> Heterozygous -> 1.0
11 (3) -> Homozygous for Allele 2 (Alt) -> 2.0
"""
struct CompactGenotypes <: GenomicTypes.AbstractGenotypeData
    data::Matrix{UInt8}      # Rows: ceil(n_samples/4), Cols: n_snps
    n_samples::Int
    n_snps::Int
    sample_ids::Vector{String}
    snp_ids::Vector{String}
end

"""
    get_snp(g::CompactGenotypes, snp_idx::Int)

Decode a single SNP column into a Float32 vector.
"""
function get_snp(g::CompactGenotypes, snp_idx::Int)
    n = g.n_samples
    res = Vector{Float32}(undef, n)
    
    # Column in data corresponds to snp_idx
    # We iterate over rows (packed samples)
    
    row_idx = 1
    sample_idx = 1
    
    @inbounds while sample_idx <= n
        byte = g.data[row_idx, snp_idx]
        
        # Process 4 samples in the byte
        # PLINK order: bits 0-1 are sample 1, 2-3 are sample 2, etc.
        
        for k in 0:3
            if sample_idx > n; break; end
            
            val = (byte >> (2*k)) & 0x03
            
            if val == 0x00      # 00 -> Homo Ref (0)
                res[sample_idx] = 0.0f0
            elseif val == 0x02  # 10 -> Heteroz (1)
                res[sample_idx] = 1.0f0
            elseif val == 0x03  # 11 -> Homo Alt (2)
                res[sample_idx] = 2.0f0
            else                # 01 -> Missing
                res[sample_idx] = NaN32
            end
            
            sample_idx += 1
        end
        row_idx += 1
    end
    return res
end

"""
    maf(g::CompactGenotypes)

Compute Minor Allele Frequency (MAF) for all SNPs.
"""
function maf(g::CompactGenotypes)
    n_snps = g.n_snps
    n_samples = g.n_samples
    mafs = zeros(Float32, n_snps)
    
    # Iterate over SNPs (columns) - this is now memory efficient!
    Threads.@threads for j in 1:n_snps
        count_ref = 0
        count_alt = 0
        n_valid = 0
        
        row_idx = 1
        sample_idx = 1
        
        @inbounds while sample_idx <= n_samples
            byte = g.data[row_idx, j]
            
            for k in 0:3
                if sample_idx > n_samples; break; end
                
                val = (byte >> (2*k)) & 0x03
                
                if val == 0x00 # 0
                    count_ref += 2
                    n_valid += 1
                elseif val == 0x02 # 1
                    count_ref += 1
                    count_alt += 1
                    n_valid += 1
                elseif val == 0x03 # 2
                    count_alt += 2
                    n_valid += 1
                end
                
                sample_idx += 1
            end
            row_idx += 1
        end
        
        if n_valid > 0
            freq_alt = count_alt / (2 * n_valid)
            mafs[j] = min(freq_alt, 1.0f0 - freq_alt)
        else
            mafs[j] = 0.0f0
        end
    end
    
    return mafs
end

"""
    missing_rate(g::CompactGenotypes)

Compute missing rate per SNP.
"""
function missing_rate(g::CompactGenotypes)
    n_snps = g.n_snps
    n_samples = g.n_samples
    rates = zeros(Float32, n_snps)
    
    Threads.@threads for j in 1:n_snps
        n_missing = 0
        row_idx = 1
        sample_idx = 1
        
        @inbounds while sample_idx <= n_samples
            byte = g.data[row_idx, j]
            for k in 0:3
                if sample_idx > n_samples; break; end
                val = (byte >> (2*k)) & 0x03
                if val == 0x01
                    n_missing += 1
                end
                sample_idx += 1
            end
            row_idx += 1
        end
        rates[j] = n_missing / n_samples
    end
    return rates
end

end # module Genotypes
