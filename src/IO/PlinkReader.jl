module PlinkReader

using ...GenomicCore
using Mmap
using DelimitedFiles

export read_plink

"""
    read_plink(bfile::String)

Read PLINK binary files (.bed, .bim, .fam).
Returns a `CompactGenotypes` object backed by a memory-mapped array.
This operation is Zero-Copy and extremely fast.
"""
function read_plink(bfile::String)
    bed_file = bfile * ".bed"
    bim_file = bfile * ".bim"
    fam_file = bfile * ".fam"

    # 1. Parse .fam (Samples)
    # Format: FID IID PID MID SEX PHENO
    if !isfile(fam_file)
        error("FAM file not found: $fam_file")
    end
    fam_data = readdlm(fam_file, String)
    n_samples = size(fam_data, 1)
    sample_ids = fam_data[:, 2] # Use IID

    # 2. Parse .bim (SNPs)
    # Format: CHR SNP CM BP A1 A2
    if !isfile(bim_file)
        error("BIM file not found: $bim_file")
    end
    bim_data = readdlm(bim_file, String)
    n_snps = size(bim_data, 1)
    snp_ids = bim_data[:, 2]

    # 3. Mmap .bed file
    if !isfile(bed_file)
        error("BED file not found: $bed_file")
    end

    open(bed_file, "r") do io
        # Check Magic Numbers
        magic = read(io, 3)
        if magic != [0x6c, 0x1b, 0x01]
            if magic[1:2] == [0x6c, 0x1b]
                error("Only SNP-major mode (magic 0x01) is supported. Found: $(magic[3])")
            else
                error("Invalid PLINK magic number: $magic")
            end
        end
        
        # Calculate dimensions
        # SNP-Major: n_snps blocks, each block has ceil(n_samples/4) bytes
        n_bytes_per_block = div(n_samples + 3, 4)
        
        # Check file size
        expected_size = 3 + n_bytes_per_block * n_snps
        actual_size = filesize(bed_file)
        if actual_size != expected_size
            @warn "BED file size mismatch. Expected: $expected_size, Actual: $actual_size. Data might be truncated or corrupt."
        end
        
        # Memory Map
        # We map the data part starting at offset 3
        # Matrix dims: (n_bytes_per_block, n_snps)
        # This matches our CompactGenotypes layout (Rows: packed samples, Cols: SNPs)
        data = Mmap.mmap(io, Matrix{UInt8}, (n_bytes_per_block, n_snps), 3)
        
        return CompactGenotypes(data, n_samples, n_snps, vec(sample_ids), vec(snp_ids))
    end
end

end # module PlinkReader
