"""
PLINK binary format (.bed/.bim/.fam) reader and writer.

PLINK .bed format specification:
- Magic bytes: 0x6c, 0x1b
- Mode byte: 0x01 (SNP-major mode)
- Genotype encoding (2 bits per genotype):
  - 00 → Homozygous reference (0)
  - 01 → Missing
  - 10 → Heterozygous (1)
  - 11 → Homozygous alternative (2)
"""

"""
    PlinkFiles

Container for PLINK file paths (.bed/.bim/.fam).
"""
struct PlinkFiles
    bed::String
    bim::String
    fam::String
end

"""
    PlinkFiles(prefix::String)

Create PlinkFiles from a prefix (e.g., "data" → "data.bed", "data.bim", "data.fam").
"""
function PlinkFiles(prefix::String)
    return PlinkFiles(
        prefix * ".bed",
        prefix * ".bim",
        prefix * ".fam"
    )
end

"""
    read_fam(fam_file::String) -> (sample_ids, family_info)

Read PLINK .fam file.

Returns:
- sample_ids: Vector of individual IDs
- family_info: Dict with keys :family_id, :paternal_id, :maternal_id, :sex, :phenotype
"""
function read_fam(fam_file::String)
    if !isfile(fam_file)
        throw(FileFormatError("FAM file not found: $fam_file", :fam, fam_file))
    end

    family_ids = String[]
    sample_ids = String[]
    paternal_ids = String[]
    maternal_ids = String[]
    sexes = Int[]
    phenotypes = Float64[]

    open(fam_file, "r") do io
        for (line_num, line) in enumerate(eachline(io))
            # Skip empty lines
            if isempty(strip(line))
                continue
            end

            # Split by whitespace
            parts = split(strip(line))
            if length(parts) != 6
                throw(FileFormatError(
                    "Invalid FAM file format at line $line_num: expected 6 columns, got $(length(parts))",
                    :fam,
                    fam_file
                ))
            end

            push!(family_ids, parts[1])
            push!(sample_ids, parts[2])
            push!(paternal_ids, parts[3])
            push!(maternal_ids, parts[4])
            push!(sexes, parse(Int, parts[5]))

            # Parse phenotype (can be -9 for missing)
            pheno_val = parse(Float64, parts[6])
            push!(phenotypes, pheno_val == -9.0 ? NaN : pheno_val)
        end
    end

    family_info = Dict{Symbol, Any}(
        :family_id => family_ids,
        :paternal_id => paternal_ids,
        :maternal_id => maternal_ids,
        :sex => sexes,
        :phenotype => phenotypes
    )

    return sample_ids, family_info
end

"""
    read_bim(bim_file::String) -> (marker_ids, marker_info)

Read PLINK .bim file.

Returns:
- marker_ids: Vector of SNP IDs
- marker_info: Dict with keys :chromosome, :genetic_distance, :position, :allele1, :allele2
"""
function read_bim(bim_file::String)
    if !isfile(bim_file)
        throw(FileFormatError("BIM file not found: $bim_file", :bim, bim_file))
    end

    chromosomes = String[]
    marker_ids = String[]
    genetic_distances = Float64[]
    positions = Int[]
    allele1s = String[]
    allele2s = String[]

    open(bim_file, "r") do io
        for (line_num, line) in enumerate(eachline(io))
            # Skip empty lines
            if isempty(strip(line))
                continue
            end

            # Split by whitespace
            parts = split(strip(line))
            if length(parts) != 6
                throw(FileFormatError(
                    "Invalid BIM file format at line $line_num: expected 6 columns, got $(length(parts))",
                    :bim,
                    bim_file
                ))
            end

            push!(chromosomes, parts[1])
            push!(marker_ids, parts[2])
            push!(genetic_distances, parse(Float64, parts[3]))
            push!(positions, parse(Int, parts[4]))
            push!(allele1s, parts[5])
            push!(allele2s, parts[6])
        end
    end

    marker_info = Dict{Symbol, Any}(
        :chromosome => chromosomes,
        :genetic_distance => genetic_distances,
        :position => positions,
        :allele1 => allele1s,
        :allele2 => allele2s
    )

    return marker_ids, marker_info
end

"""
    read_bed(bed_file::String, n_samples::Int, n_markers::Int) -> Matrix

Read PLINK .bed file.

PLINK encoding (2 bits per genotype):
- 00 → 0 (homozygous reference)
- 01 → missing
- 10 → 1 (heterozygous)
- 11 → 2 (homozygous alternative)

Returns a matrix of size (n_samples × n_markers) with values 0, 1, 2, or missing.
"""
function read_bed(bed_file::String, n_samples::Int, n_markers::Int)
    if !isfile(bed_file)
        throw(FileFormatError("BED file not found: $bed_file", :bed, bed_file))
    end

    # Calculate expected file size
    # 3 bytes header + ceil(n_samples / 4) bytes per SNP
    bytes_per_snp = cld(n_samples, 4)
    expected_size = 3 + bytes_per_snp * n_markers

    file_size = filesize(bed_file)
    if file_size != expected_size
        @warn "BED file size mismatch" expected=expected_size actual=file_size
    end

    # Initialize genotype matrix
    genotypes = Matrix{Union{UInt8, Missing}}(undef, n_samples, n_markers)

    open(bed_file, "r") do io
        # Read and verify magic numbers
        magic1 = read(io, UInt8)
        magic2 = read(io, UInt8)
        mode = read(io, UInt8)

        if magic1 != 0x6c || magic2 != 0x1b
            throw(FileFormatError(
                "Invalid BED file: incorrect magic numbers (expected 0x6c 0x1b, got 0x$(string(magic1, base=16)) 0x$(string(magic2, base=16)))",
                :bed,
                bed_file
            ))
        end

        if mode != 0x01
            throw(FileFormatError(
                "Unsupported BED file mode: expected SNP-major mode (0x01), got 0x$(string(mode, base=16))",
                :bed,
                bed_file
            ))
        end

        # Read genotypes (SNP-major mode)
        for j in 1:n_markers
            # Read bytes for this SNP
            for byte_idx in 1:bytes_per_snp
                byte_val = read(io, UInt8)

                # Decode up to 4 genotypes from this byte
                for bit_idx in 0:3
                    sample_idx = (byte_idx - 1) * 4 + bit_idx + 1
                    if sample_idx > n_samples
                        break
                    end

                    # Extract 2 bits
                    geno_code = (byte_val >> (bit_idx * 2)) & 0b11

                    # Decode according to PLINK format
                    genotypes[sample_idx, j] = if geno_code == 0b00
                        UInt8(0)  # Homozygous reference
                    elseif geno_code == 0b11
                        UInt8(2)  # Homozygous alternative
                    elseif geno_code == 0b10
                        UInt8(1)  # Heterozygous
                    else  # 0b01
                        missing   # Missing
                    end
                end
            end
        end
    end

    return genotypes
end

"""
    read_plink(prefix::String) -> CompactGenotypes

Read PLINK binary files (.bed/.bim/.fam) and return a CompactGenotypes object.

# Arguments
- `prefix::String`: File prefix (e.g., "data" for "data.bed", "data.bim", "data.fam")

# Returns
- `CompactGenotypes`: Genotype data with annotations

# Examples
```julia
# Read PLINK files
geno = read_plink("mydata")

# Access data
println("Samples: ", n_samples(geno))
println("Markers: ", n_markers(geno))
```
"""
function read_plink(prefix::String)
    files = PlinkFiles(prefix)
    return read_plink(files)
end

function read_plink(files::PlinkFiles)
    # Read .fam file
    sample_ids, family_info = read_fam(files.fam)
    n_samples = length(sample_ids)

    # Read .bim file
    marker_ids, marker_info = read_bim(files.bim)
    n_markers = length(marker_ids)

    # Read .bed file
    genotypes = read_bed(files.bed, n_samples, n_markers)

    # Create CompactGenotypes
    geno = CompactGenotypes(
        genotypes,
        sample_ids,
        marker_ids;
        chromosome = marker_info[:chromosome],
        position = marker_info[:position],
        ref_allele = marker_info[:allele1],
        alt_allele = marker_info[:allele2]
    )

    # Store additional metadata
    geno.metadata[:family_info] = family_info
    geno.metadata[:genetic_distance] = marker_info[:genetic_distance]

    return geno
end

"""
    write_bed(bed_file::String, genotypes::AbstractMatrix, n_samples::Int, n_markers::Int)

Write genotype data to PLINK .bed format.
"""
function write_bed(bed_file::String, genotypes::AbstractMatrix, n_samples::Int, n_markers::Int)
    open(bed_file, "w") do io
        # Write magic numbers and mode
        write(io, UInt8(0x6c))
        write(io, UInt8(0x1b))
        write(io, UInt8(0x01))  # SNP-major mode

        bytes_per_snp = cld(n_samples, 4)

        # Write genotypes (SNP-major mode)
        for j in 1:n_markers
            for byte_idx in 1:bytes_per_snp
                byte_val = UInt8(0)

                # Encode up to 4 genotypes in this byte
                for bit_idx in 0:3
                    sample_idx = (byte_idx - 1) * 4 + bit_idx + 1
                    if sample_idx > n_samples
                        break
                    end

                    geno_val = genotypes[sample_idx, j]

                    # Encode according to PLINK format
                    geno_code = if ismissing(geno_val)
                        UInt8(0b01)
                    elseif geno_val == 0
                        UInt8(0b00)
                    elseif geno_val == 1
                        UInt8(0b10)
                    elseif geno_val == 2
                        UInt8(0b11)
                    else
                        throw(DomainError(geno_val, "Invalid genotype value: must be 0, 1, 2, or missing"))
                    end

                    byte_val |= (geno_code << (bit_idx * 2))
                end

                write(io, byte_val)
            end
        end
    end
end

"""
    write_bim(bim_file::String, geno::CompactGenotypes)

Write marker information to PLINK .bim format.
"""
function write_bim(bim_file::String, geno::CompactGenotypes)
    open(bim_file, "w") do io
        genetic_distances = get(geno.metadata, :genetic_distance, zeros(n_markers(geno)))

        for i in 1:n_markers(geno)
            @printf(io, "%s\t%s\t%.6f\t%d\t%s\t%s\n",
                    geno.chromosome[i],
                    geno.marker_ids[i],
                    genetic_distances[i],
                    geno.position[i],
                    geno.ref_allele[i],
                    geno.alt_allele[i])
        end
    end
end

"""
    write_fam(fam_file::String, geno::CompactGenotypes)

Write sample information to PLINK .fam format.
"""
function write_fam(fam_file::String, geno::CompactGenotypes)
    family_info = get(geno.metadata, :family_info, nothing)

    open(fam_file, "w") do io
        for i in 1:n_samples(geno)
            if family_info !== nothing
                @printf(io, "%s %s %s %s %d %.1f\n",
                        family_info[:family_id][i],
                        geno.sample_ids[i],
                        family_info[:paternal_id][i],
                        family_info[:maternal_id][i],
                        family_info[:sex][i],
                        isnan(family_info[:phenotype][i]) ? -9.0 : family_info[:phenotype][i])
            else
                # Write minimal .fam file
                @printf(io, "%s %s 0 0 0 -9\n",
                        geno.sample_ids[i],
                        geno.sample_ids[i])
            end
        end
    end
end

"""
    write_plink(prefix::String, geno::CompactGenotypes)

Write CompactGenotypes to PLINK binary format (.bed/.bim/.fam).

# Arguments
- `prefix::String`: Output file prefix (e.g., "output" for "output.bed", "output.bim", "output.fam")
- `geno::CompactGenotypes`: Genotype data to write

# Examples
```julia
# Write PLINK files
write_plink("output", geno)
```
"""
function write_plink(prefix::String, geno::CompactGenotypes)
    files = PlinkFiles(prefix)

    # Decode genotypes to matrix
    genotypes = decode_to_matrix(geno)

    # Write files
    write_bed(files.bed, genotypes, n_samples(geno), n_markers(geno))
    write_bim(files.bim, geno)
    write_fam(files.fam, geno)

    return files
end
