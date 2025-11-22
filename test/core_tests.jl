using GenomicPro3
using Test
using DataFrames
using Statistics

@testset "Core Module" begin
    
    @testset "CompactGenotypes" begin
        # Create synthetic data
        n_samples = 4
        n_snps = 5
        # Packed size: ceil(4/4) = 1 row
        data = zeros(UInt8, 1, n_snps)
        
        # Manually set some values
        # Sample 1 (Row 1, bits 0-1): Homo Ref (00) -> 0
        # Sample 2 (Row 1, bits 2-3): Hetero (10) -> 2 (in bits) -> 1.0 (float)
        # Sample 3 (Row 1, bits 4-5): Homo Alt (11) -> 3 (in bits) -> 2.0 (float)
        # Sample 4 (Row 1, bits 6-7): Missing (01) -> 1 (in bits) -> NaN
        
        # SNP 1
        data[1, 1] = 0b01111000 # 01(4) 11(3) 10(2) 00(1) -> Rev order in bits?
        # Bits: 0-1 (Sample 1), 2-3 (Sample 2)...
        # Val = (byte >> 0) & 3
        # We want Sample 1 to be 00 (0)
        # Sample 2 to be 10 (2)
        # Sample 3 to be 11 (3)
        # Sample 4 to be 01 (1)
        # Byte = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 0)
        #      = 01000000 | 00110000 | 00001000 | 00000000
        #      = 01111000 = 0x78
        data[1, 1] = 0x78
        
        geno = CompactGenotypes(data, n_samples, n_snps, ["S$i" for i in 1:n_samples], ["SNP$j" for j in 1:n_snps])
        
        # Test get_snp
        snp1 = get_snp(geno, 1)
        @test snp1[1] == 0.0
        @test snp1[2] == 1.0
        @test snp1[3] == 2.0
        @test isnan(snp1[4])
        
        # Test MAF
        # SNP 1: 0, 1, 2, NaN (valid: 3 samples)
        # Ref count: 2 (from 0) + 1 (from 1) = 3
        # Alt count: 1 (from 1) + 2 (from 2) = 3
        # Total alleles: 2 * 3 = 6
        # Freq Alt: 3/6 = 0.5
        # MAF: 0.5
        mafs = maf(geno)
        @test mafs[1] ≈ 0.5
    end
    
    @testset "Phenotypes" begin
        df = DataFrame(ID = ["S1", "S2", "S3"], Trait1 = [1.0, 2.0, 3.0], Cov1 = [0.1, 0.2, 0.3])
        pheno = PhenotypeData(df, ["Trait1"], ["Cov1"], "ID")
        
        y = get_trait(pheno, "Trait1")
        @test y == [1.0, 2.0, 3.0]
        
        X = get_covariates(pheno)
        @test size(X) == (3, 1)
        @test X[1, 1] == 0.1
        
        standardize!(pheno)
        y_std = get_trait(pheno, "Trait1")
        @test abs(mean(y_std)) < 1e-6
        @test std(y_std) ≈ 1.0
    end

end
