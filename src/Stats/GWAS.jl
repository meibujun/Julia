module GWAS

using ...GenomicCore
using ...HPC
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using CUDA

export perform_gwas

"""
    perform_gwas(geno::AbstractGenotypeData, pheno::PhenotypeData; trait::String, method=:linear)

Perform Genome-Wide Association Study.
"""
function perform_gwas(geno::AbstractGenotypeData, pheno::PhenotypeData; trait::String, method=:linear)
    if method == :linear
        if HPC.has_gpu()
            return gwas_linear_gpu(geno, pheno, trait)
        else
            return gwas_linear_cpu(geno, pheno, trait)
        end
    else
        error("Method $method not implemented.")
    end
end

"""
    gwas_linear_gpu(geno, pheno, trait)

GPU-accelerated Linear Model GWAS using Frisch-Waugh-Lovell theorem.
Model: y = Xβ + Gγ + ε
1. Regress y on X -> residuals y_r
2. Regress G on X -> residuals G_r
3. Regress y_r on G_r -> γ
"""
function gwas_linear_gpu(geno::AbstractGenotypeData, pheno::PhenotypeData, trait::String)
    @info "Starting GPU GWAS for trait: $trait"
    
    # 1. Prepare Data
    y_cpu = GenomicCore.Phenotypes.get_trait(pheno, trait)
    X_cpu = GenomicCore.Phenotypes.get_covariates(pheno)
    
    # Handle missing values (simple complete case for now)
    # In a real world-class app, we'd do sophisticated imputation or subsetting
    # Assuming complete data for this implementation or pre-imputed
    
    n = length(y_cpu)
    
    # Add intercept to covariates if not present
    if size(X_cpu, 2) == 0 || !any(all(X_cpu .== 1, dims=1))
        X_cpu = hcat(ones(n), X_cpu)
    end
    
    # Transfer to GPU
    d_y = CuArray(Float32.(y_cpu))
    d_X = CuArray(Float32.(X_cpu))
    
    # 2. Unpack Genotypes to GPU
    @info "Unpacking genotypes to GPU..."
    d_G = HPC.unpack_genotypes_gpu(geno) # (n, m)
    
    # 3. Covariate Projection (FWL)
    # P = X(X'X)^-1X'
    # M = I - P
    # We need M * y and M * G
    # Use QR: X = QR. P = QQ'. M = I - QQ'
    # M * y = y - Q(Q'y)
    
    @info "Projecting covariates..."
    F = qr(d_X)
    Q = Matrix(F.Q) # CUDA.jl QR returns packed Q, need to materialize or use properly
    d_Q = CuArray(Q) # Ensure it's a dense CuArray
    
    # Project y
    # y_r = y - Q * (Q' * y)
    d_Qty = d_Q' * d_y
    d_y_r = d_y - d_Q * d_Qty
    
    # Project G
    # G_r = G - Q * (Q' * G)
    # This is the heavy lifting: (n, k) * (k, m) -> (n, m)
    d_QtG = d_Q' * d_G
    d_G_r = d_G - d_Q * d_QtG
    
    # 4. Simple Regression (Univariate)
    # y_r = g_r * gamma + e
    # gamma = (g_r' * y_r) / (g_r' * g_r)
    
    @info "Computing regression statistics..."
    
    # Denominator: sum(g_r.^2)
    d_G_r_sq = d_G_r .^ 2
    d_denom = sum(d_G_r_sq, dims=1) # (1, m)
    
    # Numerator: g_r' * y_r -> sum(g_r .* y_r)
    # Broadcast y_r to match G_r shape implicitly or use matmul
    # y_r is vector (n). G_r is (n, m).
    # G_r' * y_r is (m, 1)
    d_num = d_G_r' * d_y_r
    
    d_beta = d_num' ./ d_denom # (1, m)
    
    # 5. Standard Errors and P-values
    # RSS = sum((y_r - g_r * gamma)^2)
    # But calculating residuals for every SNP is expensive (n*m)
    # RSS = SST - SSR
    # SST = sum(y_r.^2)
    # SSR = beta^2 * sum(g_r.^2) = beta * num
    
    d_SST = sum(d_y_r .^ 2)
    d_SSR = d_beta .* d_num' # (1, m)
    d_RSS = d_SST .- d_SSR
    
    # Degrees of freedom: n - k - 1 (intercept included in k if in X)
    # k is cols in X. +1 for SNP.
    dof = n - size(d_X, 2) - 1
    
    d_sigma2 = d_RSS ./ dof
    d_se = sqrt.(d_sigma2 ./ d_denom)
    
    d_t = d_beta ./ d_se
    
    # Transfer back to CPU
    beta = Array(d_beta)[1, :]
    se = Array(d_se)[1, :]
    t_stat = Array(d_t)[1, :]
    
    # P-values
    dist = TDist(dof)
    pvals = 2 .* ccdf.(dist, abs.(t_stat))
    
    # 6. Result DataFrame
    return DataFrame(
        SNP = geno.snp_ids,
        Beta = beta,
        SE = se,
        P = pvals
    )
end

"""
    gwas_linear_cpu(geno, pheno, trait)

CPU fallback for Linear Model GWAS using Frisch-Waugh-Lovell theorem.
"""
function gwas_linear_cpu(geno::AbstractGenotypeData, pheno::PhenotypeData, trait::String)
    @info "Starting CPU GWAS for trait: $trait"
    
    # 1. Prepare Data
    y = GenomicCore.Phenotypes.get_trait(pheno, trait)
    X = GenomicCore.Phenotypes.get_covariates(pheno)
    
    n = length(y)
    if size(X, 2) == 0 || !any(all(X .== 1, dims=1))
        X = hcat(ones(n), X)
    end
    
    # 2. Covariate Projection (FWL)
    # Project y onto null space of X
    F = qr(X)
    Q = Matrix(F.Q)
    
    # y_r = y - Q * (Q' * y)
    y_r = y - Q * (Q' * y)
    
    # Pre-calculate denominator part for standard error
    # We need sum(g_r^2).
    
    n_snps = geno.n_snps
    betas = Vector{Float64}(undef, n_snps)
    ses = Vector{Float64}(undef, n_snps)
    pvals = Vector{Float64}(undef, n_snps)
    
    dof = n - size(X, 2) - 1
    dist = TDist(dof)
    
    # 3. Iterate over SNPs
    # Optimization: Process in blocks to use BLAS Level 3 if possible
    # But extracting genotypes is the bottleneck.
    
    Threads.@threads for j in 1:n_snps
        # Extract SNP (0, 1, 2, NaN)
        g = GenomicCore.Genotypes.get_snp(geno, j)
        
        # Handle missing in G (simple imputation to mean for now)
        # In robust version, we should exclude samples or impute properly
        # For speed here: impute to mean
        valid_idx = .!isnan.(g)
        if count(valid_idx) < n
            μ_g = mean(g[valid_idx])
            g[.!valid_idx] .= μ_g
        end
        
        # Project G
        # g_r = g - Q * (Q' * g)
        # This is O(N * K) per SNP.
        Qtg = Q' * g
        g_r = g - Q * Qtg
        
        # Regression
        denom = sum(g_r.^2)
        if denom < 1e-9
            betas[j] = NaN
            ses[j] = NaN
            pvals[j] = NaN
            continue
        end
        
        num = dot(g_r, y_r)
        beta = num / denom
        
        # Stats
        # RSS = SST - SSR
        # SST = sum(y_r^2) -> Constant for all SNPs? No, y_r is constant.
        # But we need RSS of the full model.
        # RSS = sum((y_r - g_r * beta)^2)
        
        rss = sum((y_r .- g_r .* beta).^2)
        sigma2 = rss / dof
        se = sqrt(sigma2 / denom)
        
        betas[j] = beta
        ses[j] = se
        
        t_stat = beta / se
        pvals[j] = 2 * ccdf(dist, abs(t_stat))
    end
    
    return DataFrame(
        SNP = geno.snp_ids,
        Beta = betas,
        SE = ses,
        P = pvals
    )
end

end # module GWAS
