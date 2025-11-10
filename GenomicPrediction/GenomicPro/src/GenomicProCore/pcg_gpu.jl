# src/GenomicProGPU/pcg_gpu.jl

"""
    solve_gblup_gpu(G::Matrix{Float64}, y::Vector{Float64}, λ::Float64; kwargs...)

Solve genomic BLUP using GPU-accelerated preconditioned conjugate gradient.
...
"""
function solve_gblup_gpu(G::Matrix{Float64},
                        y::Vector{Float64},
                        λ::Float64;
                        tolerance::Float64 = 1e-6,
                        max_iterations::Int = 1000,
                        preconditioner::Symbol = :diagonal,
                        use_mixed_precision::Bool = true,
                        verbose::Bool = true)

    # Check GPU availability
    if !CUDA.functional()
        @warn "GPU not available, falling back to CPU implementation"
        return solve_gblup(G, y, λ, method=:pcg)
    end

    n = length(y)

    verbose && println("Solving GBLUP on GPU using PCG...")

    solve_start = time()

    compute_type = use_mixed_precision ? Float32 : Float64

    # Transfer data to GPU
    G_gpu = CuArray{compute_type}(G)
    y_gpu = CuArray{compute_type}(y)

    # System: (G + Iλ)a = y,  u = Ga

    # Setup preconditioner on GPU
    M_inv_diag_gpu = setup_preconditioner_gpu(G_gpu, λ, preconditioner)

    # Initialize vectors on GPU
    a_gpu = CUDA.zeros(compute_type, n)
    r_gpu = copy(y_gpu)
    z_gpu = r_gpu .* M_inv_diag_gpu
    p_gpu = copy(z_gpu)

    rz_old = CUDA.dot(r_gpu, z_gpu)

    converged = false
    iter = 0

    for k in 1:max_iterations
        iter = k

        # Cp = (G + Iλ)p
        Cp_gpu = (G_gpu * p_gpu) .+ p_gpu .* compute_type(λ)

        pCp = CUDA.dot(p_gpu, Cp_gpu)
        α = rz_old / pCp

        CUDA.CUBLAS.axpy!(n, α, p_gpu, 1, a_gpu, 1)
        CUDA.CUBLAS.axpy!(n, -α, Cp_gpu, 1, r_gpu, 1)

        residual_norm = CUDA.norm(r_gpu)
        if residual_norm < tolerance
            converged = true
            break
        end

        z_gpu = r_gpu .* M_inv_diag_gpu
        rz_new = CUDA.dot(r_gpu, z_gpu)
        β = rz_new / rz_old

        CUDA.CUBLAS.scal!(n, β, p_gpu, 1)
        CUDA.CUBLAS.axpy!(n, one(compute_type), z_gpu, 1, p_gpu, 1)

        rz_old = rz_new
    end

    # u = Ga
    u_gpu = G_gpu * a_gpu

    u_cpu = Array(u_gpu)

    solve_time = time() - solve_start

    return (
        breeding_values = Float64.(u_cpu),
        iterations = iter,
        converged = converged,
        solve_time = solve_time
    )
end

function setup_preconditioner_gpu(G_gpu::CuArray{T},
                                  λ::Float64,
                                  preconditioner_type::Symbol) where T
    if preconditioner_type == :diagonal
        M_inv_diag = 1.0 ./ (diag(G_gpu) .+ T(λ))
        return CuArray(M_inv_diag)
    else
        return CUDA.ones(T, size(G_gpu, 1))
    end
end
