module Performance

using Logging
using Base.Threads

export configure_performance, PerformanceConfig

mutable struct PerformanceConfig
    threads::Int
    distributed::Bool
    gpu::Bool
end

const CONFIG = PerformanceConfig(nthreads(), false, false)

function configure_performance(; threads::Union{Int,Nothing} = nothing, distributed::Bool = false, gpu::Bool = false)
    if threads !== nothing && threads != nthreads()
        @warn "Threads are configured at Julia startup; requested $(threads) but currently $(nthreads())"
    end
    CONFIG.distributed = distributed
    if gpu
        CONFIG.gpu = _gpu_available()
        !CONFIG.gpu && @warn "GPU requested but CUDA not available"
    else
        CONFIG.gpu = false
    end
    return CONFIG
end

function _gpu_available()
    try
        @eval using CUDA
        return CUDA.functional()
    catch err
        @debug "CUDA unavailable" exception = err
        return false
    end
end

end
