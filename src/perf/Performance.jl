module Performance

using Logging
using Base.Threads

export configure_performance, PerformanceConfig

"""
    PerformanceConfig

记录用户的性能配置偏好，包括线程数、是否启用分布式以及 GPU 状态。
"""
mutable struct PerformanceConfig
    threads::Int
    distributed::Bool
    gpu::Bool
end

const CONFIG = PerformanceConfig(nthreads(), false, false)

"""
    configure_performance(; threads = nothing, distributed = false, gpu = false)

更新性能配置。Julia 的线程数必须在启动时指定，函数会在不匹配时给出提示。
若请求启用 GPU，会自动检测 CUDA 是否可用。
"""
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
