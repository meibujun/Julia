"""
# 配置管理模块

提供灵活的配置管理系统，支持：
- TOML 配置文件
- 环境变量覆盖
- 默认值管理
- 配置验证

## 使用示例
```julia
using GenomicPro2.Config

# 加载配置
config = load_config("GenomicPro2.toml")

# 访问配置
println("Threads: ", config.compute.threads)
println("GPU: ", config.compute.use_gpu)

# 全局配置
set_global_config!(config)
cfg = get_config()
```
"""
module Config

using TOML

# ============================================================================
# 配置结构体
# ============================================================================

"""
计算配置
"""
struct ComputeConfig
    threads::Int
    use_gpu::Bool
    gpu_device_id::Int
    parallel_backend::Symbol  # :threads, :distributed
end

"""
内存配置
"""
struct MemoryConfig
    max_memory_gb::Float64
    chunk_size::Int
    use_compression::Bool
    cache_size_mb::Int
end

"""
IO 配置
"""
struct IOConfig
    temp_dir::String
    output_dir::String
    auto_cleanup::Bool
    buffer_size_mb::Int
end

"""
日志配置
"""
struct LogConfig
    level::String          # "DEBUG", "INFO", "WARN", "ERROR"
    log_file::Union{String, Nothing}
    console_output::Bool
    performance_logging::Bool
end

"""
Web API 配置
"""
struct APIConfig
    host::String
    port::Int
    cors_enabled::Bool
    max_upload_size_mb::Int
    session_timeout_minutes::Int
end

"""
分析配置
"""
struct AnalysisConfig
    default_model::String
    cross_validation_folds::Int
    significance_threshold::Float64
    max_iterations::Int
end

"""
主配置结构
"""
struct GenomicProConfig
    compute::ComputeConfig
    memory::MemoryConfig
    io::IOConfig
    logging::LogConfig
    api::APIConfig
    analysis::AnalysisConfig
    version::String
end

# ============================================================================
# 配置加载
# ============================================================================

"""
    load_config(config_file::String="GenomicPro2.toml")

从 TOML 文件加载配置。

如果文件不存在，使用默认配置。
环境变量可以覆盖配置项。

# 环境变量
- `GENOMICPRO_THREADS`: 覆盖线程数
- `GENOMICPRO_GPU`: 覆盖 GPU 使用（true/false）
- `GENOMICPRO_LOG_LEVEL`: 覆盖日志级别
- `GENOMICPRO_API_PORT`: 覆盖 API 端口

# 示例
```julia
config = load_config()  # 使用默认文件
config = load_config("my_config.toml")  # 自定义文件
```
"""
function load_config(config_file::String="GenomicPro2.toml")
    # 如果文件存在，读取
    if isfile(config_file)
        @info "加载配置文件: $config_file"
        data = TOML.parsefile(config_file)
    else
        @info "配置文件不存在，使用默认配置"
        data = Dict{String, Any}()
    end

    # 计算配置
    compute = ComputeConfig(
        get_env_int("GENOMICPRO_THREADS", get(get(data, "compute", Dict()), "threads", Threads.nthreads())),
        get_env_bool("GENOMICPRO_GPU", get(get(data, "compute", Dict()), "use_gpu", false)),
        get_env_int("GENOMICPRO_GPU_DEVICE", get(get(data, "compute", Dict()), "gpu_device_id", 0)),
        Symbol(get(get(data, "compute", Dict()), "parallel_backend", "threads"))
    )

    # 内存配置
    memory = MemoryConfig(
        get_env_float("GENOMICPRO_MAX_MEMORY", get(get(data, "memory", Dict()), "max_memory_gb", 16.0)),
        get(get(data, "memory", Dict()), "chunk_size", 5000),
        get(get(data, "memory", Dict()), "use_compression", true),
        get(get(data, "memory", Dict()), "cache_size_mb", 1024)
    )

    # IO 配置
    io = IOConfig(
        get_env("GENOMICPRO_TEMP_DIR", get(get(data, "io", Dict()), "temp_dir", tempdir())),
        get_env("GENOMICPRO_OUTPUT_DIR", get(get(data, "io", Dict()), "output_dir", "./results")),
        get(get(data, "io", Dict()), "auto_cleanup", true),
        get(get(data, "io", Dict()), "buffer_size_mb", 128)
    )

    # 日志配置
    logging = LogConfig(
        get_env("GENOMICPRO_LOG_LEVEL", get(get(data, "logging", Dict()), "log_level", "INFO")),
        get(get(data, "logging", Dict()), "log_file", nothing),
        get(get(data, "logging", Dict()), "console_output", true),
        get(get(data, "logging", Dict()), "performance_logging", false)
    )

    # API 配置
    api = APIConfig(
        get_env("GENOMICPRO_API_HOST", get(get(data, "api", Dict()), "host", "127.0.0.1")),
        get_env_int("GENOMICPRO_API_PORT", get(get(data, "api", Dict()), "port", 8080)),
        get(get(data, "api", Dict()), "cors_enabled", true),
        get(get(data, "api", Dict()), "max_upload_size_mb", 1024),
        get(get(data, "api", Dict()), "session_timeout_minutes", 60)
    )

    # 分析配置
    analysis = AnalysisConfig(
        get(get(data, "analysis", Dict()), "default_model", "gblup"),
        get(get(data, "analysis", Dict()), "cross_validation_folds", 5),
        get(get(data, "analysis", Dict()), "significance_threshold", 5e-8),
        get(get(data, "analysis", Dict()), "max_iterations", 10000)
    )

    return GenomicProConfig(
        compute,
        memory,
        io,
        logging,
        api,
        analysis,
        "2.0.0"
    )
end

# ============================================================================
# 环境变量辅助函数
# ============================================================================

function get_env(name::String, default::String)
    return get(ENV, name, default)
end

function get_env_int(name::String, default::Int)
    val = get(ENV, name, nothing)
    return val === nothing ? default : parse(Int, val)
end

function get_env_float(name::String, default::Float64)
    val = get(ENV, name, nothing)
    return val === nothing ? default : parse(Float64, val)
end

function get_env_bool(name::String, default::Bool)
    val = get(ENV, name, nothing)
    return val === nothing ? default : lowercase(val) in ["true", "1", "yes"]
end

# ============================================================================
# 全局配置
# ============================================================================

const GLOBAL_CONFIG = Ref{Union{GenomicProConfig, Nothing}}(nothing)

"""
    get_config()

获取全局配置。如果未设置，则加载默认配置。
"""
function get_config()
    if GLOBAL_CONFIG[] === nothing
        GLOBAL_CONFIG[] = load_config()
    end
    return GLOBAL_CONFIG[]
end

"""
    set_global_config!(config::GenomicProConfig)

设置全局配置。
"""
function set_global_config!(config::GenomicProConfig)
    GLOBAL_CONFIG[] = config
    @info "全局配置已更新"
end

"""
    reset_config!()

重置全局配置为默认值。
"""
function reset_config!()
    GLOBAL_CONFIG[] = nothing
    @info "全局配置已重置"
end

# ============================================================================
# 配置验证
# ============================================================================

"""
    validate_config(config::GenomicProConfig)

验证配置的有效性。

返回 (is_valid, warnings) 元组。
"""
function validate_config(config::GenomicProConfig)
    warnings = String[]

    # 验证线程数
    if config.compute.threads < 1
        push!(warnings, "线程数必须 >= 1")
    elseif config.compute.threads > Threads.nthreads() * 2
        push!(warnings, "配置的线程数 ($(config.compute.threads)) 远大于可用线程数 ($(Threads.nthreads()))")
    end

    # 验证内存
    if config.memory.max_memory_gb < 1.0
        push!(warnings, "最大内存必须 >= 1 GB")
    end

    # 验证端口
    if config.api.port < 1024 || config.api.port > 65535
        push!(warnings, "API 端口必须在 1024-65535 之间")
    end

    # 验证日志级别
    valid_levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if !(config.logging.level in valid_levels)
        push!(warnings, "无效的日志级别: $(config.logging.level)")
    end

    # 验证目录
    if !isdir(dirname(config.io.temp_dir)) && !isdir(config.io.temp_dir)
        push!(warnings, "临时目录不存在且无法创建: $(config.io.temp_dir)")
    end

    is_valid = isempty(warnings)
    return (is_valid, warnings)
end

# ============================================================================
# 配置导出
# ============================================================================

"""
    save_config(config::GenomicProConfig, filepath::String)

将配置保存为 TOML 文件。
"""
function save_config(config::GenomicProConfig, filepath::String)
    data = Dict(
        "compute" => Dict(
            "threads" => config.compute.threads,
            "use_gpu" => config.compute.use_gpu,
            "gpu_device_id" => config.compute.gpu_device_id,
            "parallel_backend" => String(config.compute.parallel_backend)
        ),
        "memory" => Dict(
            "max_memory_gb" => config.memory.max_memory_gb,
            "chunk_size" => config.memory.chunk_size,
            "use_compression" => config.memory.use_compression,
            "cache_size_mb" => config.memory.cache_size_mb
        ),
        "io" => Dict(
            "temp_dir" => config.io.temp_dir,
            "output_dir" => config.io.output_dir,
            "auto_cleanup" => config.io.auto_cleanup,
            "buffer_size_mb" => config.io.buffer_size_mb
        ),
        "logging" => Dict(
            "log_level" => config.logging.level,
            "log_file" => config.logging.log_file,
            "console_output" => config.logging.console_output,
            "performance_logging" => config.logging.performance_logging
        ),
        "api" => Dict(
            "host" => config.api.host,
            "port" => config.api.port,
            "cors_enabled" => config.api.cors_enabled,
            "max_upload_size_mb" => config.api.max_upload_size_mb,
            "session_timeout_minutes" => config.api.session_timeout_minutes
        ),
        "analysis" => Dict(
            "default_model" => config.analysis.default_model,
            "cross_validation_folds" => config.analysis.cross_validation_folds,
            "significance_threshold" => config.analysis.significance_threshold,
            "max_iterations" => config.analysis.max_iterations
        )
    )

    open(filepath, "w") do io
        TOML.print(io, data)
    end

    @info "配置已保存到: $filepath"
end

"""
    print_config(config::GenomicProConfig)

打印配置摘要。
"""
function print_config(config::GenomicProConfig)
    println("="^60)
    println("GenomicPro2 配置 (v$(config.version))")
    println("="^60)

    println("\n计算配置:")
    println("  线程数: $(config.compute.threads)")
    println("  使用 GPU: $(config.compute.use_gpu)")
    println("  GPU 设备: $(config.compute.gpu_device_id)")
    println("  并行后端: $(config.compute.parallel_backend)")

    println("\n内存配置:")
    println("  最大内存: $(config.memory.max_memory_gb) GB")
    println("  块大小: $(config.memory.chunk_size)")
    println("  压缩: $(config.memory.use_compression)")
    println("  缓存大小: $(config.memory.cache_size_mb) MB")

    println("\nIO 配置:")
    println("  临时目录: $(config.io.temp_dir)")
    println("  输出目录: $(config.io.output_dir)")
    println("  自动清理: $(config.io.auto_cleanup)")

    println("\n日志配置:")
    println("  级别: $(config.logging.level)")
    println("  日志文件: $(config.logging.log_file === nothing ? "无" : config.logging.log_file)")
    println("  控制台输出: $(config.logging.console_output)")
    println("  性能日志: $(config.logging.performance_logging)")

    println("\nAPI 配置:")
    println("  地址: http://$(config.api.host):$(config.api.port)")
    println("  CORS: $(config.api.cors_enabled)")
    println("  最大上传: $(config.api.max_upload_size_mb) MB")

    println("\n分析配置:")
    println("  默认模型: $(config.analysis.default_model)")
    println("  交叉验证折数: $(config.analysis.cross_validation_folds)")
    println("  显著性阈值: $(config.analysis.significance_threshold)")

    println("="^60)
end

# 导出
export GenomicProConfig, ComputeConfig, MemoryConfig, IOConfig, LogConfig, APIConfig, AnalysisConfig
export load_config, save_config, print_config
export get_config, set_global_config!, reset_config!
export validate_config

end # module Config
