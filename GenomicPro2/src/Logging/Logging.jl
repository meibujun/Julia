"""
# 日志框架模块

提供结构化日志系统，支持：
- 多级别日志（DEBUG, INFO, WARN, ERROR）
- 多输出（控制台、文件）
- 性能日志
- 结构化日志数据

## 使用示例
```julia
using GenomicPro2.Logging

# 设置日志
setup_logging(level="INFO", log_file="genomicpro2.log")

# 使用日志
@info "Starting analysis" n_samples=1000 n_snps=50000

# 性能日志
@log_performance "GRM Computation" begin
    grm = compute_grm(genotypes)
end

# 错误日志
@error "Analysis failed" exception=e
```
"""
module Logging

using Dates
using Printf
using ..Config

# ============================================================================
# 日志级别
# ============================================================================

@enum LogLevel DEBUG=1 INFO=2 WARN=3 ERROR=4

function loglevel_from_string(s::String)
    s_upper = uppercase(s)
    if s_upper == "DEBUG"
        return DEBUG
    elseif s_upper == "INFO"
        return INFO
    elseif s_upper == "WARN" || s_upper == "WARNING"
        return WARN
    elseif s_upper == "ERROR"
        return ERROR
    else
        @warn "Unknown log level: $s, using INFO"
        return INFO
    end
end

# ============================================================================
# 日志记录器
# ============================================================================

mutable struct Logger
    level::LogLevel
    outputs::Vector{IO}
    structured::Bool
    performance::Bool
    timestamp_format::String
end

const GLOBAL_LOGGER = Ref{Union{Logger, Nothing}}(nothing)

"""
    setup_logging(; level::String="INFO",
                    log_file::Union{String, Nothing}=nothing,
                    console::Bool=true,
                    structured::Bool=true,
                    performance::Bool=false)

设置日志系统。

# 参数
- `level`: 日志级别（DEBUG, INFO, WARN, ERROR）
- `log_file`: 日志文件路径（如果为 nothing 则不写文件）
- `console`: 是否输出到控制台
- `structured`: 是否使用结构化日志（JSON 格式）
- `performance`: 是否启用性能日志

# 示例
```julia
setup_logging(level="DEBUG", log_file="debug.log")
setup_logging(level="INFO", console=true, performance=true)
```
"""
function setup_logging(; level::String="INFO",
                        log_file::Union{String, Nothing}=nothing,
                        console::Bool=true,
                        structured::Bool=false,
                        performance::Bool=false)

    log_level = loglevel_from_string(level)

    outputs = IO[]

    # 控制台输出
    if console
        push!(outputs, stdout)
    end

    # 文件输出
    if log_file !== nothing
        # 创建目录（如果不存在）
        dir = dirname(log_file)
        if !isempty(dir) && !isdir(dir)
            mkpath(dir)
        end

        file_io = open(log_file, "a")  # 追加模式
        push!(outputs, file_io)

        @info "日志将写入文件: $log_file"
    end

    logger = Logger(
        log_level,
        outputs,
        structured,
        performance,
        "yyyy-mm-dd HH:MM:SS"
    )

    GLOBAL_LOGGER[] = logger

    @info "日志系统已初始化" level=level structured=structured performance=performance
end

"""
    setup_logging_from_config(config::GenomicProConfig)

从配置对象设置日志。
"""
function setup_logging_from_config(config::GenomicProConfig)
    setup_logging(
        level = config.logging.level,
        log_file = config.logging.log_file,
        console = config.logging.console_output,
        performance = config.logging.performance_logging
    )
end

"""
    get_logger()

获取全局日志记录器。如果未初始化，则使用默认设置。
"""
function get_logger()
    if GLOBAL_LOGGER[] === nothing
        setup_logging()  # 使用默认设置
    end
    return GLOBAL_LOGGER[]
end

# ============================================================================
# 日志写入
# ============================================================================

"""
写入日志消息
"""
function write_log(level::LogLevel, message::String; kwargs...)
    logger = get_logger()

    # 检查日志级别
    if level < logger.level
        return
    end

    # 时间戳
    timestamp = Dates.format(now(), logger.timestamp_format)

    # 级别字符串
    level_str = if level == DEBUG
        "DEBUG"
    elseif level == INFO
        "INFO"
    elseif level == WARN
        "WARN"
    else
        "ERROR"
    end

    # 格式化消息
    if logger.structured
        # 结构化日志（类似 JSON）
        fields = ["time=\"$timestamp\"", "level=$level_str", "msg=\"$message\""]

        for (k, v) in kwargs
            push!(fields, "$k=$(repr(v))")
        end

        log_line = "{" * join(fields, ", ") * "}"
    else
        # 传统格式
        log_line = "[$timestamp] $level_str - $message"

        if !isempty(kwargs)
            fields = ["$k=$v" for (k, v) in kwargs]
            log_line *= " | " * join(fields, ", ")
        end
    end

    # 写入所有输出
    for io in logger.outputs
        println(io, log_line)
        flush(io)
    end
end

# ============================================================================
# 宏定义
# ============================================================================

"""
    @debug message [field=value ...]

记录 DEBUG 级别日志。
"""
macro debug(message, kwargs...)
    quote
        write_log(DEBUG, $(esc(message)); $(esc.(kwargs)...))
    end
end

"""
    @info message [field=value ...]

记录 INFO 级别日志。
"""
macro info(message, kwargs...)
    quote
        write_log(INFO, $(esc(message)); $(esc.(kwargs)...))
    end
end

"""
    @warn message [field=value ...]

记录 WARN 级别日志。
"""
macro warn(message, kwargs...)
    quote
        write_log(WARN, $(esc(message)); $(esc.(kwargs)...))
    end
end

"""
    @error message [field=value ...]

记录 ERROR 级别日志。
"""
macro error(message, kwargs...)
    quote
        write_log(ERROR, $(esc(message)); $(esc.(kwargs)...))
    end
end

# ============================================================================
# 性能日志
# ============================================================================

"""
    @log_performance name expr

记录代码块的执行时间。

# 示例
```julia
@log_performance "GRM Computation" begin
    grm = compute_grm(genotypes)
end
```
"""
macro log_performance(name, expr)
    quote
        logger = get_logger()

        if logger.performance
            task_name = $(esc(name))
            @info "Performance task started" task=task_name

            start_time = time()
            start_mem = Base.gc_live_bytes() / 1024^2  # MB

            result = $(esc(expr))

            elapsed = time() - start_time
            end_mem = Base.gc_live_bytes() / 1024^2
            mem_delta = end_mem - start_mem

            @info "Performance task completed" task=task_name time_seconds=elapsed memory_mb=mem_delta

            result
        else
            $(esc(expr))
        end
    end
end

"""
    PerformanceTimer

用于更灵活的性能测量的计时器。

# 示例
```julia
timer = PerformanceTimer("My Task")
start!(timer)

# ... 执行代码 ...

stop!(timer)
log_performance(timer)
```
"""
mutable struct PerformanceTimer
    name::String
    start_time::Union{Float64, Nothing}
    end_time::Union{Float64, Nothing}
    start_memory::Union{Float64, Nothing}
    end_memory::Union{Float64, Nothing}

    PerformanceTimer(name::String) = new(name, nothing, nothing, nothing, nothing)
end

function start!(timer::PerformanceTimer)
    timer.start_time = time()
    timer.start_memory = Base.gc_live_bytes() / 1024^2
    @debug "Timer started" task=timer.name
end

function stop!(timer::PerformanceTimer)
    timer.end_time = time()
    timer.end_memory = Base.gc_live_bytes() / 1024^2
    @debug "Timer stopped" task=timer.name
end

function elapsed(timer::PerformanceTimer)
    if timer.start_time === nothing || timer.end_time === nothing
        return nothing
    end
    return timer.end_time - timer.start_time
end

function memory_delta(timer::PerformanceTimer)
    if timer.start_memory === nothing || timer.end_memory === nothing
        return nothing
    end
    return timer.end_memory - timer.start_memory
end

function log_performance(timer::PerformanceTimer)
    elapsed_time = elapsed(timer)
    mem = memory_delta(timer)

    if elapsed_time !== nothing
        @info "Performance measurement" task=timer.name time_seconds=elapsed_time memory_mb=mem
    else
        @warn "Timer not properly started/stopped" task=timer.name
    end
end

# ============================================================================
# 日志轮转
# ============================================================================

"""
    rotate_log_file(log_file::String; max_size_mb::Int=100, max_files::Int=5)

日志文件轮转。当文件超过指定大小时，重命名并创建新文件。
"""
function rotate_log_file(log_file::String; max_size_mb::Int=100, max_files::Int=5)
    if !isfile(log_file)
        return false
    end

    file_size_mb = filesize(log_file) / 1024^2

    if file_size_mb < max_size_mb
        return false
    end

    @info "Rotating log file" file=log_file size_mb=file_size_mb

    # 删除最旧的日志
    oldest_log = "$log_file.$(max_files)"
    if isfile(oldest_log)
        rm(oldest_log)
    end

    # 轮转现有日志
    for i in (max_files-1):-1:1
        old_name = "$log_file.$i"
        new_name = "$log_file.$(i+1)"

        if isfile(old_name)
            mv(old_name, new_name)
        end
    end

    # 重命名当前日志
    mv(log_file, "$log_file.1")

    @info "Log file rotated" new_file=log_file

    return true
end

# ============================================================================
# 日志过滤和搜索
# ============================================================================

"""
    search_logs(log_file::String, pattern::Regex; max_lines::Int=100)

在日志文件中搜索匹配的行。
"""
function search_logs(log_file::String, pattern::Regex; max_lines::Int=100)
    if !isfile(log_file)
        @warn "Log file not found" file=log_file
        return String[]
    end

    matches = String[]

    open(log_file, "r") do io
        for line in eachline(io)
            if occursin(pattern, line)
                push!(matches, line)

                if length(matches) >= max_lines
                    break
                end
            end
        end
    end

    return matches
end

# ============================================================================
# 清理
# ============================================================================

"""
    close_logger()

关闭日志记录器，释放文件句柄。
"""
function close_logger()
    logger = GLOBAL_LOGGER[]

    if logger !== nothing
        for io in logger.outputs
            if io != stdout && io != stderr
                close(io)
            end
        end

        @info "Logger closed"
        GLOBAL_LOGGER[] = nothing
    end
end

# 导出
export setup_logging, setup_logging_from_config, get_logger, close_logger
export @debug, @info, @warn, @error
export @log_performance, PerformanceTimer, start!, stop!, log_performance
export rotate_log_file, search_logs

end # module Logging
