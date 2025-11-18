# GenomicPro 2.0 可观测性、API设计与安全性

**版本**: 2.0
**日期**: 2025-11-15

---

## 目录

**第一部分：可观测性和监控**
1. [监控指标体系](#1-监控指标体系)
2. [分布式追踪](#2-分布式追踪)
3. [结构化日志](#3-结构化日志)
4. [告警系统](#4-告警系统)

**第二部分：API设计规范**
5. [RESTful API设计](#5-restful-api设计)
6. [版本控制策略](#6-版本控制策略)
7. [文档生成](#7-文档生成)

**第三部分：安全性设计**
8. [认证和授权](#8-认证和授权)
9. [数据安全](#9-数据安全)
10. [隐私保护](#10-隐私保护)

---

# 第一部分：可观测性和监控

## 1. 监控指标体系

### 1.1 指标分类

```julia
using Prometheus

"""
指标注册表
"""
const METRICS_REGISTRY = CollectorRegistry()

# ============================================================================
# 计数器 (Counter) - 只增不减
# ============================================================================

"""
请求计数器
"""
const request_counter = Counter(
    "genomicpro_requests_total",
    "Total number of requests";
    registry=METRICS_REGISTRY,
    labels=[:method, :status, :endpoint]
)

"""
错误计数器
"""
const error_counter = Counter(
    "genomicpro_errors_total",
    "Total number of errors";
    registry=METRICS_REGISTRY,
    labels=[:error_type, :component]
)

# ============================================================================
# 计量器 (Gauge) - 可增可减
# ============================================================================

"""
活跃任务数
"""
const active_tasks = Gauge(
    "genomicpro_active_tasks",
    "Number of active tasks";
    registry=METRICS_REGISTRY,
    labels=[:task_type]
)

"""
内存使用
"""
const memory_usage = Gauge(
    "genomicpro_memory_bytes",
    "Memory usage in bytes";
    registry=METRICS_REGISTRY,
    labels=[:component]
)

"""
GPU 利用率
"""
const gpu_utilization = Gauge(
    "genomicpro_gpu_utilization_percent",
    "GPU utilization percentage";
    registry=METRICS_REGISTRY,
    labels=[:device_id]
)

# ============================================================================
# 直方图 (Histogram) - 分布统计
# ============================================================================

"""
请求延迟
"""
const request_duration = Histogram(
    "genomicpro_request_duration_seconds",
    "Request duration in seconds";
    registry=METRICS_REGISTRY,
    labels=[:method, :endpoint],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

"""
GRM 计算时间
"""
const grm_computation_time = Histogram(
    "genomicpro_grm_computation_seconds",
    "GRM computation time";
    registry=METRICS_REGISTRY,
    labels=[:method, :n_samples, :n_markers],
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

# ============================================================================
# 摘要 (Summary) - 分位数统计
# ============================================================================

"""
模型训练时间
"""
const model_training_time = Summary(
    "genomicpro_model_training_seconds",
    "Model training time";
    registry=METRICS_REGISTRY,
    labels=[:model_type],
    quantiles=[0.5, 0.9, 0.95, 0.99]
)

# ============================================================================
# 指标收集器
# ============================================================================

"""
自动指标收集
"""
struct MetricsCollector
    registry::CollectorRegistry
    update_interval::Float64
    enabled::Bool

    function MetricsCollector(; interval::Float64=60.0, enabled::Bool=true)
        collector = new(METRICS_REGISTRY, interval, enabled)

        if enabled
            # 启动后台收集
            @async collect_system_metrics(collector)
        end

        return collector
    end
end

function collect_system_metrics(collector::MetricsCollector)
    while collector.enabled
        # 系统内存
        set!(memory_usage, Sys.total_memory() - Sys.free_memory(); labels=Dict(:component => "system"))

        # Julia 内存
        set!(memory_usage, Base.gc_live_bytes(); labels=Dict(:component => "julia"))

        # GPU 指标
        if CUDA.functional()
            for (i, dev) in enumerate(CUDA.devices())
                CUDA.device!(dev)
                util = CUDA.utilization(dev)
                set!(gpu_utilization, util; labels=Dict(:device_id => string(i)))
            end
        end

        sleep(collector.update_interval)
    end
end

"""
指标导出器
"""
function export_metrics(collector::MetricsCollector, format::Symbol=:prometheus)
    if format == :prometheus
        return Prometheus.exposition(collector.registry)
    elseif format == :json
        return JSON3.write(collect_all_metrics(collector))
    end
end

# ============================================================================
# 使用示例
# ============================================================================

"""
记录 API 请求
"""
function handle_request_with_metrics(method::String, endpoint::String, handler::Function)
    # 增加活跃任务
    inc!(active_tasks; labels=Dict(:task_type => "api_request"))

    # 测量时间
    duration = @elapsed begin
        try
            # 执行请求
            result = handler()

            # 记录成功
            inc!(request_counter; labels=Dict(
                :method => method,
                :status => "200",
                :endpoint => endpoint
            ))

            status = "success"
            return result

        catch e
            # 记录错误
            inc!(request_counter; labels=Dict(
                :method => method,
                :status => "500",
                :endpoint => endpoint
            ))

            inc!(error_counter; labels=Dict(
                :error_type => string(typeof(e)),
                :component => "api"
            ))

            status = "error"
            rethrow(e)

        finally
            # 减少活跃任务
            dec!(active_tasks; labels=Dict(:task_type => "api_request"))
        end
    end

    # 记录延迟
    observe!(request_duration, duration; labels=Dict(
        :method => method,
        :endpoint => endpoint
    ))
end

"""
记录 GRM 计算
"""
function compute_grm_with_metrics(geno::AbstractGenotypeData; method::Symbol=:vanraden)
    n = n_samples(geno)
    m = n_markers(geno)

    duration = @elapsed G = compute_grm(geno; method=method)

    # 记录指标
    observe!(grm_computation_time, duration; labels=Dict(
        :method => string(method),
        :n_samples => string(n),
        :n_markers => string(m)
    ))

    return G
end
```

---

## 2. 分布式追踪

### 2.1 追踪上下文

```julia
using UUIDs

"""
追踪上下文
"""
struct TraceContext
    trace_id::String
    span_id::String
    parent_span_id::Union{String, Nothing}
    sampled::Bool

    function TraceContext(;
        trace_id::String=generate_trace_id(),
        span_id::String=generate_span_id(),
        parent_span_id::Union{String, Nothing}=nothing,
        sampled::Bool=true
    )
        new(trace_id, span_id, parent_span_id, sampled)
    end
end

generate_trace_id() = string(uuid4())
generate_span_id() = string(uuid4())[1:16]

"""
Span (追踪片段)
"""
mutable struct Span
    trace_context::TraceContext
    operation_name::String
    start_time::Float64
    end_time::Union{Float64, Nothing}
    tags::Dict{String, Any}
    logs::Vector{Tuple{Float64, String}}

    function Span(ctx::TraceContext, operation::String)
        new(
            ctx,
            operation,
            time(),
            nothing,
            Dict{String, Any}(),
            Tuple{Float64, String}[]
        )
    end
end

"""
开始 Span
"""
function start_span(operation::String; parent::Union{Span, Nothing}=nothing)
    ctx = if isnothing(parent)
        TraceContext()
    else
        TraceContext(
            trace_id=parent.trace_context.trace_id,
            parent_span_id=parent.trace_context.span_id
        )
    end

    return Span(ctx, operation)
end

"""
结束 Span
"""
function finish_span!(span::Span)
    span.end_time = time()

    # 报告到追踪后端 (Jaeger/Zipkin)
    report_span(span)
end

"""
添加标签
"""
function set_tag!(span::Span, key::String, value)
    span.tags[key] = value
end

"""
添加日志
"""
function log!(span::Span, message::String)
    push!(span.logs, (time(), message))
end

# ============================================================================
# 追踪装饰器
# ============================================================================

"""
自动追踪函数
"""
macro traced(operation_name, expr)
    quote
        span = start_span($(esc(operation_name)))

        try
            result = $(esc(expr))
            set_tag!(span, "status", "success")
            result
        catch e
            set_tag!(span, "status", "error")
            set_tag!(span, "error.type", string(typeof(e)))
            set_tag!(span, "error.message", string(e))
            rethrow(e)
        finally
            finish_span!(span)
        end
    end
end

# ============================================================================
# 使用示例
# ============================================================================

"""
完整的追踪示例
"""
function predict_with_tracing(genotype_id::String, model_config::Dict)
    # 根 Span
    root_span = start_span("genomic_prediction")
    set_tag!(root_span, "genotype_id", genotype_id)
    set_tag!(root_span, "model_type", model_config[:type])

    try
        # 子 Span: 数据加载
        @traced "load_data" begin
            genotypes = load_genotypes(genotype_id)
            phenotypes = load_phenotypes(genotype_id)
            log!(root_span, "Data loaded successfully")
        end

        # 子 Span: QC
        @traced "quality_control" begin
            genotypes_qc, phenotypes_qc = apply_qc(genotypes, phenotypes)
            log!(root_span, "QC completed")
        end

        # 子 Span: GRM 计算
        @traced "compute_grm" begin
            G = compute_grm(genotypes_qc)
            set_tag!(root_span, "grm.size", size(G, 1))
        end

        # 子 Span: 模型训练
        @traced "train_model" begin
            model = create_model(model_config[:type])
            fit!(model, genotypes_qc, phenotypes_qc; G=G)
        end

        # 子 Span: 预测
        predictions = @traced "predict" begin
            predict(model, genotypes_qc)
        end

        set_tag!(root_span, "status", "success")
        return predictions

    catch e
        set_tag!(root_span, "error", true)
        log!(root_span, "Error: $(string(e))")
        rethrow(e)

    finally
        finish_span!(root_span)
    end
end
```

---

## 3. 结构化日志

### 3.1 结构化日志系统

```julia
using Logging, LoggingExtras, JSON3, Dates

"""
结构化日志记录器
"""
struct StructuredLogger <: AbstractLogger
    min_level::LogLevel
    io::IO
    include_metadata::Bool

    function StructuredLogger(
        io::IO=stderr;
        min_level::LogLevel=Logging.Info,
        include_metadata::Bool=true
    )
        new(min_level, io, include_metadata)
    end
end

function Logging.handle_message(
    logger::StructuredLogger,
    level, message, _module, group, id, file, line;
    kwargs...
)
    if level < logger.min_level
        return
    end

    # 构建结构化日志
    log_entry = Dict{String, Any}(
        "timestamp" => now(),
        "level" => string(level),
        "message" => string(message),
        "logger" => string(_module)
    )

    # 添加元数据
    if logger.include_metadata
        log_entry["file"] = string(file)
        log_entry["line"] = line
    end

    # 添加额外字段
    for (k, v) in kwargs
        log_entry[string(k)] = v
    end

    # 输出 JSON
    println(logger.io, JSON3.write(log_entry))
    flush(logger.io)
end

Logging.min_enabled_level(logger::StructuredLogger) = logger.min_level
Logging.shouldlog(logger::StructuredLogger, level, _module, group, id) = true
Logging.catch_exceptions(logger::StructuredLogger) = false

# ============================================================================
# 日志上下文
# ============================================================================

"""
日志上下文
"""
struct LogContext
    fields::Dict{Symbol, Any}
end

const CURRENT_LOG_CONTEXT = Ref{Union{LogContext, Nothing}}(nothing)

"""
设置日志上下文
"""
function with_log_context(f::Function, fields::Dict{Symbol, Any})
    old_context = CURRENT_LOG_CONTEXT[]
    CURRENT_LOG_CONTEXT[] = LogContext(fields)

    try
        f()
    finally
        CURRENT_LOG_CONTEXT[] = old_context
    end
end

macro with_context(fields, expr)
    quote
        with_log_context($(esc(fields))) do
            $(esc(expr))
        end
    end
end

"""
增强的日志宏
"""
macro log_info(msg, kwargs...)
    quote
        ctx_fields = if !isnothing(CURRENT_LOG_CONTEXT[])
            CURRENT_LOG_CONTEXT[].fields
        else
            Dict{Symbol, Any}()
        end

        @info $(esc(msg)) ctx_fields... $(esc.(kwargs)...)
    end
end

# ============================================================================
# 使用示例
# ============================================================================

# 设置全局日志
global_logger(StructuredLogger(stderr; min_level=Logging.Info))

# 带上下文的日志
@with_context Dict(:user_id => "user123", :request_id => "req456") begin
    @log_info "Starting analysis" dataset_id="data001"
    # 输出: {"timestamp":"2025-11-15T...", "level":"Info", "message":"Starting analysis",
    #        "user_id":"user123", "request_id":"req456", "dataset_id":"data001"}

    @log_info "Analysis complete" duration_seconds=123.45
end
```

---

## 4. 告警系统

### 4.1 告警规则

```julia
"""
告警规则
"""
struct AlertRule
    name::String
    condition::Function  # (metrics) -> Bool
    severity::Symbol     # :critical, :warning, :info
    message::String
    cooldown_seconds::Int
    last_triggered::Ref{Union{DateTime, Nothing}}

    function AlertRule(
        name::String,
        condition::Function,
        severity::Symbol,
        message::String;
        cooldown::Int=300
    )
        new(name, condition, severity, message, cooldown, Ref{Union{DateTime, Nothing}}(nothing))
    end
end

"""
检查并触发告警
"""
function check_alert!(rule::AlertRule, metrics::Dict)
    # 检查冷却期
    if !isnothing(rule.last_triggered[])
        elapsed = (now() - rule.last_triggered[]).value / 1000  # seconds
        if elapsed < rule.cooldown_seconds
            return false
        end
    end

    # 检查条件
    if rule.condition(metrics)
        # 触发告警
        trigger_alert(rule, metrics)
        rule.last_triggered[] = now()
        return true
    end

    return false
end

"""
告警管理器
"""
struct AlertManager
    rules::Vector{AlertRule}
    notifiers::Vector{AlertNotifier}

    function AlertManager()
        new(AlertRule[], AlertNotifier[])
    end
end

function add_rule!(manager::AlertManager, rule::AlertRule)
    push!(manager.rules, rule)
end

function add_notifier!(manager::AlertManager, notifier::AlertNotifier)
    push!(manager.notifiers, notifier)
end

"""
评估所有告警规则
"""
function evaluate_alerts!(manager::AlertManager, metrics::Dict)
    for rule in manager.rules
        if check_alert!(rule, metrics)
            # 通知所有通知器
            for notifier in manager.notifiers
                notify(notifier, rule, metrics)
            end
        end
    end
end

# ============================================================================
# 告警通知器
# ============================================================================

abstract type AlertNotifier end

"""
邮件通知器
"""
struct EmailNotifier <: AlertNotifier
    smtp_server::String
    recipients::Vector{String}
end

function notify(notifier::EmailNotifier, rule::AlertRule, metrics::Dict)
    subject = "[$(uppercase(string(rule.severity)))] $(rule.name)"
    body = """
    Alert: $(rule.name)
    Severity: $(rule.severity)
    Time: $(now())

    Message: $(rule.message)

    Metrics:
    $(JSON3.write(metrics, indent=2))
    """

    # 发送邮件
    send_email(notifier.smtp_server, notifier.recipients, subject, body)
end

"""
Slack 通知器
"""
struct SlackNotifier <: AlertNotifier
    webhook_url::String
    channel::String
end

function notify(notifier::SlackNotifier, rule::AlertRule, metrics::Dict)
    color = if rule.severity == :critical
        "danger"
    elseif rule.severity == :warning
        "warning"
    else
        "good"
    end

    payload = Dict(
        "channel" => notifier.channel,
        "attachments" => [
            Dict(
                "color" => color,
                "title" => rule.name,
                "text" => rule.message,
                "fields" => [
                    Dict("title" => "Severity", "value" => string(rule.severity), "short" => true),
                    Dict("title" => "Time", "value" => string(now()), "short" => true)
                ]
            )
        ]
    )

    HTTP.post(notifier.webhook_url, [], JSON3.write(payload))
end

# ============================================================================
# 预定义告警规则
# ============================================================================

# 高错误率告警
high_error_rate_alert = AlertRule(
    "High Error Rate",
    metrics -> begin
        total = get(metrics, "requests_total", 0)
        errors = get(metrics, "errors_total", 0)
        total > 0 && (errors / total) > 0.05  # 5% 错误率
    end,
    :critical,
    "Error rate exceeds 5%";
    cooldown=300
)

# 内存使用告警
high_memory_alert = AlertRule(
    "High Memory Usage",
    metrics -> begin
        memory_bytes = get(metrics, "memory_bytes", 0)
        total_memory = Sys.total_memory()
        (memory_bytes / total_memory) > 0.9  # 90% 内存使用
    end,
    :warning,
    "Memory usage exceeds 90%";
    cooldown=600
)

# GPU 利用率低告警
low_gpu_utilization_alert = AlertRule(
    "Low GPU Utilization",
    metrics -> begin
        gpu_util = get(metrics, "gpu_utilization_percent", 100.0)
        gpu_util < 20.0  # GPU 利用率低于 20%
    end,
    :info,
    "GPU utilization is below 20%";
    cooldown=1800
)

# 创建告警管理器
alert_manager = AlertManager()
add_rule!(alert_manager, high_error_rate_alert)
add_rule!(alert_manager, high_memory_alert)
add_rule!(alert_manager, low_gpu_utilization_alert)

# 添加通知器
add_notifier!(alert_manager, EmailNotifier("smtp.example.com", ["admin@example.com"]))
add_notifier!(alert_manager, SlackNotifier("https://hooks.slack.com/...", "#alerts"))

# 定期评估
@async begin
    while true
        metrics = collect_current_metrics()
        evaluate_alerts!(alert_manager, metrics)
        sleep(60)  # 每分钟检查一次
    end
end
```

---

# 第二部分：API设计规范

## 5. RESTful API设计

### 5.1 API 路由设计

```julia
using HTTP, JSON3

"""
API 路由
"""
const API_ROUTES = Dict{String, Function}()

"""
注册路由
"""
macro route(method, path, handler)
    quote
        API_ROUTES[$(string(method)) * " " * $(esc(path))] = $(esc(handler))
    end
end

# ============================================================================
# 基因型数据 API
# ============================================================================

"""
GET /api/v2/genotypes
列出所有基因型数据集
"""
@route GET "/api/v2/genotypes" function list_genotypes(req::HTTP.Request)
    # 分页参数
    params = HTTP.queryparams(HTTP.URI(req.target))
    page = parse(Int, get(params, "page", "1"))
    per_page = parse(Int, get(params, "per_page", "20"))

    # 过滤参数
    filters = Dict(
        :min_samples => get(params, "min_samples", nothing),
        :min_markers => get(params, "min_markers", nothing)
    )

    # 查询数据库
    genotypes, total = query_genotypes(page, per_page, filters)

    # 构建响应
    response = Dict(
        "data" => [serialize_genotype(g) for g in genotypes],
        "pagination" => Dict(
            "page" => page,
            "per_page" => per_page,
            "total" => total,
            "pages" => cld(total, per_page)
        ),
        "links" => Dict(
            "self" => "/api/v2/genotypes?page=$page",
            "next" => page < cld(total, per_page) ? "/api/v2/genotypes?page=$(page+1)" : nothing,
            "prev" => page > 1 ? "/api/v2/genotypes?page=$(page-1)" : nothing
        )
    )

    return HTTP.Response(200, JSON3.write(response))
end

"""
POST /api/v2/genotypes
上传新的基因型数据
"""
@route POST "/api/v2/genotypes" function create_genotype(req::HTTP.Request)
    # 解析请求体
    body = JSON3.read(req.body)

    # 验证输入
    validation_result = validate_genotype_upload(body)
    if !validation_result.valid
        return HTTP.Response(400, JSON3.write(Dict(
            "error" => "Validation failed",
            "details" => validation_result.errors
        )))
    end

    # 创建数据集
    genotype_id = create_genotype_dataset(body)

    # 返回 201 Created
    response = Dict(
        "id" => genotype_id,
        "message" => "Genotype dataset created successfully",
        "links" => Dict(
            "self" => "/api/v2/genotypes/$genotype_id"
        )
    )

    return HTTP.Response(201, JSON3.write(response))
end

"""
GET /api/v2/genotypes/:id
获取特定基因型数据集
"""
@route GET "/api/v2/genotypes/:id" function get_genotype(req::HTTP.Request, id::String)
    genotype = find_genotype(id)

    if isnothing(genotype)
        return HTTP.Response(404, JSON3.write(Dict(
            "error" => "Genotype dataset not found",
            "id" => id
        )))
    end

    response = serialize_genotype(genotype)
    return HTTP.Response(200, JSON3.write(response))
end

# ============================================================================
# 分析任务 API
# ============================================================================

"""
POST /api/v2/analyses
创建新的分析任务
"""
@route POST "/api/v2/analyses" function create_analysis(req::HTTP.Request)
    body = JSON3.read(req.body)

    # 创建异步任务
    task_id = create_analysis_task(
        genotype_id = body["genotype_id"],
        phenotype_id = body["phenotype_id"],
        model_type = body["model_type"],
        config = body["config"]
    )

    # 返回 202 Accepted
    response = Dict(
        "task_id" => task_id,
        "status" => "queued",
        "links" => Dict(
            "self" => "/api/v2/analyses/$task_id",
            "status" => "/api/v2/analyses/$task_id/status"
        )
    )

    return HTTP.Response(202, JSON3.write(response))
end

"""
GET /api/v2/analyses/:id/status
查询分析状态
"""
@route GET "/api/v2/analyses/:id/status" function get_analysis_status(req::HTTP.Request, id::String)
    status = query_task_status(id)

    if isnothing(status)
        return HTTP.Response(404, JSON3.write(Dict("error" => "Task not found")))
    end

    response = Dict(
        "task_id" => id,
        "status" => status.state,  # queued, running, completed, failed
        "progress" => status.progress,  # 0-100
        "started_at" => status.started_at,
        "completed_at" => status.completed_at,
        "error" => status.error
    )

    return HTTP.Response(200, JSON3.write(response))
end

"""
GET /api/v2/analyses/:id/results
获取分析结果
"""
@route GET "/api/v2/analyses/:id/results" function get_analysis_results(req::HTTP.Request, id::String)
    task = find_task(id)

    if isnothing(task)
        return HTTP.Response(404, JSON3.write(Dict("error" => "Task not found")))
    end

    if task.status != :completed
        return HTTP.Response(409, JSON3.write(Dict(
            "error" => "Task not completed",
            "status" => string(task.status)
        )))
    end

    # 返回结果
    results = load_results(task.result_path)

    return HTTP.Response(200, JSON3.write(results))
end

# ============================================================================
# API 中间件
# ============================================================================

"""
认证中间件
"""
function auth_middleware(handler::Function)
    return function(req::HTTP.Request)
        # 检查 Authorization header
        auth_header = HTTP.header(req, "Authorization", "")

        if isempty(auth_header)
            return HTTP.Response(401, JSON3.write(Dict(
                "error" => "Missing authorization header"
            )))
        end

        # 验证 token
        token = replace(auth_header, "Bearer " => "")
        user = verify_token(token)

        if isnothing(user)
            return HTTP.Response(401, JSON3.write(Dict(
                "error" => "Invalid or expired token"
            )))
        end

        # 将用户信息添加到请求
        req.context = Dict(:user => user)

        # 调用处理器
        return handler(req)
    end
end

"""
速率限制中间件
"""
function rate_limit_middleware(handler::Function; max_requests::Int=100, window_seconds::Int=60)
    request_counts = Dict{String, Vector{Float64}}()

    return function(req::HTTP.Request)
        # 获取客户端 IP
        client_ip = HTTP.header(req, "X-Forwarded-For", req.context[:remote_addr])

        # 检查速率
        now_time = time()
        if !haskey(request_counts, client_ip)
            request_counts[client_ip] = Float64[]
        end

        # 清除过期请求
        filter!(t -> now_time - t < window_seconds, request_counts[client_ip])

        # 检查限制
        if length(request_counts[client_ip]) >= max_requests
            return HTTP.Response(429, JSON3.write(Dict(
                "error" => "Rate limit exceeded",
                "retry_after" => window_seconds
            )))
        end

        # 记录请求
        push!(request_counts[client_ip], now_time)

        # 调用处理器
        return handler(req)
    end
end

"""
CORS 中间件
"""
function cors_middleware(handler::Function)
    return function(req::HTTP.Request)
        response = handler(req)

        # 添加 CORS headers
        HTTP.setheader(response, "Access-Control-Allow-Origin" => "*")
        HTTP.setheader(response, "Access-Control-Allow-Methods" => "GET, POST, PUT, DELETE, OPTIONS")
        HTTP.setheader(response, "Access-Control-Allow-Headers" => "Content-Type, Authorization")

        return response
    end
end

# ============================================================================
# API 服务器
# ============================================================================

"""
启动 API 服务器
"""
function start_api_server(; port::Int=8080)
    # 应用中间件
    router = cors_middleware(
        rate_limit_middleware(
            auth_middleware(route_handler)
        )
    )

    # 启动服务器
    HTTP.serve(router, "0.0.0.0", port)
    @info "API server started on port $port"
end
```

---

## 6. 版本控制策略

### 6.1 API 版本化

```julia
"""
API 版本管理
"""
struct APIVersion
    major::Int
    minor::Int

    APIVersion(major::Int, minor::Int=0) = new(major, minor)
end

Base.string(v::APIVersion) = "v$(v.major).$(v.minor)"

const CURRENT_VERSION = APIVersion(2, 0)
const SUPPORTED_VERSIONS = [APIVersion(1, 0), APIVersion(2, 0)]

"""
版本协商
"""
function negotiate_version(req::HTTP.Request)::APIVersion
    # 从 URL 路径提取版本
    path_match = match(r"/api/v(\d+)(?:\.(\d+))?/", req.target)

    if !isnothing(path_match)
        major = parse(Int, path_match.captures[1])
        minor = isnothing(path_match.captures[2]) ? 0 : parse(Int, path_match.captures[2])
        requested = APIVersion(major, minor)

        if requested in SUPPORTED_VERSIONS
            return requested
        end
    end

    # 从 Accept header 提取
    accept = HTTP.header(req, "Accept", "")
    if occursin(r"application/vnd\.genomicpro\.v(\d+)", accept)
        # ...
    end

    # 默认返回当前版本
    return CURRENT_VERSION
end

"""
版本特定的处理器
"""
function version_specific_handler(v1_handler::Function, v2_handler::Function)
    return function(req::HTTP.Request)
        version = negotiate_version(req)

        if version.major == 1
            return v1_handler(req)
        elseif version.major == 2
            return v2_handler(req)
        else
            return HTTP.Response(400, "Unsupported API version")
        end
    end
end
```

---

## 7. 文档生成

### 7.1 OpenAPI/Swagger 规范

```julia
"""
OpenAPI 规范生成
"""
function generate_openapi_spec()
    spec = Dict(
        "openapi" => "3.0.0",
        "info" => Dict(
            "title" => "GenomicPro API",
            "version" => "2.0.0",
            "description" => "High-performance genomic prediction API"
        ),
        "servers" => [
            Dict("url" => "https://api.genomicpro.io/v2")
        ],
        "paths" => Dict(),
        "components" => Dict(
            "schemas" => Dict(),
            "securitySchemes" => Dict(
                "BearerAuth" => Dict(
                    "type" => "http",
                    "scheme" => "bearer",
                    "bearerFormat" => "JWT"
                )
            )
        )
    )

    # 定义 schemas
    spec["components"]["schemas"]["Genotype"] = Dict(
        "type" => "object",
        "properties" => Dict(
            "id" => Dict("type" => "string"),
            "n_samples" => Dict("type" => "integer"),
            "n_markers" => Dict("type" => "integer"),
            "created_at" => Dict("type" => "string", "format" => "date-time")
        )
    )

    # 定义路径
    spec["paths"]["/genotypes"] = Dict(
        "get" => Dict(
            "summary" => "List all genotype datasets",
            "parameters" => [
                Dict("name" => "page", "in" => "query", "schema" => Dict("type" => "integer")),
                Dict("name" => "per_page", "in" => "query", "schema" => Dict("type" => "integer"))
            ],
            "responses" => Dict(
                "200" => Dict(
                    "description" => "Successful response",
                    "content" => Dict(
                        "application/json" => Dict(
                            "schema" => Dict(
                                "type" => "object",
                                "properties" => Dict(
                                    "data" => Dict(
                                        "type" => "array",
                                        "items" => Dict("\$ref" => "#/components/schemas/Genotype")
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return spec
end

# 导出为 JSON
openapi_json = JSON3.write(generate_openapi_spec(), indent=2)
write("openapi.json", openapi_json)
```

---

# 第三部分：安全性设计

## 8. 认证和授权

### 8.1 JWT 认证

```julia
using JWTs

"""
JWT 管理器
"""
struct JWTManager
    secret::String
    issuer::String
    expiration_hours::Int

    function JWTManager(secret::String; issuer::String="genomicpro", expiration::Int=24)
        new(secret, issuer, expiration)
    end
end

"""
生成 JWT
"""
function generate_token(manager::JWTManager, user_id::String, roles::Vector{String})
    claims = Dict(
        "sub" => user_id,
        "iss" => manager.issuer,
        "iat" => time(),
        "exp" => time() + manager.expiration_hours * 3600,
        "roles" => roles
    )

    return JWT.encode(HS256(manager.secret), claims)
end

"""
验证 JWT
"""
function verify_token(manager::JWTManager, token::String)
    try
        claims = JWT.decode(HS256(manager.secret), token)

        # 检查过期
        if claims["exp"] < time()
            return nothing
        end

        return claims
    catch e
        @error "Token verification failed" exception=e
        return nothing
    end
end

# ============================================================================
# 基于角色的访问控制 (RBAC)
# ============================================================================

"""
权限检查
"""
function has_permission(user_claims::Dict, required_role::String)::Bool
    user_roles = get(user_claims, "roles", String[])
    return required_role in user_roles
end

"""
权限装饰器
"""
macro require_role(role, expr)
    quote
        user = get(req.context, :user, nothing)

        if isnothing(user)
            return HTTP.Response(401, "Unauthorized")
        end

        if !has_permission(user, $(esc(role)))
            return HTTP.Response(403, "Forbidden")
        end

        $(esc(expr))
    end
end

# 使用示例
@route DELETE "/api/v2/genotypes/:id" function delete_genotype(req::HTTP.Request, id::String)
    @require_role "admin" begin
        delete_genotype_dataset(id)
        return HTTP.Response(204)
    end
end
```

---

## 9. 数据安全

### 9.1 数据加密

```julia
using LibSodium

"""
数据加密管理器
"""
struct EncryptionManager
    key::Vector{UInt8}

    function EncryptionManager(key::Union{Vector{UInt8}, Nothing}=nothing)
        if isnothing(key)
            key = LibSodium.randombytes(LibSodium.crypto_secretbox_KEYBYTES)
        end
        new(key)
    end
end

"""
加密数据
"""
function encrypt(manager::EncryptionManager, data::Vector{UInt8})::Vector{UInt8}
    nonce = LibSodium.randombytes(LibSodium.crypto_secretbox_NONCEBYTES)
    ciphertext = LibSodium.crypto_secretbox(data, nonce, manager.key)
    return vcat(nonce, ciphertext)
end

"""
解密数据
"""
function decrypt(manager::EncryptionManager, encrypted::Vector{UInt8})::Vector{UInt8}
    nonce = encrypted[1:LibSodium.crypto_secretbox_NONCEBYTES]
    ciphertext = encrypted[(LibSodium.crypto_secretbox_NONCEBYTES+1):end]
    return LibSodium.crypto_secretbox_open(ciphertext, nonce, manager.key)
end

# ============================================================================
# 静态数据加密
# ============================================================================

"""
加密的基因型存储
"""
struct EncryptedGenotypeRepository <: IGenotypeRepository
    base_path::String
    encryption_manager::EncryptionManager
end

function save(repo::EncryptedGenotypeRepository, data::CompactGenotypes, id::String)
    # 序列化
    serialized = serialize_to_bytes(data)

    # 加密
    encrypted = encrypt(repo.encryption_manager, serialized)

    # 写入文件
    filepath = joinpath(repo.base_path, "$id.enc")
    write(filepath, encrypted)

    @info "Encrypted genotype saved" id=id size=length(encrypted)
end

function load(repo::EncryptedGenotypeRepository, id::String)::CompactGenotypes
    filepath = joinpath(repo.base_path, "$id.enc")

    # 读取
    encrypted = read(filepath)

    # 解密
    decrypted = decrypt(repo.encryption_manager, encrypted)

    # 反序列化
    return deserialize_from_bytes(decrypted)
end
```

---

## 10. 隐私保护

### 10.1 差分隐私

```julia
"""
差分隐私机制
"""
struct DifferentialPrivacy
    ε::Float64  # 隐私预算
    δ::Float64  # 失败概率

    function DifferentialPrivacy(; ε::Float64=1.0, δ::Float64=1e-5)
        @assert ε > 0 "ε must be positive"
        @assert 0 ≤ δ < 1 "δ must be in [0, 1)"
        new(ε, δ)
    end
end

"""
拉普拉斯机制
"""
function laplace_mechanism(dp::DifferentialPrivacy, value::Float64, sensitivity::Float64)::Float64
    scale = sensitivity / dp.ε
    noise = rand(Laplace(0, scale))
    return value + noise
end

"""
高斯机制
"""
function gaussian_mechanism(dp::DifferentialPrivacy, value::Float64, sensitivity::Float64)::Float64
    σ = sensitivity * sqrt(2 * log(1.25 / dp.δ)) / dp.ε
    noise = randn() * σ
    return value + noise
end

"""
私有化等位基因频率
"""
function privatize_allele_frequencies(
    freqs::Vector{Float64},
    dp::DifferentialPrivacy
)::Vector{Float64}
    # 敏感度：单个样本对频率的最大影响
    n = 1000  # 假设样本数
    sensitivity = 1.0 / n

    return [laplace_mechanism(dp, f, sensitivity) for f in freqs]
end

"""
联邦学习支持
"""
struct FederatedLearningClient
    data::AbstractGenotypeData
    dp::DifferentialPrivacy
end

function compute_local_gradient(client::FederatedLearningClient, model::AbstractGenomicModel)
    # 计算本地梯度
    gradient = compute_gradient(model, client.data)

    # 添加噪声
    privatized_gradient = [
        gaussian_mechanism(client.dp, g, 1.0)  # 假设敏感度为 1
        for g in gradient
    ]

    return privatized_gradient
end
```

---

## 总结

本文档涵盖了 GenomicPro 2.0 的**可观测性、API设计和安全性**三大关键领域：

### 可观测性
- ✅ 完整的指标体系 (Prometheus)
- ✅ 分布式追踪 (Span-based)
- ✅ 结构化日志 (JSON)
- ✅ 智能告警系统

### API设计
- ✅ RESTful 设计规范
- ✅ 版本控制策略
- ✅ OpenAPI/Swagger 文档
- ✅ 中间件架构

### 安全性
- ✅ JWT 认证
- ✅ RBAC 授权
- ✅ 数据加密
- ✅ 差分隐私

这些设计确保了 GenomicPro 2.0 达到**企业级/研究级**标准。

---

**文档版本**: 1.0
**最后更新**: 2025-11-15
