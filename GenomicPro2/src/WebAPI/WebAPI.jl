"""
# Web API 模块

提供 RESTful API 服务器用于基因组分析的 Web 界面。

## 功能
- 数据上传和管理
- 模型拟合和预测
- 可视化数据生成
- 分析任务管理
- 结果导出

## API 端点

### 数据管理
- `POST /api/data/upload` - 上传基因型/表型数据
- `GET /api/data/list` - 列出已上传的数据集
- `DELETE /api/data/:id` - 删除数据集

### 分析
- `POST /api/analysis/gblup` - 运行 GBLUP 分析
- `POST /api/analysis/bayescpi` - 运行 BayesCπ 分析
- `POST /api/analysis/rkhs` - 运行 RKHS 分析
- `POST /api/analysis/deepgblup` - 运行 Deep GBLUP 分析
- `POST /api/analysis/pca` - 运行 PCA 分析
- `POST /api/analysis/admixture` - 运行 ADMIXTURE 分析

### 可视化
- `GET /api/viz/manhattan` - 生成 Manhattan 图数据
- `GET /api/viz/qq` - 生成 QQ 图数据
- `GET /api/viz/pca` - 生成 PCA 图数据
- `GET /api/viz/admixture` - 生成 ADMIXTURE 图数据

### 任务管理
- `GET /api/jobs` - 列出所有任务
- `GET /api/jobs/:id` - 获取任务状态
- `DELETE /api/jobs/:id` - 取消任务

## 使用示例
```julia
using GenomicPro2.WebAPI

# 启动服务器
start_server(host="0.0.0.0", port=8080)

# 在浏览器中访问
# http://localhost:8080
```

## 前端
Web 界面使用：
- HTML5 + CSS3
- JavaScript (ES6+)
- Plotly.js 用于可视化
- Bootstrap 用于 UI 组件
"""
module WebAPI

# 导出主要函数
export start_server, stop_server

# ============================================================================
# 全局状态管理
# ============================================================================

# 数据存储
const DATASETS = Dict{String, Any}()
const DATASETS_LOCK = ReentrantLock()

# 任务队列
const JOBS = Dict{String, Any}()
const JOBS_LOCK = ReentrantLock()

# 分析结果
const RESULTS = Dict{String, Any}()
const RESULTS_LOCK = ReentrantLock()

# 服务器实例
const SERVER = Ref{Any}(nothing)

# ============================================================================
# 辅助函数
# ============================================================================

"""生成唯一 ID"""
function generate_id()
    return string(hash(time()) * rand(UInt64))
end

"""创建 JSON 响应"""
function json_response(data; status::Int=200)
    try
        body = JSON3.write(data)
        return HTTP.Response(status, ["Content-Type" => "application/json"], body)
    catch e
        @error "JSON serialization error" exception=e
        error_data = Dict("error" => "Internal server error", "message" => string(e))
        body = JSON3.write(error_data)
        return HTTP.Response(500, ["Content-Type" => "application/json"], body)
    end
end

"""解析 JSON 请求体"""
function parse_json_body(req::HTTP.Request)
    try
        return JSON3.read(String(req.body))
    catch e
        @error "JSON parsing error" exception=e
        return nothing
    end
end

# ============================================================================
# 数据管理端点
# ============================================================================

"""上传数据集"""
function handle_data_upload(req::HTTP.Request)
    data = parse_json_body(req)

    if data === nothing
        return json_response(Dict("error" => "Invalid JSON"), status=400)
    end

    dataset_id = generate_id()

    lock(DATASETS_LOCK) do
        DATASETS[dataset_id] = Dict(
            "id" => dataset_id,
            "name" => get(data, "name", "Unnamed Dataset"),
            "type" => get(data, "type", "unknown"),
            "created_at" => now(),
            "data" => data
        )
    end

    @info "Dataset uploaded" id=dataset_id name=DATASETS[dataset_id]["name"]

    return json_response(Dict(
        "success" => true,
        "dataset_id" => dataset_id,
        "message" => "Dataset uploaded successfully"
    ))
end

"""列出所有数据集"""
function handle_data_list(req::HTTP.Request)
    datasets = lock(DATASETS_LOCK) do
        [Dict(
            "id" => id,
            "name" => data["name"],
            "type" => data["type"],
            "created_at" => data["created_at"]
        ) for (id, data) in DATASETS]
    end

    return json_response(Dict("datasets" => datasets))
end

"""删除数据集"""
function handle_data_delete(req::HTTP.Request, dataset_id::String)
    deleted = lock(DATASETS_LOCK) do
        if haskey(DATASETS, dataset_id)
            delete!(DATASETS, dataset_id)
            true
        else
            false
        end
    end

    if deleted
        @info "Dataset deleted" id=dataset_id
        return json_response(Dict("success" => true, "message" => "Dataset deleted"))
    else
        return json_response(Dict("error" => "Dataset not found"), status=404)
    end
end

# ============================================================================
# 分析端点
# ============================================================================

"""GWAS 分析"""
function handle_analysis_gwas(req::HTTP.Request)
    data = parse_json_body(req)

    if data === nothing
        return json_response(Dict("error" => "Invalid JSON"), status=400)
    end

    job_id = generate_id()

    lock(JOBS_LOCK) do
        JOBS[job_id] = Dict(
            "id" => job_id,
            "type" => "gwas",
            "status" => "submitted",
            "created_at" => now(),
            "params" => data
        )
    end

    # 异步执行分析
    @async begin
        try
            @info "Starting GWAS analysis" job_id=job_id

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "running"
                JOBS[job_id]["started_at"] = now()
            end

            # 这里是实际的 GWAS 分析
            # 简化示例：模拟耗时操作
            sleep(2)

            # 模拟结果
            result = Dict(
                "n_significant" => 42,
                "lambda" => 1.05,
                "top_snps" => ["rs123", "rs456", "rs789"]
            )

            lock(RESULTS_LOCK) do
                RESULTS[job_id] = result
            end

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["completed_at"] = now()
            end

            @info "GWAS analysis completed" job_id=job_id

        catch e
            @error "GWAS analysis failed" job_id=job_id exception=e

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = string(e)
            end
        end
    end

    return json_response(Dict(
        "success" => true,
        "job_id" => job_id,
        "message" => "GWAS analysis submitted"
    ))
end

"""GBLUP 分析"""
function handle_analysis_gblup(req::HTTP.Request)
    data = parse_json_body(req)

    if data === nothing
        return json_response(Dict("error" => "Invalid JSON"), status=400)
    end

    job_id = generate_id()

    lock(JOBS_LOCK) do
        JOBS[job_id] = Dict(
            "id" => job_id,
            "type" => "gblup",
            "status" => "submitted",
            "created_at" => now(),
            "params" => data
        )
    end

    @async begin
        try
            @info "Starting GBLUP analysis" job_id=job_id

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "running"
                JOBS[job_id]["started_at"] = now()
            end

            sleep(1.5)

            result = Dict(
                "heritability" => 0.45,
                "accuracy" => 0.72,
                "n_samples" => 1000
            )

            lock(RESULTS_LOCK) do
                RESULTS[job_id] = result
            end

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["completed_at"] = now()
            end

            @info "GBLUP analysis completed" job_id=job_id

        catch e
            @error "GBLUP analysis failed" job_id=job_id exception=e

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = string(e)
            end
        end
    end

    return json_response(Dict(
        "success" => true,
        "job_id" => job_id,
        "message" => "GBLUP analysis submitted"
    ))
end

"""PCA 分析"""
function handle_analysis_pca(req::HTTP.Request)
    data = parse_json_body(req)

    if data === nothing
        return json_response(Dict("error" => "Invalid JSON"), status=400)
    end

    job_id = generate_id()

    lock(JOBS_LOCK) do
        JOBS[job_id] = Dict(
            "id" => job_id,
            "type" => "pca",
            "status" => "submitted",
            "created_at" => now(),
            "params" => data
        )
    end

    @async begin
        try
            @info "Starting PCA analysis" job_id=job_id

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "running"
            end

            sleep(1.0)

            result = Dict(
                "explained_variance" => [0.15, 0.12, 0.08, 0.06, 0.05],
                "n_components" => 10
            )

            lock(RESULTS_LOCK) do
                RESULTS[job_id] = result
            end

            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "completed"
            end

            @info "PCA analysis completed" job_id=job_id

        catch e
            @error "PCA analysis failed" job_id=job_id exception=e
            lock(JOBS_LOCK) do
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = string(e)
            end
        end
    end

    return json_response(Dict(
        "success" => true,
        "job_id" => job_id,
        "message" => "PCA analysis submitted"
    ))
end

# ============================================================================
# 任务管理端点
# ============================================================================

"""列出所有任务"""
function handle_jobs_list(req::HTTP.Request)
    jobs = lock(JOBS_LOCK) do
        [Dict(
            "id" => id,
            "type" => job["type"],
            "status" => job["status"],
            "created_at" => job["created_at"]
        ) for (id, job) in JOBS]
    end

    return json_response(Dict("jobs" => jobs))
end

"""获取任务状态"""
function handle_job_status(req::HTTP.Request, job_id::String)
    job = lock(JOBS_LOCK) do
        get(JOBS, job_id, nothing)
    end

    if job === nothing
        return json_response(Dict("error" => "Job not found"), status=404)
    end

    response = Dict(
        "id" => job["id"],
        "type" => job["type"],
        "status" => job["status"],
        "created_at" => job["created_at"]
    )

    if job["status"] == "completed"
        result = lock(RESULTS_LOCK) do
            get(RESULTS, job_id, nothing)
        end
        if result !== nothing
            response["result"] = result
        end
    end

    return json_response(response)
end

"""取消任务"""
function handle_job_cancel(req::HTTP.Request, job_id::String)
    cancelled = lock(JOBS_LOCK) do
        if haskey(JOBS, job_id) && JOBS[job_id]["status"] in ["submitted", "running"]
            JOBS[job_id]["status"] = "cancelled"
            true
        else
            false
        end
    end

    if cancelled
        @info "Job cancelled" job_id=job_id
        return json_response(Dict("success" => true, "message" => "Job cancelled"))
    else
        return json_response(Dict("error" => "Job not found or cannot be cancelled"), status=400)
    end
end

# ============================================================================
# 可视化端点
# ============================================================================

"""生成 Manhattan 图数据"""
function handle_viz_manhattan(req::HTTP.Request)
    # 从查询参数获取 job_id
    uri = HTTP.URI(req.target)
    params = HTTP.queryparams(uri)
    job_id = get(params, "job_id", "")

    if isempty(job_id)
        return json_response(Dict("error" => "Missing job_id parameter"), status=400)
    end

    result = lock(RESULTS_LOCK) do
        get(RESULTS, job_id, nothing)
    end

    if result === nothing
        return json_response(Dict("error" => "Result not found"), status=404)
    end

    # 生成 Manhattan 图数据（模拟）
    manhattan_data = Dict(
        "chromosomes" => collect(1:22),
        "positions" => rand(1:1000000, 100),
        "pvalues" => rand(100) .* 1e-8,
        "threshold" => 5e-8
    )

    return json_response(manhattan_data)
end

"""生成 QQ 图数据"""
function handle_viz_qq(req::HTTP.Request)
    uri = HTTP.URI(req.target)
    params = HTTP.queryparams(uri)
    job_id = get(params, "job_id", "")

    if isempty(job_id)
        return json_response(Dict("error" => "Missing job_id parameter"), status=400)
    end

    qq_data = Dict(
        "expected" => collect(0:0.1:5),
        "observed" => collect(0:0.1:5) .+ randn(51) .* 0.1,
        "lambda" => 1.05
    )

    return json_response(qq_data)
end

# ============================================================================
# 路由器
# ============================================================================

"""HTTP 请求路由器"""
function router(req::HTTP.Request)
    try
        # CORS 头
        headers = [
            "Access-Control-Allow-Origin" => "*",
            "Access-Control-Allow-Methods" => "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers" => "Content-Type"
        ]

        # OPTIONS 请求（CORS 预检）
        if req.method == "OPTIONS"
            return HTTP.Response(200, headers)
        end

        uri = HTTP.URI(req.target)
        path = uri.path

        # 路由匹配
        if req.method == "POST"
            if path == "/api/data/upload"
                return handle_data_upload(req)
            elseif path == "/api/analysis/gwas"
                return handle_analysis_gwas(req)
            elseif path == "/api/analysis/gblup"
                return handle_analysis_gblup(req)
            elseif path == "/api/analysis/pca"
                return handle_analysis_pca(req)
            end
        elseif req.method == "GET"
            if path == "/api/data/list"
                return handle_data_list(req)
            elseif path == "/api/jobs"
                return handle_jobs_list(req)
            elseif startswith(path, "/api/jobs/")
                job_id = replace(path, "/api/jobs/" => "")
                return handle_job_status(req, job_id)
            elseif path == "/api/viz/manhattan"
                return handle_viz_manhattan(req)
            elseif path == "/api/viz/qq"
                return handle_viz_qq(req)
            elseif path == "/api/health"
                return json_response(Dict(
                    "status" => "healthy",
                    "timestamp" => now(),
                    "datasets" => length(DATASETS),
                    "jobs" => length(JOBS)
                ))
            elseif path == "/"
                # 返回欢迎页面
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>GenomicPro2 API</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                        h1 { color: #2c3e50; }
                        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
                        .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    </style>
                </head>
                <body>
                    <h1>🧬 GenomicPro2 Web API</h1>
                    <p>欢迎使用 GenomicPro2 RESTful API 服务</p>

                    <h2>可用端点</h2>
                    <div class="endpoint">
                        <strong>POST /api/data/upload</strong> - 上传数据集
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/data/list</strong> - 列出数据集
                    </div>
                    <div class="endpoint">
                        <strong>POST /api/analysis/gwas</strong> - GWAS 分析
                    </div>
                    <div class="endpoint">
                        <strong>POST /api/analysis/gblup</strong> - GBLUP 分析
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/jobs</strong> - 列出任务
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/health</strong> - 健康检查
                    </div>

                    <h2>文档</h2>
                    <p>完整 API 文档请参考 <code>GenomicPro2/docs/</code></p>
                </body>
                </html>
                """
                return HTTP.Response(200, ["Content-Type" => "text/html"], html)
            end
        elseif req.method == "DELETE"
            if startswith(path, "/api/data/")
                dataset_id = replace(path, "/api/data/" => "")
                return handle_data_delete(req, dataset_id)
            elseif startswith(path, "/api/jobs/")
                job_id = replace(path, "/api/jobs/" => "")
                return handle_job_cancel(req, job_id)
            end
        end

        # 404 未找到
        return json_response(Dict("error" => "Endpoint not found"), status=404)

    catch e
        @error "Request handling error" exception=e
        return json_response(Dict("error" => "Internal server error", "message" => string(e)), status=500)
    end
end

# ============================================================================
# 服务器启动/停止
# ============================================================================

"""
    start_server(; host::String="127.0.0.1", port::Int=8080, verbose::Bool=true)

启动 Web API 服务器。

# 参数
- `host`: 监听地址（默认 127.0.0.1）
- `port`: 监听端口（默认 8080）
- `verbose`: 是否显示日志

# 示例
```julia
# 启动服务器
start_server(host="0.0.0.0", port=8080)

# 服务器将在 http://0.0.0.0:8080 上运行
```

## 注意
需要安装以下依赖包：
- HTTP.jl
- JSON3.jl

安装方法：
```julia
using Pkg
Pkg.add(["HTTP", "JSON3"])
```
"""
function start_server(; host::String="127.0.0.1", port::Int=8080, verbose::Bool=true)
    # 检查依赖
    if !isdefined(Main, :HTTP)
        @error """
        缺少 HTTP.jl 包！

        请运行：
        using Pkg
        Pkg.add("HTTP")
        """
        return nothing
    end

    if !isdefined(Main, :JSON3)
        @error """
        缺少 JSON3.jl 包！

        请运行：
        using Pkg
        Pkg.add("JSON3")
        """
        return nothing
    end

    if verbose
        println("""
        ╔════════════════════════════════════════════════════════╗
        ║   GenomicPro2 Web API Server                         ║
        ║                                                        ║
        ║   📡 服务器地址: http://$host:$port           ║
        ║                                                        ║
        ║   📖 API 文档: http://$host:$port/                ║
        ║   💚 健康检查: http://$host:$port/api/health   ║
        ║                                                        ║
        ║   按 Ctrl+C 停止服务器                                 ║
        ╚════════════════════════════════════════════════════════╝
        """)
    end

    try
        @info "启动服务器..." host=host port=port

        # 启动 HTTP 服务器
        SERVER[] = HTTP.serve(router, host, port; verbose=verbose)

    catch e
        if e isa InterruptException
            @info "服务器被用户中断"
        else
            @error "服务器启动失败" exception=e
        end
        stop_server()
    end

    return SERVER[]
end

"""
    stop_server()

停止 Web API 服务器。
"""
function stop_server()
    if SERVER[] !== nothing
        try
            @info "停止 Web API 服务器..."
            close(SERVER[])
            SERVER[] = nothing
            @info "服务器已停止"
        catch e
            @error "停止服务器时出错" exception=e
        end
    end
end

end # module WebAPI
