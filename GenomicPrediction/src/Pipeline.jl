# Pipeline.jl - 高级工作流 API
# ==========================================================
# 本文件已更新，以无缝集成自动化的实验追踪系统。
# 每次运行工作流时，其配置、数据哈希和结果都会被自动记录。
# ==========================================================

module Pipeline

using TOML
using SHA
using ..GenomicPrediction

# --- 1. 模块接口 ---
export run_pipeline

# --- 2. 自定义异常类型 ---
struct PipelineConfigError <: Exception; msg::String; end
Base.showerror(io::IO, e::PipelineConfigError) = print(io, "PipelineConfigError: ", e.msg)

# --- 3. 核心工作流函数 ---

@doc raw"""
    run_pipeline(config_path::String)
"""
function run_pipeline(config_path::String)
    println("--- 开始执行基因组预测工作流 ---")

    # 1. 初始化实验数据库
    db_path = get(ENV, "GP_DB_PATH", "genomic_prediction_experiments.db")
    init_db(db_path)

    # 2. 读取和验证配置
    config = TOML.parsefile(config_path)
    _validate_config(config)

    # 3. 计算数据哈希以实现可追溯性
    data_hash = _calculate_data_hash(config["data"])
    println("  数据哈希: $data_hash")

    # 4. 加载数据
    println("\n[步骤 1/3] 正在加载数据...")
    data = _load_data_from_config(config["data"])

    # 5. 创建模型
    println("\n[步骤 2/3] 正在创建模型...")
    model_name = config["model"]["name"]
    model_params = Dict(Symbol(k) => v for (k,v) in config["model"]["params"])
    model = create_model(model_name, model_params)

    # 6. 执行分析
    println("\n[步骤 3/3] 正在执行分析...")
    results = nothing
    model_path = get(config["output"], "model_path", nothing)

    if config["analysis"]["type"] == "train"
        fit!(model, data)
        if !isnothing(model_path); save_model(model, model_path); end

    elseif config["analysis"]["type"] == "cross_validation"
        k = get(config["analysis"], "k", 5)
        results = cross_validate(model, data; k=k)
    end

    # 7. 自动记录实验
    println("\n正在记录实验结果...")
    log_experiment!(
        model_name=model_name,
        params=model_params,
        data_hash=data_hash,
        results=results,
        model_path=model_path
    )

    println("\n--- 工作流执行完毕 ---")
end

# --- 4. 辅助函数 ---

function _validate_config(config)
    required = ["data", "model", "analysis"]
    for s in required; if !haskey(config, s); throw(PipelineConfigError("配置文件缺少 '$s' 部分")); end; end
end

function _calculate_data_hash(data_config)
    ctx = SHA256_CTX()

    # 将所有输入文件的内容整合到哈希计算中
    for key in ["genotypes", "phenotypes", "covariates", "pedigree"]
        if haskey(data_config, key)
            path = data_config[key]
            if isfile(path)
                update!(ctx, read(path))
            end
        end
    end

    return bytes2hex(digest!(ctx))
end

function _load_data_from_config(data_config)
    format = get(data_config, "format", "csv")
    args = (data_config["genotypes"], data_config["phenotypes"])
    kwargs = (cov_path=get(data_config, "covariates", nothing), ped_path=get(data_config, "pedigree", nothing))

    if format == "csv"; return load_csv(args...; kwargs...);
    elseif format == "pgen"; return load_pgen(args...; kwargs...);
    else throw(PipelineConfigError("不支持的数据格式: '$format'")); end
end

end # module Pipeline
