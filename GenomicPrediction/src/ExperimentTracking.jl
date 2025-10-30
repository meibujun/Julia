# ExperimentTracking.jl - 自动化的实验追踪系统
# ==========================================================
# 提供一个轻量级的、基于 SQLite 的实验追踪系统，以确保分析的
# 可复现性和系统化管理。
# ==========================================================

module ExperimentTracking

using SQLite
using DBInterface
using Dates
using TOML

# --- 1. 模块接口 ---
export init_db, log_experiment!, summarize_experiments

# --- 2. 数据库核心功能 ---

const DB_PATH = Ref("genomic_prediction_experiments.db")

@doc raw"""
    init_db(db_path::String="genomic_prediction_experiments.db")
初始化实验数据库，如果尚不存在，则创建一个新的数据库文件和表。
"""
function init_db(db_path::String="genomic_prediction_experiments.db")
    DB_PATH[] = db_path
    db = SQLite.DB(db_path)

    # 创建 `experiments` 表
    DBInterface.execute(db, """
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        model_name TEXT NOT NULL,
        parameters TEXT,
        data_hash TEXT,
        mean_accuracy REAL,
        mean_mse REAL,
        saved_model_path TEXT
    )
    """)
    println("实验数据库已在 '$db_path' 初始化。")
    return db
end

@doc raw"""
    log_experiment!(; model_name, params, data_hash, results=nothing, model_path=nothing)
将一次实验的详细信息记录到数据库中。
"""
function log_experiment!(; model_name::String, params::Dict, data_hash::String, results=nothing, model_path::Union{String, Nothing}=nothing)
    db = SQLite.DB(DB_PATH[])

    params_str = TOML.string(params)
    mean_acc = isnothing(results) ? nothing : results.mean_accuracy
    mean_mse = isnothing(results) ? nothing : results.mean_mse

    stmt = DBInterface.prepare(db, """
    INSERT INTO experiments (timestamp, model_name, parameters, data_hash, mean_accuracy, mean_mse, saved_model_path)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """)

    DBInterface.execute(stmt, (string(now()), model_name, params_str, data_hash, mean_acc, mean_mse, model_path))

    println("实验已成功记录到数据库。")
end

@doc raw"""
    summarize_experiments(n::Int=10)
显示最近 `n` 次实验的摘要。
"""
function summarize_experiments(n::Int=10)
    db = SQLite.DB(DB_PATH[])
    query = "SELECT id, timestamp, model_name, mean_accuracy, mean_mse FROM experiments ORDER BY timestamp DESC LIMIT ?"

    results = DBInterface.execute(db, query, (n,))

    println("--- 最近的 $n 次实验摘要 ---")
    for row in results
        println("ID: $(row.id), 时间: $(row.timestamp), 模型: $(row.model_name), 准确性: $(round(something(row.mean_accuracy, NaN), digits=4))")
    end
    println("--------------------------")
end

end # module ExperimentTracking
