# --- BLUP 评估模块 ---
# 该模块是遗传评估的核心计算引擎。
# 它实现了混合模型方程 (MME) 的构建和求解，以获得育种值 (BLUP)。

"""
    EvaluationResult

一个结构体(struct)，用于存储BLUP评估的完整结果。

# 字段
- `fixed_effects::Vector{Float64}`: 估计出的固定效应值 (β̂)。
- `random_effects::Vector{Float64}`: 预测出的随机效应值 (û)，即育种值 (EBV)。
- `h2::Float64`: 模型中使用的遗传力参数。
- `variance_components::Tuple{Float64, Float64}`: 模型中使用的方差组分，格式为 `(加性遗传方差, 残差方差)` 即 `(σ²_a, σ²_e)`。
- `animal_ids::Vector`: 与随机效应 `u` 向量顺序一致的动物ID列表。
"""
struct EvaluationResult
    fixed_effects::Vector{Float64}
    random_effects::Vector{Float64}
    h2::Float64
    variance_components::Tuple{Float64, Float64}
    animal_ids::Vector
end

"""
    Base.show(io::IO, res::EvaluationResult)

重载 `show` 函数，以便在打印 `EvaluationResult` 对象时，以一种更美观、更易读的格式显示其摘要信息。
"""
function Base.show(io::IO, res::EvaluationResult)
    println(io, "--- BLUP 评估结果摘要 ---")
    println(io, "模型使用的遗传力 (h²):         ", round(res.h2, digits=4))
    println(io, "加性遗传方差 (σ²_a):        ", round(res.variance_components[1], digits=4))
    println(io, "残差方差 (σ²_e):            ", round(res.variance_components[2], digits=4))
    println(io, "估计的固定效应数量:         ", length(res.fixed_effects))
    println(io, "预测的随机效应 (育种值) 数量: ", length(res.random_effects))
    println(io, "-----------------------------")
end


"""
    run_evaluation(model::ModelSpec, dm::DataManager; method::Symbol=:BLUP, h2::Float64=0.3) -> EvaluationResult

根据指定的模型和数据，运行遗传评估。

此函数是执行BLUP分析的核心入口。当前版本实现了一个基础的BLUP求解器，
它假设方差组分（或遗传力）是已知的。未来的版本将加入REML等方法来估计方差组分。

# 参数
- `model::ModelSpec`: 定义模型结构的对象。
- `dm::DataManager`: 包含所有数据的对象。
- `method::Symbol`: 指定评估方法。当前仅支持 `:BLUP`。
- `h2::Float64`: 模型的遗传力，取值范围在 (0, 1) 之间。

# 返回
- `EvaluationResult`: 一个包含所有评估结果的对象。
"""
function run_evaluation(model::ModelSpec, dm::DataManager; method::Symbol=:BLUP, h2::Float64=0.3)
    # --- 输入验证 ---
    if method != :BLUP
        error("当前仅支持 `:BLUP` 方法。REML和贝叶斯方法即将推出。")
    end
    if !(0 < h2 < 1)
        error("遗传力 (h2) 的值必须在 0 和 1 之间。")
    end

    println("--- 开始BLUP评估 ---")

    # --- 1. 构建模型矩阵 (y, X, Z) ---
    println("步骤 1/6: 构建模型矩阵...")
    y, X, Z = model(dm)

    # --- 2. 获取关系矩阵 A ---
    # 当前实现假定是动物加性遗传模型，因此需要A矩阵。
    # 未来可以根据模型定义（如 GBLUP）灵活选择 G 或 H 矩阵。
    println("步骤 2/6: 获取关系矩阵...")
    if !isdefined(dm, :A) || isempty(dm.A)
        @info "未找到预先计算的A矩阵，现在从谱系数据开始计算..."
        compute_relationship_matrix(dm, type=:pedigree)
    end
    A = dm.A

    # 检查Z矩阵和A矩阵的维度是否匹配
    if size(Z, 2) != size(A, 1)
        error("维度不匹配：Z矩阵的列数 ($(size(Z, 2))) 与A矩阵的维度 ($(size(A, 1))) 不同。这通常意味着有表型记录的动物不在谱系中，或反之。")
    end

    # --- 3. 设置方差组分和 lambda ---
    println("步骤 3/6: 设置方差组分...")
    # 在这个简化的BLUP实现中，我们假设总表型方差 σ²_p = 1。
    # 因此，加性遗传方差 σ²_a = h² * σ²_p = h²
    # 残差方差 σ²_e = (1 - h²) * σ²_p = 1 - h²
    sigma2_a = h2
    sigma2_e = 1.0 - h2
    lambda = sigma2_e / sigma2_a
    println("λ (lambda) 值计算为: ", round(lambda, digits=4))

    # --- 4. 获取 A 矩阵的逆 ---
    println("步骤 4/6: 计算A矩阵的逆...")
    # 警告：这是当前实现的主要性能瓶颈。对于大型矩阵，直接求逆非常耗时。
    # 高效的求解器会直接使用稀疏的 A-inverse 矩阵。
    A_inv = inv(Matrix(A)) # 将稀疏矩阵转换为稠密矩阵以便求逆

    # --- 5. 构建混合模型方程 (MME) ---
    # MME 形式如下:
    # [ X'X   X'Z     ] [β̂]   [ X'y ]
    # [ Z'X   Z'Z+λA⁻¹ ] [û] = [ Z'y ]
    println("步骤 5/6: 构建混合模型方程 (MME)...")

    X_t = transpose(X)
    Z_t = transpose(Z)

    # 左手边 (LHS)
    lhs_11 = X_t * X
    lhs_12 = X_t * Z
    lhs_21 = Z_t * X
    lhs_22 = Z_t * Z + lambda .* A_inv

    LHS = [lhs_11 lhs_12; lhs_21 lhs_22]

    # 右手边 (RHS)
    rhs_1 = X_t * y
    rhs_2 = Z_t * y
    RHS = [rhs_1; rhs_2]

    # --- 6. 求解 MME ---
    println("步骤 6/6: 求解 MME...")
    # 使用 Julia 内置的反斜杠运算符 `\`，这是一个高效且数值稳定的线性方程组求解器。
    solution = LHS \ RHS
    println("...求解完成。")

    # 提取解
    n_fixed = size(X, 2)
    beta_hat = solution[1:n_fixed]
    u_hat = solution[n_fixed+1:end]

    # 获取与u_hat顺序一致的动物ID列表
    animal_ids = sort(unique(dm.pedigree.animal))

    result = EvaluationResult(beta_hat, u_hat, h2, (sigma2_a, sigma2_e), animal_ids)

    println("--- BLUP评估完成 ---")
    return result
end


"""
    save_results(result::EvaluationResult, filepath::String)

将评估结果（特别是育种值）保存到CSV文件中。

# 参数
- `result::EvaluationResult`: 包含评估结果的对象。
- `filepath::String`: 输出CSV文件的路径。
"""
function save_results(result::EvaluationResult, filepath::String)
    # 确保动物ID数量与育种值数量匹配
    if length(result.animal_ids) != length(result.random_effects)
       error("结果保存失败：动物ID数量与预测的育种值数量不匹配。")
    end

    # 创建一个DataFrame用于保存
    results_df = DataFrame(
        animal = result.animal_ids,
        breeding_value = result.random_effects
    )

    try
        CSV.write(filepath, results_df)
        println("评估结果已成功保存至 '$filepath'。")
    catch e
        error("保存结果至 '$filepath' 失败。原因: $e")
    end
end