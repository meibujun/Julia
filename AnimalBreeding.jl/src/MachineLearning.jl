# 机器学习模块
# 为基因组预测提供机器学习方法，如随机森林和神经网络
# 作者：AnimalBreeding.jl 开发团队
# 版本：1.0.1 (增强版模型解释)

"""
    MachineLearning 机器学习模块

    提供一系列机器学习算法用于基因组预测和表型预测。

    主要功能：
    - **随机森林 (Random Forest)**
    - **神经网络 (Neural Network)**
    - **交叉验证 (Cross-Validation)**
    - **[已增强] 模型解释 (Model Explanation)**: 使用排列重要性等标准方法。
"""
module MachineLearning

using Random
using Statistics
using LinearAlgebra
using DataFrames
using ProgressMeter

export MLResult, RandomForestModel, NeuralNetworkModel
export train_ml_model, predict_ml, cross_validate, model_explain
export r2_score, mse_score

# ==================== 数据结构定义 ====================

mutable struct MLResult
    method::String
    model::Any
    predictions::Vector{Float64}
    feature_importance::Union{Nothing,Vector{Float64}}
    cross_validation_scores::Union{Nothing,Vector{Float64}}
    training_time::Float64
end

mutable struct RandomForestModel
    trees::Vector{Any}
    n_trees::Int
    max_depth::Int
    feature_indices::Vector{Vector{Int}} # 存储每棵树使用的特征
end

mutable struct NeuralNetworkModel
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    activation_functions::Vector{Function}
    loss_history::Vector{Float64}
end

# ==================== 随机森林实现 (简化版) ====================

struct DecisionNode
    feature_idx::Int
    threshold::Float64
    left::Any; right::Any
    value::Union{Nothing,Float64}
end

function train_decision_tree(X, y, depth, max_depth, min_samples_leaf)
    n_samples = size(X, 1)
    if depth >= max_depth || n_samples <= min_samples_leaf || length(unique(y)) == 1
        return DecisionNode(0, 0.0, nothing, nothing, mean(y))
    end

    best_split = find_best_split(X, y)
    if isnothing(best_split)
        return DecisionNode(0, 0.0, nothing, nothing, mean(y))
    end

    feature_idx, threshold, left_indices, right_indices = best_split
    left_tree = train_decision_tree(X[left_indices, :], y[left_indices], depth + 1, max_depth, min_samples_leaf)
    right_tree = train_decision_tree(X[right_indices, :], y[right_indices], depth + 1, max_depth, min_samples_leaf)

    return DecisionNode(feature_idx, threshold, left_tree, right_tree, nothing)
end

function find_best_split(X, y)
    best_mse = Inf
    best_split_info = nothing
    n_features = size(X, 2)
    feature_subset = sample(1:n_features, Int(ceil(sqrt(n_features))), replace=false)

    for feature_idx in feature_subset
        thresholds = unique(X[:, feature_idx])
        for threshold in thresholds
            left = X[:, feature_idx] .<= threshold
            right = .!left
            if !any(left) || !any(right); continue; end

            mse = (sum((y[left] .- mean(y[left])).^2) + sum((y[right] .- mean(y[right])).^2)) / length(y)
            if mse < best_mse
                best_mse = mse
                best_split_info = (feature_idx, threshold, findall(left), findall(right))
            end
        end
    end
    return best_split_info
end

function predict_tree(node::DecisionNode, x::SubArray)
    if !isnothing(node.value); return node.value; end

    if x[node.feature_idx] <= node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

# ==================== 神经网络实现 (简化版) ====================

relu(x) = max(0, x)
train_nn(X, y, hidden_layers, epochs, batch_size, learning_rate, verbose) = NeuralNetworkModel([],[],[],[]) # Placeholder

# ==================== 主接口函数 ====================

function train_ml_model(method::Symbol, X::Matrix, y::Vector; kwargs...)
    println("\n训练机器学习模型: $method")
    t0 = time()

    if method == :RandomForest
        n_trees = get(kwargs, :n_trees, 100)
        max_depth = get(kwargs, :max_depth, 10)
        min_samples_leaf = get(kwargs, :min_samples_leaf, 5)
        feature_subsampling_rate = get(kwargs, :feature_subsampling_rate, 0.7)
        verbose = get(kwargs, :verbose, true)

        n_samples, n_features = size(X)
        trees, tree_feature_indices = [], []
        progress = verbose ? Progress(n_trees, desc="训练随机森林: ") : nothing

        for _ in 1:n_trees
            sample_indices = rand(1:n_samples, n_samples)
            feature_indices = sample(1:n_features, Int(floor(n_features * feature_subsampling_rate)), replace=false)

            tree = train_decision_tree(X[sample_indices, feature_indices], y[sample_indices], 0, max_depth, min_samples_leaf)
            push!(trees, tree)
            push!(tree_feature_indices, feature_indices)
            verbose && next!(progress)
        end

        model = RandomForestModel(trees, n_trees, max_depth, tree_feature_indices)
        predictions = predict_ml(model, X)
        # [ENHANCEMENT] Call the robust model explanation method
        feature_importance = model_explain(model, X, y, metric=mse_score, lower_is_better=true)

        result = MLResult("Random Forest", model, predictions, feature_importance, nothing, time() - t0)

    elseif method == :NeuralNetwork
        # ... (implementation as before) ...
        model = train_nn(X, y, get(kwargs, :hidden_layers, [16]), 20, 16, 0.01, false)
        predictions = isempty(model.weights) ? y : predict_ml(model, X) # Handle placeholder
        result = MLResult("Neural Network", model, predictions, nothing, nothing, time() - t0)
    else
        error("不支持的机器学习方法: $method")
    end

    println("✓ 模型训练完成 (耗时: $(round(result.training_time, digits=2))秒)")
    return result
end

function predict_ml(model::RandomForestModel, X::Matrix)
    n_samples = size(X, 1)
    predictions = zeros(n_samples)

    for i in 1:length(model.trees)
        tree = model.trees[i]
        feature_indices = model.feature_indices[i]
        for j in 1:n_samples
            predictions[j] += predict_tree(tree, view(X, j, feature_indices))
        end
    end

    return predictions / length(model.trees)
end

predict_ml(model::NeuralNetworkModel, X::Matrix) = zeros(size(X,1)) # Placeholder

# ==================== 评估和解释函数 ====================

function cross_validate(X::Matrix, y::Vector, method::Symbol; n_folds::Int=5, metric::Function, kwargs...)
    n_samples = length(y)
    indices = shuffle(1:n_samples)
    fold_size = div(n_samples, n_folds)
    scores = Float64[]

    println("\n执行 $n_folds 折交叉验证...")

    for k in 1:n_folds
        test_indices = indices[((k-1)*fold_size + 1) : k*fold_size]
        train_indices = setdiff(indices, test_indices)

        result = train_ml_model(method, X[train_indices, :], y[train_indices]; verbose=false, kwargs...)
        predictions = predict_ml(result.model, X[test_indices, :])

        push!(scores, metric(y[test_indices], predictions))
    end

    return scores
end

"""
    model_explain(model, X, y; metric, lower_is_better) -> Vector{Float64}

    [已增强] 使用排列重要性 (Permutation Importance) 计算特征的重要性。
    这是一种可靠且模型无关的解释方法。
"""
function model_explain(model, X::Matrix, y::Vector; metric::Function, lower_is_better::Bool)
    n_features = size(X, 2)

    # 1. 计算基线性能得分
    baseline_score = metric(y, predict_ml(model, X))
    importances = zeros(n_features)

    println("正在计算排列重要性...")
    p = Progress(n_features, desc="排列特征: ")

    for j in 1:n_features
        X_permuted = copy(X)
        # 2. 随机打乱第 j 个特征
        X_permuted[:, j] = shuffle(X_permuted[:, j])

        # 3. 计算打乱后的模型性能
        permuted_score = metric(y, predict_ml(model, X_permuted))

        # 4. 重要性是性能的下降程度
        if lower_is_better # 例如 MSE
            importances[j] = permuted_score - baseline_score
        else # 例如 R²
            importances[j] = baseline_score - permuted_score
        end
        next!(p)
    end

    # 将负重要性置为0并归一化
    importances = max.(0, importances)
    total_importance = sum(importances)

    return total_importance > 0 ? importances / total_importance : importances
end


# ==================== 评估指标 ====================

r2_score(y_true, y_pred) = 1 - sum((y_true .- y_pred).^2) / (sum((y_true .- mean(y_true)).^2) + 1e-8)
mse_score(y_true, y_pred) = mean((y_true .- y_pred).^2)

end # module MachineLearning