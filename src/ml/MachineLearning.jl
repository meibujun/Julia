module MachineLearning

using DataFrames
using Statistics
using Random
using DecisionTree
using Flux

export MLModel, train_ml_model, predict, cross_validate

"""
    MLModel

机器学习模型包装器，记录训练方法、特征标准化参数及超参数配置。
"""
struct MLModel
    method::Symbol
    fitted::Any
    feature_names::Vector{Symbol}
    mean::Vector{Float64}
    std::Vector{Float64}
    params::Dict{Symbol,Any}
end

"""
    train_ml_model(method, features, labels; kwargs...)

训练指定机器学习模型，并自动完成特征标准化。支持随机森林、梯度提升树以及
简单神经网络。
"""
function train_ml_model(method::Symbol, features, labels; kwargs...)
    X, feature_names = _coerce_features(features)
    y = Float64.(labels)
    μ, σ = _standardise!(X)
    if method == :RandomForest
        n_trees = get(kwargs, :n_trees, 100)
        max_depth = get(kwargs, :max_depth, -1)
        min_samples_leaf = get(kwargs, :min_samples_leaf, 5)
        model = DecisionTree.RandomForestRegressor(transpose(X), y; n_trees, max_depth, min_samples_leaf)
    elseif method == :GradientBoosting
        shrinkage = get(kwargs, :shrinkage, 0.1)
        max_depth = get(kwargs, :max_depth, 3)
        n_rounds = get(kwargs, :n_rounds, 50)
        model = DecisionTree.GBRegressor(transpose(X), y; shrinkage, max_depth, n_rounds)
    elseif method == :NeuralNetwork
        epochs = get(kwargs, :epochs, 200)
        hidden = get(kwargs, :hidden, 32)
        η = get(kwargs, :η, 1e-3)
        model = _train_neural_network(X, y; epochs, hidden, η)
    else
        throw(ArgumentError("Unsupported ML method $(method)"))
    end
    return MLModel(method, model, feature_names, μ, σ, Dict(kwargs))
end

"""
    predict(model, features)

使用已训练模型对新特征进行预测，内部自动应用训练时的标准化参数。
"""
function predict(model::MLModel, features)
    X, _ = _coerce_features(features; names = model.feature_names)
    _apply_standardisation!(X, model.mean, model.std)
    if model.method == :RandomForest
        return DecisionTree.predict(model.fitted, transpose(X))
    elseif model.method == :GradientBoosting
        return DecisionTree.predict(model.fitted, transpose(X))
    elseif model.method == :NeuralNetwork
        return vec(Array(model.fitted(transpose(X))))
    else
        throw(ArgumentError("Unsupported ML method $(model.method)"))
    end
end

"""
    cross_validate(features, labels, method; n_folds = 5)

执行 K 折交叉验证，返回平均相关系数及各折结果，便于评估模型稳定性。
"""
function cross_validate(features, labels, method::Symbol; n_folds::Int = 5, rng::AbstractRNG = Random.default_rng(), kwargs...)
    X, _ = _coerce_features(features)
    y = Float64.(labels)
    n = size(X, 1)
    indices = collect(1:n)
    shuffle!(rng, indices)
    fold_size = ceil(Int, n / n_folds)
    folds = [indices[((i - 1) * fold_size + 1):min(i * fold_size, n)] for i in 1:n_folds]
    isempty(last(folds)) && pop!(folds)
    corrs = Float64[]
    for fold in folds
        test_idx = fold
        train_idx = setdiff(indices, test_idx)
        model = train_ml_model(method, X[train_idx, :], y[train_idx]; kwargs...)
        preds = predict(model, X[test_idx, :])
        push!(corrs, cor(preds, y[test_idx]))
    end
    return Dict(:mean_correlation => mean(corrs), :fold_correlations => corrs)
end

"""
    _coerce_features(features; names = Symbol[])

将输入特征转换为 `Matrix{Float64}`，并返回对应的列名。
"""
function _coerce_features(features; names = Symbol[])
    if features isa DataFrame
        X = Matrix{Float64}(features)
        feature_names = Symbol.(names(features))
    else
        X = Matrix{Float64}(features)
        feature_names = isempty(names) ? Symbol.("x" .* string.(1:size(X, 2))) : names
    end
    return X, feature_names
end

"""
    _standardise!(X)

原位标准化特征矩阵，同时返回均值和标准差向量。
"""
function _standardise!(X::Matrix{Float64})
    μ = vec(mean(X; dims = 1))
    σ = vec(std(X; dims = 1))
    for j in eachindex(σ)
        σ[j] = σ[j] == 0 ? 1.0 : σ[j]
    end
    for j in axes(X, 2)
        X[:, j] .= (X[:, j] .- μ[j]) ./ σ[j]
    end
    return μ, σ
end

"""
    _apply_standardisation!(X, μ, σ)

根据训练阶段的统计量对特征执行同样的标准化步骤。
"""
function _apply_standardisation!(X::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    for j in axes(X, 2)
        X[:, j] .= (X[:, j] .- μ[j]) ./ σ[j]
    end
    return X
end

"""
    _train_neural_network(X, y; epochs, hidden, η)

使用 Flux 实现的简单两层感知机训练过程。
"""
function _train_neural_network(X::Matrix{Float64}, y::Vector{Float64}; epochs::Int, hidden::Int, η::Float64)
    n_features = size(X, 2)
    model = Chain(Dense(n_features, hidden, relu), Dense(hidden, 1))
    ps = Flux.params(model)
    opt = Flux.Adam(η)
    X_batch = transpose(X)
    y_batch = reshape(y, 1, :)
    for epoch in 1:epochs
        grads = Flux.gradient(ps) do
            ŷ = model(X_batch)
            Flux.Losses.mse(ŷ, y_batch)
        end
        Flux.Optimise.update!(opt, ps, grads)
    end
    return model
end

end
