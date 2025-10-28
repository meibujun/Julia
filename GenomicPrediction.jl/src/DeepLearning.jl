#=###############################################################################
# 深度学习模块
# 使用 Flux.jl 构建 MLP/CNN/Transformer 模型, 支持 GPU 训练与早停机制。
###############################################################################=#

using Flux
using Flux: Chain, Dense, Conv, BatchNorm, Dropout, relu, gelu, DataLoader,
            LayerNorm, MultiheadAttention, TransformerEncoder, positionalencoding, params
using CUDA
using Random
using Statistics

"""
    mutable struct TrainingHistory

记录训练与验证阶段的损失、指标轨迹。
"""
Base.@kwdef mutable struct TrainingHistory
    loss::Vector{Float64} = Float64[]
    val_loss::Vector{Float64} = Float64[]
    metrics::Dict{String,Vector{Float64}} = Dict{String,Vector{Float64}}()
    val_metrics::Dict{String,Vector{Float64}} = Dict{String,Vector{Float64}}()
end

"""
    to_device(x, device::Symbol)

根据 `device` 参数将数据/模型迁移到 CPU 或 GPU。
"""
function to_device(x, device::Symbol)
    if device == :gpu
        CUDA.functional() || error("CUDA 未就绪, 请在支持 GPU 的环境下运行或改用 :cpu")
        return cu(x)
    else
        return x
    end
end

# -------------------------- 模型构建器 --------------------------

"""
    build_mlp(input_dim, hidden_dims; activation=relu, dropout_rate=0.1, output_dim=1)

构建多层感知机, 默认在隐藏层后添加 Dropout 以缓解过拟合。
"""
function build_mlp(input_dim::Integer, hidden_dims::AbstractVector{<:Integer};
                   activation = relu, dropout_rate::Real = 0.1, output_dim::Integer = 1)
    layers = Vector{Any}()
    in_dim = input_dim
    for h in hidden_dims
        push!(layers, Dense(in_dim, h, activation))
        dropout_rate > 0 && push!(layers, Dropout(dropout_rate))
        in_dim = h
    end
    push!(layers, Dense(in_dim, output_dim))
    return Chain(layers...)
end

"""
    build_cnn(input_channels, input_length, conv_spec, dense_dims; activation=relu)

构建 1D 卷积网络, 适合 SNP 连续片段特征。
- `conv_spec` 为 `(out_channels, kernel, stride)` 元组数组。
- `dense_dims` 为卷积输出扁平化后的全连接层尺寸。
"""
function build_cnn(input_channels::Integer, input_length::Integer,
                   conv_spec::Vector{Tuple{Int,Int,Int}}, dense_dims::Vector{Int};
                   activation = relu, dropout_rate::Real = 0.1)
    layers = Any[
        x -> reshape(x, (input_channels, input_length, size(x, 2)))
    ]
    in_ch = input_channels
    current_length = input_length
    for (out_ch, kernel, stride) in conv_spec
        push!(layers, Conv((kernel,), in_ch => out_ch, activation; stride = stride))
        push!(layers, BatchNorm(out_ch))
        dropout_rate > 0 && push!(layers, Dropout(dropout_rate))
        current_length = max(1, Int(floor((current_length - kernel) / stride)) + 1)
        in_ch = out_ch
    end
    flat_dim = in_ch * current_length
    push!(layers, x -> reshape(x, (flat_dim, size(x, 3))))
    in_dim = flat_dim
    for h in dense_dims
        push!(layers, Dense(in_dim, h, activation))
        dropout_rate > 0 && push!(layers, Dropout(dropout_rate))
        in_dim = h
    end
    push!(layers, Dense(in_dim, 1))
    return Chain(layers...)
end

"""
    build_transformer(seq_len, embed_dim, num_heads, num_layers; ff_dim=4*embed_dim,
                      dropout_rate=0.1)

构建轻量 Transformer, 利用多头注意力捕捉远距离位点关联。
"""
function build_transformer(seq_len::Integer, embed_dim::Integer, num_heads::Integer, num_layers::Integer;
                           ff_dim::Integer = 4 * embed_dim, dropout_rate::Real = 0.1)
    encoder_blocks = Any[]
    for _ in 1:num_layers
        attn = MultiheadAttention(embed_dim, num_heads; dropout = dropout_rate)
        ff = Chain(Dense(embed_dim, ff_dim, gelu), Dropout(dropout_rate), Dense(ff_dim, embed_dim))
        push!(encoder_blocks, TransformerEncoder(attn, ff, LayerNorm(embed_dim)))
    end
    pos_enc = reshape(positionalencoding(embed_dim, seq_len), (embed_dim, seq_len, 1))
    embedding = Dense(seq_len, embed_dim * seq_len, gelu)
    return Chain(
        x -> reshape(x, (seq_len, size(x, 2))),
        embedding,
        x -> reshape(x, (embed_dim, seq_len, size(x, 2))),
        x -> x .+ pos_enc,
        encoder_blocks...,
        x -> reshape(x, (embed_dim * seq_len, size(x, 3))),
        Dropout(dropout_rate),
        Dense(embed_dim * seq_len, 1)
    )
end

# -------------------------- 训练逻辑 --------------------------

"""
    _prepare_data(X, y, device, batch_size, rng)

将矩阵 `X` (样本×特征) 转换为 Flux DataLoader。
"""
function _prepare_data(X::AbstractMatrix, y::AbstractVector, device::Symbol, batch_size::Integer, rng::AbstractRNG)
    finite_mask = .!isnan.(y)
    Xd = Float32.(permutedims(X[finite_mask, :]))  # 转换为 (特征, 批)
    yd = reshape(Float32.(y[finite_mask]), 1, :)
    return DataLoader((to_device(Xd, device), to_device(yd, device)); batchsize = batch_size, shuffle = true, rng = rng)
end

"""
    _evaluate_metrics(model, data, device, metric_fns)

计算一批数据上的指标, `metric_fns` 为名称=>函数字典。
"""
function _evaluate_metrics(model, xb, yb, metric_fns::Dict{String,Function})
    isempty(metric_fns) && return Dict{String,Float64}()
    preds = model(xb) |> Array |> vec
    truth = Array(yb) |> vec
    return Dict(name => float(fn(preds, truth)) for (name, fn) in metric_fns)
end

"""
    train_deep_model!(model, X, y; kwargs...) -> (TrainingHistory, Any)

训练 Flux 模型, 支持验证集、早停、梯度裁剪与自定义指标。
"""
function train_deep_model!(model, X::AbstractMatrix, y::AbstractVector;
                           epochs::Integer = 50,
                           batch_size::Integer = 64,
                           optimizer = Flux.ADAM(),
                           loss_fn = Flux.Losses.mse,
                           device::Symbol = :cpu,
                           val_data::Union{Nothing,Tuple{AbstractMatrix,AbstractVector}} = nothing,
                           rng::AbstractRNG = Random.default_rng(),
                           metric_fns::Dict{String,Function} = Dict{String,Function}(),
                           early_stopping::Bool = true,
                           patience::Integer = 10,
                           min_delta::Real = 1e-4,
                           grad_clip::Real = 5.0)
    data_loader = _prepare_data(X, y, device, batch_size, rng)
    model_device = device == :gpu ? fmap(cu, model) : model
    ps = params(model_device)
    history = TrainingHistory()
    best_val = Inf
    best_weights = [copy(p) for p in ps]
    wait = 0

    for epoch in 1:epochs
        epoch_loss = 0.0
        batches = 0
        for (xb, yb) in data_loader
            grads = gradient(ps) do
                pred = model_device(xb)
                loss_fn(pred, yb)
            end
            if grad_clip > 0
                Flux.Optimise.clip!(grads, grad_clip)
            end
            Flux.Optimise.update!(optimizer, ps, grads)
            epoch_loss += loss_fn(model_device(xb), yb) |> float
            batches += 1
        end
        push!(history.loss, epoch_loss / max(batches, 1))

        if !isempty(metric_fns)
            batch = first(data_loader)
            metrics = _evaluate_metrics(model_device, batch[1], batch[2], metric_fns)
            for (name, value) in metrics
                push!(get!(history.metrics, name, Float64[]), value)
            end
        end

        if isnothing(val_data)
            continue
        end
        X_val, y_val = val_data
        val_mask = .!isnan.(y_val)
        X_val = X_val[val_mask, :]
        y_val = y_val[val_mask]
        X_val_t = Float32.(permutedims(X_val))
        y_val_t = reshape(Float32.(y_val), 1, :)
        preds = model_device(to_device(X_val_t, device))
        current_val = loss_fn(preds, to_device(y_val_t, device)) |> float
        push!(history.val_loss, current_val)
        if !isempty(metric_fns)
            val_metrics = Dict(name => float(fn(vec(Array(preds)), y_val)) for (name, fn) in metric_fns)
            for (name, value) in val_metrics
                push!(get!(history.val_metrics, name, Float64[]), value)
            end
        end

        if early_stopping
            if current_val + min_delta < best_val
                best_val = current_val
                best_weights = [copy(p) for p in ps]
                wait = 0
            else
                wait += 1
                wait >= patience && break
            end
        end
    end

    if early_stopping && isfinite(best_val)
        for (param, best) in zip(ps, best_weights)
            param .= best
        end
    end

    return history, model_device
end

