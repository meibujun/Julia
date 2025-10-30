# DeepLearning.jl - 深度学习模型模块
# ==========================================================
# 本文件经过 GPU 加速重构。所有深度学习模型现在都可以自动检测
# 并利用可用的 CUDA GPU 来加速训练和推理。
# ==========================================================

module DeepLearning

# --- 1. 导入依赖 ---
using Flux
using DataFrames
using Statistics
using ..GenomicPrediction: AbstractModel, GenomicData, fit!, predict
using LinearAlgebra
using NNlib
using GraphNeuralNetworks
using Flux: train!, mse
using Random
using CUDA

# --- 2. 模块接口 ---
export FNNModel, CNNModel, TransformerModel, GNNModel

# --- 3. GPU/CPU 设备选择辅助函数 ---
function _get_device()
    if CUDA.functional()
        println("检测到 CUDA GPU，将使用 GPU 进行计算。")
        return gpu
    else
        println("未检测到 CUDA GPU，将使用 CPU 进行计算。")
        return cpu
    end
end

# --- 4. FNN/MLP 实现 (GPU 加速) ---
@doc raw"""
    FNNModel(input_dim::Int; ...)
"""
mutable struct FNNModel <: AbstractModel
    chain::Chain; optimizer; epochs::Int; history::Vector{Float32}
    function FNNModel(input_dim::Int; hidden_layers=[64, 32], epochs=20, learning_rate=0.001)
        layers = [Dense(input_dim => hidden_layers[1], relu), [Dense(hidden_layers[i] => hidden_layers[i+1], relu) for i in 1:(length(hidden_layers)-1)]..., Dense(hidden_layers[end] => 1)]
        new(Chain(layers...), Adam(learning_rate), epochs, [])
    end
end

function fit!(model::FNNModel, data::GenomicData; rng=nothing)
    device = _get_device()
    model.chain = model.chain |> device

    X = Float32.(Matrix(data.genotypes[!, 2:end]))' |> device
    y = Float32.(data.phenotypes[!, 2])' |> device
    loss(m, x, y) = mse(m(x), y)
    loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)

    println("开始 FNN 模型训练..."); empty!(model.history)
    for epoch in 1:model.epochs
        train!(loss, model.chain, loader, model.optimizer)
        current_loss = loss(model.chain, X, y) |> cpu
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::FNNModel, new_data::DataFrame)
    device = model.chain isa Flux.Chain ? cpu : CUDA.device(model.chain)
    X_new = Float32.(Matrix(new_data[!, 2:end]))' |> device
    return model.chain(X_new) |> cpu
end

# --- 5. CNN 实现 (GPU 加速) ---
@doc raw"""
    CNNModel(input_dim::Int; ...)
"""
mutable struct CNNModel <: AbstractModel
    chain::Chain; optimizer; epochs::Int; history::Vector{Float32}
    function CNNModel(input_dim::Int; epochs=20, learning_rate=0.001)
        dummy_conv = Chain(x -> reshape(x, input_dim, 1, 1, 1), Conv((3, 1), 1=>4, relu), MaxPool((2, 1)), flatten)
        conv_output_size = size(dummy_conv(zeros(Float32, input_dim, 1)), 1)
        chain = Chain(x -> reshape(x, size(x, 1), 1, 1, size(x, 2)), Conv((3, 1), 1=>4, relu), MaxPool((2, 1)), flatten, Dense(conv_output_size => 128, relu), Dense(128 => 1))
        new(chain, Adam(learning_rate), epochs, [])
    end
end

function fit!(model::CNNModel, data::GenomicData; rng=nothing)
    device = _get_device()
    model.chain = model.chain |> device

    X = Float32.(Matrix(data.genotypes[!, 2:end]))' |> device
    y = Float32.(data.phenotypes[!, 2])' |> device
    loss(m, x, y) = mse(m(x), y)
    loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)

    println("开始 CNN 模型训练..."); empty!(model.history)
    for epoch in 1:model.epochs
        train!(loss, model.chain, loader, model.optimizer)
        current_loss = loss(model.chain, X, y) |> cpu
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::CNNModel, new_data::DataFrame)
    device = model.chain isa Flux.Chain ? cpu : CUDA.device(model.chain)
    X_new = Float32.(Matrix(new_data[!, 2:end]))' |> device
    return model.chain(X_new) |> cpu
end

# --- 6. Transformer 实现 (GPU 加速) ---
struct TransformerBlock; attention::MultiHeadAttention; norm1::LayerNorm; feedforward::Chain; norm2::LayerNorm; end
Flux.@functor TransformerBlock
function TransformerBlock(d_model::Int, n_head::Int; d_ff=256); TransformerBlock(MultiHeadAttention(n_head, d_model), LayerNorm(d_model), Chain(Dense(d_model => d_ff, relu), Dense(d_ff => d_model)), LayerNorm(d_model)); end
(b::TransformerBlock)(x) = b.norm2(x + b.feedforward(b.norm1(x + b.attention(x, x, x))))

@doc raw"""
    TransformerModel(input_dim::Int; ...)
"""
mutable struct TransformerModel <: AbstractModel
    chain::Chain; optimizer; epochs::Int; history::Vector{Float32}
    function TransformerModel(input_dim::Int; d_model=32, n_head=4, n_layers=2, epochs=10)
        layers = [Dense(1 => d_model), (x -> permutedims(x, (2, 1, 3))), [TransformerBlock(d_model, n_head) for _ in 1:n_layers]..., (x -> mean(x, dims=2)), flatten, Dense(d_model => 1)]
        new(Chain(layers...), Adam(), epochs, [])
    end
end

function fit!(model::TransformerModel, data::GenomicData; rng=nothing)
    device = _get_device()
    model.chain = model.chain |> device

    X_mat = Float32.(Matrix(data.genotypes[!, 2:end]))'
    y_mat = Float32.(data.phenotypes[!, 2])'
    X = reshape(X_mat, size(X_mat, 1), 1, size(X_mat, 2)) |> device
    y = y_mat |> device
    loss(m, x, y) = mse(m(x), y)
    loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)

    println("开始 Transformer 模型训练..."); empty!(model.history)
    for epoch in 1:model.epochs
        train!(loss, model.chain, loader, model.optimizer)
        current_loss = loss(model.chain, X, y) |> cpu
        push!(model.history, current_loss)
        if epoch % 5 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::TransformerModel, new_data::DataFrame)
    device = model.chain isa Flux.Chain ? cpu : CUDA.device(model.chain)
    X_new_mat = Float32.(Matrix(new_data[!, 2:end]))'
    X_new = reshape(X_new_mat, size(X_new_mat, 1), 1, size(X_new_mat, 2)) |> device
    return model.chain(X_new) |> cpu
end

# --- 7. GNN 实现 (GPU 加速) ---
@doc raw"""
    GNNModel(input_dim::Int; ...)
"""
mutable struct GNNModel <: AbstractModel
    chain::Chain; optimizer; epochs::Int; history::Vector{Float32}; graph::Union{GNNGraph, Nothing}
    function GNNModel(input_dim::Int; gcn_dims=[16, 32], epochs=20)
        chain = GNNChain(GCNConv(input_dim => gcn_dims[1], relu), GCNConv(gcn_dims[1] => gcn_dims[2]), GlobalPool(mean), Dense(gcn_dims[2] => 1))
        new(chain, Adam(), epochs, [], nothing)
    end
end

function fit!(model::GNNModel, data::GenomicData; rng=nothing)
    device = _get_device()
    model.chain = model.chain |> device

    X = Float32.(Matrix(data.genotypes[!, 2:end]))
    y = Float32.(data.phenotypes[!, 2])

    p = mean(X, dims=1) ./ 2; M = X .- (2 .* p); K = M * M'
    K_norm = (K .- minimum(K)) ./ (maximum(K) - minimum(K))
    adj = K_norm .> 0.7

    model.graph = GNNGraph(adj, ndata=X') |> device
    y_device = y |> device

    loss(m, g, y_target) = mse(m(g, g.ndata.x), y_target')

    println("开始 GNN 模型训练..."); empty!(model.history)
    for epoch in 1:model.epochs
        grads = gradient(m -> loss(m, model.graph, y_device), model.chain)
        Flux.update!(model.optimizer, model.chain, grads[1])
        current_loss = loss(model.chain, model.graph, y_device) |> cpu
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::GNNModel, new_data::DataFrame)
    if isnothing(model.graph); error("模型未训练或图结构未创建。"); end
    # 预测在创建的图上进行，结果需要移回 CPU
    full_prediction = model.chain(model.graph, model.graph.ndata.x) |> cpu
    println("警告: GNN predict 返回训练图中所有节点的预测值。")
    return full_prediction[1, :]
end

end # module DeepLearning
