# DeepLearning.jl - 深度学习模型模块
# ==========================================================
# 负责实现用于基因组预测的深度学习模型，例如全连接神经网络（FNN/MLP）、
# 卷积神经网络（CNN）、Transformer 和图神经网络（GNN）。
#
# 该模块利用 Flux.jl 框架，并为基因组数据提供定制化的网络结构。
# 所有模型都遵循 `AbstractModel` 接口。
# ==========================================================

module DeepLearning

# --- 1. 导入依赖 ---
using Flux
using DataFrames
using Statistics
using ..GenomicPrediction: AbstractModel
using ..DataProcessing: GenomicData
using LinearAlgebra
using GraphNeuralNetworks
using Flux: MultiHeadAttention, LayerNorm, Adam

# --- 2. 模块接口实现 ---
export FNNModel, CNNModel, TransformerModel, GNNModel

# --- FNN/MLP Implementation ---
@doc raw"""
    FNNModel(input_dim::Int; hidden_layers=[64, 32], epochs=20, learning_rate=0.001)
"""
mutable struct FNNModel <: AbstractModel
    chain::Chain
    optimizer
    epochs::Int
    history::Vector{Float32}
    function FNNModel(input_dim::Int; hidden_layers=[64, 32], epochs=20, learning_rate=0.001)
        layers = Any[Dense(input_dim => hidden_layers[1], relu)]
        for i in 1:(length(hidden_layers)-1)
            push!(layers, Dense(hidden_layers[i] => hidden_layers[i+1], relu))
        end
        push!(layers, Dense(hidden_layers[end] => 1))
        chain = Chain(layers...)
        optimizer = Adam(learning_rate)
        new(chain, optimizer, epochs, [])
    end
end

function fit!(model::FNNModel, data::GenomicData)
    X = Float32.(Matrix(data.genotypes))'
    y = Float32.(data.phenotypes[!, 1])'
    loss(m, x, y) = Flux.mse(m(x), y)
    opt_state = Flux.setup(model.optimizer, model.chain)
    println("开始 FNN 模型训练 (轮数: $(model.epochs))...")
    empty!(model.history)
    loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)
    for epoch in 1:model.epochs
        for (x_batch, y_batch) in loader
            grads = gradient(m -> loss(m, x_batch, y_batch), model.chain)
            Flux.update!(opt_state, model.chain, grads[1])
        end
        current_loss = loss(model.chain, X, y)
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::FNNModel, new_data::DataFrame)
    X_new = Float32.(Matrix(new_data))'
    return model.chain(X_new)[1, :]
end

# --- CNN Implementation ---
function get_conv_output_size(input_dim)
    conv_layers = Chain(
        x -> reshape(x, (input_dim, 1, 1, 1)),
        Conv((5, 1), 1=>16, relu),
        MaxPool((2, 1)),
        Conv((3, 1), 16=>32, relu),
        MaxPool((2, 1)),
        Flux.flatten
    )
    dummy_input = randn(Float32, input_dim)
    return size(conv_layers(dummy_input), 1)
end

@doc raw"""
    CNNModel(input_dim::Int; epochs=20, learning_rate=0.001)
"""
mutable struct CNNModel <: AbstractModel
    chain::Chain
    optimizer
    epochs::Int
    history::Vector{Float32}
    function CNNModel(input_dim::Int; epochs=20, learning_rate=0.001)
        conv_output_size = get_conv_output_size(input_dim)
        chain = Chain(
            x -> reshape(x, (input_dim, 1, 1, size(x, 2))),
            Conv((5, 1), 1=>16, relu),
            MaxPool((2, 1)),
            Conv((3, 1), 16=>32, relu),
            MaxPool((2, 1)),
            Flux.flatten,
            Dense(conv_output_size => 128, relu),
            Dense(128 => 1)
        )
        optimizer = Adam(learning_rate)
        new(chain, optimizer, epochs, [])
    end
end

function fit!(model::CNNModel, data::GenomicData)
    X = Float32.(Matrix(data.genotypes))'
    y = Float32.(data.phenotypes[!, 1])'
    loss(m, x, y) = Flux.mse(m(x), y)
    opt_state = Flux.setup(model.optimizer, model.chain)
    loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)
    println("开始 CNN 模型训练 (轮数: $(model.epochs))...")
    empty!(model.history)
    for epoch in 1:model.epochs
        for (x_batch, y_batch) in loader
            grads = gradient(m -> loss(m, x_batch, y_batch), model.chain)
            Flux.update!(opt_state, model.chain, grads[1])
        end
        current_loss = loss(model.chain, X, y)
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::CNNModel, new_data::DataFrame)
    X_new = Float32.(Matrix(new_data))'
    return model.chain(X_new)[1, :]
end

# --- Transformer Implementation ---
struct SimpleTransformerBlock
    attention::MultiHeadAttention
    norm1::LayerNorm
    feedforward::Chain
    norm2::LayerNorm
end
Flux.@functor SimpleTransformerBlock
function SimpleTransformerBlock(d_model::Int, n_head::Int, d_ff::Int)
    attention = MultiHeadAttention(d_model; nheads=n_head)
    norm1 = LayerNorm(d_model)
    feedforward = Chain(Dense(d_model => d_ff, relu), Dense(d_ff => d_model))
    norm2 = LayerNorm(d_model)
    return SimpleTransformerBlock(attention, norm1, feedforward, norm2)
end
function (b::SimpleTransformerBlock)(x)
    attn_out, _ = b.attention(x, x, x)
    x = b.norm1(x .+ attn_out)
    ff_out = b.feedforward(x)
    return b.norm2(x .+ ff_out)
end

@doc raw"""
    TransformerModel(input_dim::Int; n_head=2, d_model=16, n_layers=2, epochs=10)
"""
mutable struct TransformerModel <: AbstractModel
    chain::Chain
    optimizer
    epochs::Int
    history::Vector{Float32}
    function TransformerModel(input_dim::Int; n_head=2, d_model=16, n_layers=2, epochs=10)
        encoder_blocks = [SimpleTransformerBlock(d_model, n_head, d_model * 4) for _ in 1:n_layers]
        chain = Chain(
            Dense(input_dim => d_model),
            x -> reshape(x, (size(x, 1), 1, size(x, 2))), # Reshape to (d_model, 1, batch_size)
            Chain(encoder_blocks...),
            x -> reshape(x, (size(x, 1), size(x, 3))),   # Reshape back to (d_model, batch_size)
            Dense(d_model => 1)
        )
        optimizer = Adam()
        new(chain, optimizer, epochs, [])
    end
end

function fit!(model::TransformerModel, data::GenomicData)
    X = Float32.(Matrix(data.genotypes))'
    y = Float32.(data.phenotypes[!, 1])'
    loss(m, x, y) = Flux.mse(m(x), y)
    opt_state = Flux.setup(model.optimizer, model.chain)
    println("开始 Transformer 模型训练 (轮数: $(model.epochs))...")
    empty!(model.history)
    loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)
    for epoch in 1:model.epochs
        for (x_batch, y_batch) in loader
             grads = gradient(m -> loss(m, x_batch, y_batch), model.chain)
             Flux.update!(opt_state, model.chain, grads[1])
        end
        current_loss = loss(model.chain, X, y)
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::TransformerModel, new_data::DataFrame)
    X_new = Float32.(Matrix(new_data))'
    return model.chain(X_new)[1, :]
end

# --- GNN Implementation ---
@doc raw"""
    GNNModel(input_dim::Int; gcn_dims=[16, 8], epochs=10)
"""
mutable struct GNNModel <: AbstractModel
    gnn_base::GNNChain
    regression_head::Dense
    optimizer
    epochs::Int
    history::Vector{Float32}
    function GNNModel(input_dim::Int; gcn_dims=[16, 8], epochs=10)
        gnn_base = GNNChain([GCNConv(din => dout, relu) for (din, dout) in zip((input_dim, gcn_dims...), gcn_dims)]...)
        regression_head = Dense(gcn_dims[end] => 1)
        optimizer = Adam()
        new(gnn_base, regression_head, optimizer, epochs, [])
    end
end

function fit!(model::GNNModel, data::GenomicData)
    n = size(data.genotypes, 1)
    adj_matrix = ones(Float32, n, n) - I(n)
    g = GNNGraph(adj_matrix)
    X = Float32.(Matrix(data.genotypes))' # GNN expects (features, nodes)
    y = Float32.(data.phenotypes[!, 1])'

    full_model = (gnn=model.gnn_base, head=model.regression_head)
    loss(m, g, x, y) = Flux.mse(m.head(m.gnn(g, x)), y)

    opt_state = Flux.setup(model.optimizer, full_model)

    println("开始 GNN 模型训练 (轮数: $(model.epochs))...")
    empty!(model.history)
    for epoch in 1:model.epochs
        grads = gradient(m -> loss(m, g, X, y), full_model)
        Flux.update!(opt_state, full_model, grads[1])
        current_loss = loss(full_model, g, X, y)
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1; println("轮数: $epoch, 损失: $current_loss"); end
    end
    println("训练完成。")
end

function predict(model::GNNModel, new_data::DataFrame)
    n = size(new_data, 1)
    adj_matrix = ones(Float32, n, n) - I(n)
    g = GNNGraph(adj_matrix)
    X_new = Float32.(Matrix(new_data))' # GNN expects (features, nodes)

    full_model = (gnn=model.gnn_base, head=model.regression_head)
    output = full_model.head(full_model.gnn(g, X_new))
    return vec(output)
end

end # module DeepLearning
