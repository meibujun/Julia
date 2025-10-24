# DeepLearning.jl: 深度学习模块
# ------------------------------
# ... (module comments) ...

module DeepLearning

using Flux
using Flux: train!
using DataFrames
using ..DataProcessing

abstract type AbstractModel end

# --- FNNModel ---
mutable struct FNNModel <: AbstractModel
    # ... (FNNModel implementation) ...
    chain::Chain
    optimizer
    epochs::Int
    history::Vector{Float32}
    function FNNModel(input_dim::Int; hidden_dims::Vector{Int}=[64, 32], epochs::Int=10)
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims
            push!(layers, Dense(last_dim => h_dim, relu))
            last_dim = h_dim
        end
        push!(layers, Dense(last_dim => 1))
        chain = Chain(layers...)
        optimizer = Adam
        new(chain, optimizer, epochs, [])
    end
end

function fit!(model::FNNModel, data::GenomicData)
    # ... (FNNModel fit! implementation) ...
    X = Float32.(Matrix(data.genotypes))'
    y = Float32.(data.phenotypes[!, 1]')
    dataset = [(X, y)]
    loss(m, x, y) = Flux.mse(m(x), y)
    opt_state = Flux.setup(model.optimizer(), model.chain)
    println("开始 FNN 模型训练 (轮数: $(model.epochs))...")
    empty!(model.history)
    for epoch in 1:model.epochs
        train!(loss, model.chain, dataset, opt_state)
        current_loss = loss(model.chain, X, y)
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1
            println("轮数: $epoch, 损失: $current_loss")
        end
    end
    println("训练完成。")
    return nothing
end

function predict(model::FNNModel, new_data::DataFrame)
    # ... (FNNModel predict implementation) ...
    X_new = Float32.(Matrix(new_data))'
    return model.chain(X_new)[1, :]
end


# --- CNNModel ---
mutable struct CNNModel <: AbstractModel
    chain::Chain
    optimizer
    epochs::Int
    history::Vector{Float32}
    function CNNModel(input_len::Int; epochs::Int=10)
        # 1. Define the convolutional part of the model
        conv_part = Chain(
            Conv((3,), 1=>4, relu),
            MaxPool((2,)),
            Conv((3,), 4=>8, relu),
            MaxPool((2,)),
            Flux.flatten
        )

        # 2. Determine the output size of the conv part dynamically
        # Create a dummy input tensor of the correct size
        dummy_input = rand(Float32, input_len, 1, 1) # (width, channels, batch)
        conv_output_size = size(conv_part(dummy_input), 1)

        # 3. Define the full chain with the correct dense layer size
        chain = Chain(
            x -> reshape(x, (input_len, 1, size(x, 2))),
            conv_part,
            Dense(conv_output_size => 16, relu),
            Dense(16 => 1)
        )

        optimizer = Adam
        new(chain, optimizer, epochs, [])
    end
end

function fit!(model::CNNModel, data::GenomicData)
    # ... (CNNModel fit! implementation is identical to FNN) ...
    X = Float32.(Matrix(data.genotypes))'
    y = Float32.(data.phenotypes[!, 1]')
    dataset = [(X, y)]
    loss(m, x, y) = Flux.mse(m(x), y)
    opt_state = Flux.setup(model.optimizer(), model.chain)
    println("开始 CNN 模型训练 (轮数: $(model.epochs))...")
    empty!(model.history)
    for epoch in 1:model.epochs
        train!(loss, model.chain, dataset, opt_state)
        current_loss = loss(model.chain, X, y)
        push!(model.history, current_loss)
        if epoch % 10 == 0 || epoch == 1
            println("轮数: $epoch, 损失: $current_loss")
        end
    end
    println("训练完成。")
    return nothing
end

function predict(model::CNNModel, new_data::DataFrame)
    # ... (CNNModel predict implementation is identical to FNN) ...
    X_new = Float32.(Matrix(new_data))'
    return model.chain(X_new)[1, :]
end


end # module DeepLearning
