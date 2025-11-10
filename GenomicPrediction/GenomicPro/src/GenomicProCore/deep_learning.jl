# src/GenomicProPredict/deep_learning.jl

using Lux, Optimisers, Random, CUDA, Zygote

abstract type AbstractDeepLearningModel end

struct LocallyConnectedLayer{F} <: Lux.AbstractExplicitLayer
    in_channels::Int
    out_channels::Int
    kernel_size::Int
    activation::F
end

function LocallyConnectedLayer(in_channels::Int, out_channels::Int, kernel_size::Int; activation::F=identity) where {F}
    return LocallyConnectedLayer{F}(in_channels, out_channels, kernel_size, activation)
end

function Lux.initialparameters(rng::AbstractRNG, l::LocallyConnectedLayer)
    weight = randn(rng, l.out_channels, l.in_channels, l.kernel_size)
    bias = zeros(rng, l.out_channels, 1)
    return (weight=weight, bias=bias)
end

function (l::LocallyConnectedLayer)(x, ps, st)
    # Simplified implementation
    # A full implementation would perform a locally connected operation
    return l.activation.(x * ps.weight) .+ ps.bias, st
end

struct DeepGBLUPModel <: AbstractDeepLearningModel
    hidden_layers::Vector{Int}
    activation::Symbol
    dropout_rate::Float64
    learning_rate::Float64
    batch_size::Int
    n_epochs::Int
    early_stopping_patience::Int

    parameters::Dict{Symbol, Any}

    function DeepGBLUPModel(;
                           hidden_layers::Vector{Int} = [512, 256, 128],
                           activation::Symbol = :relu,
                           dropout_rate::Float64 = 0.3,
                           learning_rate::Float64 = 0.001,
                           batch_size::Int = 256,
                           n_epochs::Int = 100,
                           early_stopping_patience::Int = 10)

        parameters = Dict{Symbol, Any}()
        new(hidden_layers, activation, dropout_rate, learning_rate, batch_size, n_epochs, early_stopping_patience, parameters)
    end
end

function train_deep_gblup!(model::DeepGBLUPModel,
                          genotypes_train::AbstractGenotypeData,
                          phenotypes_train::Vector{Float64},
                          genotypes_val::AbstractGenotypeData,
                          phenotypes_val::Vector{Float64};
                          verbose::Bool = true,
                          use_gpu::Bool = true)

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    n_train = length(phenotypes_train)
    n_markers = size(genotypes_train, 2)

    # GBLUP component
    G = compute_grm(genotypes_train)
    vc = estimate_variance_components(G, phenotypes_train)
    λ = vc.residual_variance / vc.genetic_variance
    gblup_results = solve_gblup(G, phenotypes_train, λ)
    u_gblup = gblup_results.breeding_values

    # NN component
    nn_model = build_nn_architecture(model, n_markers)
    ps, st = Lux.setup(rng, nn_model)

    if use_gpu && CUDA.functional()
        ps = ps |> gpu
        st = st |> gpu
        device = gpu
    else
        device = cpu
    end

    opt_state = Optimisers.setup(Adam(model.learning_rate), ps)

    best_val_loss = Inf
    patience_counter = 0

    # Training loop
    for epoch in 1:model.n_epochs
        # Mini-batch training
        for batch_indices in Iterators.partition(1:n_train, model.batch_size)
            X_batch = genotypes_train[batch_indices, :] |> device
            y_batch = phenotypes_train[batch_indices] |> device
            u_gblup_batch = u_gblup[batch_indices] |> device

            loss, grads = Zygote.withgradient(ps) do p
                y_pred_nn, _ = nn_model(X_batch, p, st)
                y_pred_total = y_pred_nn + u_gblup_batch
                mse(y_pred_total, y_batch)
            end

            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        end

        # Validation
        y_pred_val_nn, _ = nn_model(genotypes_val, ps, st)
        y_pred_val_total = y_pred_val_nn + predict_from_relatives(compute_grm(genotypes_val, genotypes_train), G, u_gblup)
        val_loss = mse(y_pred_val_total, phenotypes_val)

        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
        else
            patience_counter += 1
        end

        if patience_counter >= model.early_stopping_patience
            break
        end
    end

    model.parameters[:ps] = ps
    model.parameters[:st] = st
    model.parameters[:u_gblup] = u_gblup
end

function build_nn_architecture(model::DeepGBLUPModel, input_dim::Int)
    layers = [
        LocallyConnectedLayer(input_dim, 32, 1, activation=relu),
        FlattenLayer(),
        Dense(32 * input_dim, 128, relu),
        Dense(128, 1)
    ]
    return Chain(layers...)
end

function predict_deep_gblup(model::DeepGBLUPModel, genotypes_test::AbstractGenotypeData, genotypes_train::AbstractGenotypeData)
    ps = model.parameters[:ps]
    st = model.parameters[:st]
    u_gblup = model.parameters[:u_gblup]
    nn_model = build_nn_architecture(model, size(genotypes_test, 2))

    # GBLUP prediction for test set
    G_test_train = compute_grm(genotypes_test, genotypes_train)
    G_train = compute_grm(genotypes_train)
    u_gblup_test = predict_from_relatives(G_test_train, G_train, u_gblup)

    # NN prediction
    y_pred_nn, _ = nn_model(genotypes_test, ps, st)

    return y_pred_nn + u_gblup_test
end
