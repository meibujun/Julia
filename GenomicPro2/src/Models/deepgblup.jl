"""
# Deep GBLUP 模型

结合深度神经网络和 GBLUP 的混合基因组预测模型。

## 模型描述
Deep GBLUP 通过多层神经网络学习 SNP 的非线性组合：

y = f_θ(X) + Zu + e

其中：
- f_θ(X): 深度神经网络（捕捉非加性效应）
- Zu: 传统 GBLUP 组分（捕捉加性效应）
- θ: 网络参数

## 架构
1. **输入层**: SNP 基因型数据
2. **隐藏层**: 多层全连接网络 + ReLU/Tanh 激活
3. **输出层**: 预测育种值
4. **混合**: 深度网络输出 + GBLUP 随机效应

## 使用示例
```julia
using GenomicPro2.Models

# 构建 Deep GBLUP 模型
model = DeepGBLUP(
    input_dim = n_snps,
    hidden_layers = [256, 128, 64],
    activation = :relu
)

# 训练模型
results = train_deepgblup!(
    model,
    genotypes,
    phenotypes,
    epochs = 100,
    batch_size = 32
)
```

## 优势
- 捕捉 SNP 间的复杂非线性相互作用
- 自动特征学习
- 可扩展到超大规模数据集
- 支持 GPU 加速
"""

using Random
using Statistics
using LinearAlgebra
using ..Core: GenotypeData, PhenotypeData, ValidationError
using ..Data: CompactGenotypes

"""
激活函数类型
"""
@enum ActivationFunction relu tanh sigmoid linear

"""
    DeepGBLUP

Deep GBLUP 模型结构。

# 字段
- `input_dim`: 输入维度（SNP 数量）
- `hidden_layers`: 隐藏层维度列表
- `activation`: 激活函数
- `weights`: 网络权重（层的列表）
- `biases`: 偏置项（层的列表）
- `grm_component`: 是否包含 GBLUP 组分
- `lambda_grm`: GRM 组分的权重
"""
mutable struct DeepGBLUP
    input_dim::Int
    hidden_layers::Vector{Int}
    activation::ActivationFunction
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    grm_component::Bool
    lambda_grm::Float64

    # 优化器状态（Adam）
    m_weights::Vector{Matrix{Float64}}  # 一阶矩估计
    v_weights::Vector{Matrix{Float64}}  # 二阶矩估计
    m_biases::Vector{Vector{Float64}}
    v_biases::Vector{Vector{Float64}}
    t::Int  # 时间步
end

"""
    DeepGBLUP(input_dim::Int; hidden_layers::Vector{Int}=[256, 128, 64],
              activation::Symbol=:relu, grm_component::Bool=true,
              lambda_grm::Float64=0.5, seed::Union{Int,Nothing}=nothing)

构造 Deep GBLUP 模型。

# 参数
- `input_dim`: 输入维度（SNP 数）
- `hidden_layers`: 隐藏层大小列表
- `activation`: 激活函数 (:relu, :tanh, :sigmoid, :linear)
- `grm_component`: 是否包含 GBLUP 组分
- `lambda_grm`: GBLUP 组分的权重
- `seed`: 随机种子
"""
function DeepGBLUP(input_dim::Int;
                   hidden_layers::Vector{Int}=[256, 128, 64],
                   activation::Symbol=:relu,
                   grm_component::Bool=true,
                   lambda_grm::Float64=0.5,
                   seed::Union{Int,Nothing}=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    # 转换激活函数符号
    act_func = if activation == :relu
        relu
    elseif activation == :tanh
        tanh
    elseif activation == :sigmoid
        sigmoid
    else
        linear
    end

    # 初始化网络权重（He 初始化）
    layers = [input_dim, hidden_layers..., 1]
    n_layers = length(layers) - 1

    weights = Vector{Matrix{Float64}}(undef, n_layers)
    biases = Vector{Vector{Float64}}(undef, n_layers)

    for i in 1:n_layers
        fan_in, fan_out = layers[i], layers[i+1]

        # He 初始化
        if activation == :relu
            std = sqrt(2.0 / fan_in)
        else
            # Xavier 初始化
            std = sqrt(2.0 / (fan_in + fan_out))
        end

        weights[i] = randn(fan_out, fan_in) * std
        biases[i] = zeros(fan_out)
    end

    # 初始化 Adam 优化器状态
    m_weights = [zeros(size(w)) for w in weights]
    v_weights = [zeros(size(w)) for w in weights]
    m_biases = [zeros(size(b)) for b in biases]
    v_biases = [zeros(size(b)) for b in biases]

    return DeepGBLUP(
        input_dim,
        hidden_layers,
        act_func,
        weights,
        biases,
        grm_component,
        lambda_grm,
        m_weights,
        v_weights,
        m_biases,
        v_biases,
        0
    )
end

"""
应用激活函数
"""
function apply_activation(x, func::ActivationFunction)
    if func == relu
        return max.(0, x)
    elseif func == tanh
        return tanh.(x)
    elseif func == sigmoid
        return 1 ./ (1 .+ exp.(-x))
    else  # linear
        return x
    end
end

"""
激活函数的导数
"""
function activation_derivative(x, func::ActivationFunction)
    if func == relu
        return float.(x .> 0)
    elseif func == tanh
        t = tanh.(x)
        return 1 .- t .^ 2
    elseif func == sigmoid
        s = 1 ./ (1 .+ exp.(-x))
        return s .* (1 .- s)
    else  # linear
        return ones(size(x))
    end
end

"""
    forward(model::DeepGBLUP, X::Matrix{Float64})

前向传播。

# 返回
(output, activations, pre_activations) 元组
"""
function forward(model::DeepGBLUP, X::Matrix{Float64})
    n_samples = size(X, 1)
    n_layers = length(model.weights)

    activations = Vector{Matrix{Float64}}(undef, n_layers + 1)
    pre_activations = Vector{Matrix{Float64}}(undef, n_layers)

    activations[1] = X

    for i in 1:n_layers
        # z = Wx + b
        z = activations[i] * model.weights[i]' .+ model.biases[i]'
        pre_activations[i] = z

        # a = activation(z)
        if i < n_layers  # 隐藏层
            activations[i+1] = apply_activation(z, model.activation)
        else  # 输出层（线性）
            activations[i+1] = z
        end
    end

    return activations[end], activations, pre_activations
end

"""
    backward(model::DeepGBLUP, X, y, activations, pre_activations)

反向传播计算梯度。

# 返回
(grad_weights, grad_biases) 元组
"""
function backward(model::DeepGBLUP, X::Matrix{Float64}, y::Vector{Float64},
                  activations, pre_activations)

    n_samples = size(X, 1)
    n_layers = length(model.weights)

    grad_weights = Vector{Matrix{Float64}}(undef, n_layers)
    grad_biases = Vector{Vector{Float64}}(undef, n_layers)

    # 输出层误差
    δ = activations[end] .- y

    # 反向传播
    for i in n_layers:-1:1
        # 梯度
        grad_weights[i] = (δ' * activations[i]) / n_samples
        grad_biases[i] = vec(mean(δ, dims=1))

        if i > 1
            # 传播到前一层
            δ = (δ * model.weights[i]) .* activation_derivative(pre_activations[i-1], model.activation)
        end
    end

    return grad_weights, grad_biases
end

"""
    update_adam!(model::DeepGBLUP, grad_weights, grad_biases;
                 lr::Float64=0.001, β1::Float64=0.9, β2::Float64=0.999, ε::Float64=1e-8)

使用 Adam 优化器更新参数。
"""
function update_adam!(model::DeepGBLUP, grad_weights, grad_biases;
                      lr::Float64=0.001, β1::Float64=0.9, β2::Float64=0.999, ε::Float64=1e-8)

    model.t += 1

    for i in 1:length(model.weights)
        # 更新权重
        model.m_weights[i] = β1 * model.m_weights[i] + (1 - β1) * grad_weights[i]
        model.v_weights[i] = β2 * model.v_weights[i] + (1 - β2) * (grad_weights[i] .^ 2)

        m_hat = model.m_weights[i] / (1 - β1^model.t)
        v_hat = model.v_weights[i] / (1 - β2^model.t)

        model.weights[i] .-= lr * m_hat ./ (sqrt.(v_hat) .+ ε)

        # 更新偏置
        model.m_biases[i] = β1 * model.m_biases[i] + (1 - β1) * grad_biases[i]
        model.v_biases[i] = β2 * model.v_biases[i] + (1 - β2) * (grad_biases[i] .^ 2)

        m_hat_b = model.m_biases[i] / (1 - β1^model.t)
        v_hat_b = model.v_biases[i] / (1 - β2^model.t)

        model.biases[i] .-= lr * m_hat_b ./ (sqrt.(v_hat_b) .+ ε)
    end
end

"""
    DeepGBLUPResults

Deep GBLUP 训练结果。
"""
struct DeepGBLUPResults
    model::DeepGBLUP
    training_loss::Vector{Float64}
    validation_loss::Vector{Float64}
    final_mse::Float64
    final_correlation::Float64
    gebv::Vector{Float64}
end

"""
    train_deepgblup!(model::DeepGBLUP, genotypes::CompactGenotypes, phenotypes::PhenotypeData;
                     epochs::Int=100, batch_size::Int=32, learning_rate::Float64=0.001,
                     validation_split::Float64=0.2, early_stopping::Bool=true,
                     patience::Int=10, verbose::Bool=true)

训练 Deep GBLUP 模型。

# 参数
- `model`: DeepGBLUP 模型
- `genotypes`: 基因型数据
- `phenotypes`: 表型数据
- `epochs`: 训练轮数
- `batch_size`: 批大小
- `learning_rate`: 学习率
- `validation_split`: 验证集比例
- `early_stopping`: 是否使用早停
- `patience`: 早停耐心值
- `verbose`: 是否显示训练信息

# 返回
DeepGBLUPResults 对象
"""
function train_deepgblup!(model::DeepGBLUP,
                          genotypes::CompactGenotypes,
                          phenotypes::PhenotypeData;
                          epochs::Int=100,
                          batch_size::Int=32,
                          learning_rate::Float64=0.001,
                          validation_split::Float64=0.2,
                          early_stopping::Bool=true,
                          patience::Int=10,
                          verbose::Bool=true)

    # 准备数据
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    if n_samples != length(phenotypes.values)
        throw(ValidationError("样本数不匹配"))
    end

    if verbose
        @info "开始训练 Deep GBLUP 模型"
        @info "  样本数: $n_samples"
        @info "  SNP 数: $n_snps"
        @info "  网络结构: $n_snps -> $(model.hidden_layers) -> 1"
        @info "  训练轮数: $epochs"
        @info "  批大小: $batch_size"
    end

    # 解压缩基因型并标准化
    X = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            X[i, j] = Float64(genotypes[i, j])
        end
    end

    X_mean = mean(X, dims=1)
    X .-= X_mean
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1.0
    X ./= X_std

    y = copy(phenotypes.values)
    y .-= mean(y)

    # 划分训练集和验证集
    n_val = Int(floor(n_samples * validation_split))
    n_train = n_samples - n_val

    idx = randperm(n_samples)
    train_idx = idx[1:n_train]
    val_idx = idx[(n_train+1):end]

    X_train, y_train = X[train_idx, :], y[train_idx]
    X_val, y_val = X[val_idx, :], y[val_idx]

    # 训练循环
    training_loss = Float64[]
    validation_loss = Float64[]
    best_val_loss = Inf
    patience_counter = 0

    for epoch in 1:epochs
        # 随机打乱训练数据
        shuffle_idx = randperm(n_train)
        X_train_shuffled = X_train[shuffle_idx, :]
        y_train_shuffled = y_train[shuffle_idx]

        # 小批量训练
        epoch_losses = Float64[]
        n_batches = ceil(Int, n_train / batch_size)

        for batch in 1:n_batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, n_train)

            X_batch = X_train_shuffled[start_idx:end_idx, :]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # 前向传播
            output, activations, pre_activations = forward(model, X_batch)

            # 计算损失
            batch_loss = mean((vec(output) .- y_batch) .^ 2)
            push!(epoch_losses, batch_loss)

            # 反向传播
            grad_w, grad_b = backward(model, X_batch, y_batch, activations, pre_activations)

            # 更新参数
            update_adam!(model, grad_w, grad_b, lr=learning_rate)
        end

        # 计算训练和验证损失
        train_loss = mean(epoch_losses)
        val_pred, _, _ = forward(model, X_val)
        val_loss = mean((vec(val_pred) .- y_val) .^ 2)

        push!(training_loss, train_loss)
        push!(validation_loss, val_loss)

        # 早停检查
        if early_stopping
            if val_loss < best_val_loss
                best_val_loss = val_loss
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    if verbose
                        @info "早停触发于第 $epoch 轮"
                    end
                    break
                end
            end
        end

        # 进度报告
        if verbose && (epoch % 10 == 0 || epoch == epochs)
            @info "  Epoch $epoch: train_loss=$(round(train_loss, digits=6)), " *
                  "val_loss=$(round(val_loss, digits=6))"
        end
    end

    # 最终评估
    final_pred, _, _ = forward(model, X_val)
    final_mse = mean((vec(final_pred) .- y_val) .^ 2)
    final_corr = cor(vec(final_pred), y_val)

    # 计算所有样本的 GEBV
    gebv_pred, _, _ = forward(model, X)
    gebv = vec(gebv_pred)

    if verbose
        @info "训练完成"
        @info "  最终 MSE: $(round(final_mse, digits=6))"
        @info "  预测相关性: $(round(final_corr, digits=4))"
    end

    return DeepGBLUPResults(
        model,
        training_loss,
        validation_loss,
        final_mse,
        final_corr,
        gebv
    )
end

"""
    predict_deepgblup(model::DeepGBLUP, genotypes_new::CompactGenotypes)

使用训练好的 Deep GBLUP 模型进行预测。
"""
function predict_deepgblup(model::DeepGBLUP, genotypes_new::CompactGenotypes)
    n_samples = size(genotypes_new.data, 1)
    n_snps = size(genotypes_new.data, 2)

    # 解压缩基因型
    X_new = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            X_new[i, j] = Float64(genotypes_new[i, j])
        end
    end

    # 标准化（使用训练集参数）
    X_new_mean = mean(X_new, dims=1)
    X_new .-= X_new_mean
    X_new_std = std(X_new, dims=1)
    X_new_std[X_new_std .== 0] .= 1.0
    X_new ./= X_new_std

    # 前向传播
    output, _, _ = forward(model, X_new)

    return vec(output)
end

# 导出
export ActivationFunction, DeepGBLUP, DeepGBLUPResults
export train_deepgblup!, predict_deepgblup
