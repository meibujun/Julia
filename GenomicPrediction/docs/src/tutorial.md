# 快速入门教程

本教程将引导您完成使用 `GenomicPrediction.jl` 进行一次完整基因组预测分析的基本步骤。

## 1. 加载数据

首先，我们需要加载基因型和表型数据。数据应为 CSV 格式，第一列为个体 ID。

```julia
using GenomicPrediction, DataFrames, Random

# 创建一些模拟数据
Random.seed!(42)
geno = DataFrame(hcat(1:100, rand(0:2, 100, 50)), :auto)
pheno = DataFrame(ID=1:100, y=rand(100) * 10)
CSV.write("geno.csv", geno)
CSV.write("pheno.csv", pheno)

# 从 CSV 文件加载数据
data = load_csv("geno.csv", "pheno.csv")
```

## 2. 选择和训练模型

接下来，我们选择一个模型进行训练。这里我们使用 GBLUP 模型。`fit!` 函数会就地训练模型。

```julia
# 初始化 GBLUP 模型，lambda 是方差比
model = GBLUPModel(10.0)

# 训练模型
fit!(model, data)
```

## 3. 进行预测

训练完成后，我们可以使用 `predict` 函数来预测新个体的表型值。

```julia
# 创建一些新的基因型数据用于预测
new_geno = DataFrame(hcat(101:105, rand(0:2, 5, 50)), :auto)

# 进行预测
predictions = predict(model, new_geno)
println("预测值: ", predictions)
```

## 4. 评估模型性能

为了了解模型的性能，我们可以使用 k-折交叉验证。

```julia
# 定义一个 GBLUP 模型原型
model_prototype = GBLUPModel(10.0)

# 执行 5-折交叉验证
cv_results = cross_validate(model_prototype, data; k=5)

println("平均准确性 (相关系数): ", cv_results.mean_accuracy)
println("平均均方误差: ", cv_results.mean_mse)
```

## 5. 保存和加载模型

训练好的模型可以保存到磁盘，以便将来重用。

```julia
# 保存模型
save_model(model, "gblup_model.bson")

# 加载模型
loaded_model = load_model("gblup_model.bson")

# 使用加载的模型进行预测
new_predictions = predict(loaded_model, new_geno)
@assert predictions == new_predictions
```

这是一个基本的工作流程。您可以尝试使用 `BayesAModel`、`CNNModel` 等其他模型，或者使用 `grid_search` 进行自动超参数调优。
