"""Python 接口示例, 演示如何通过 PyJulia 调用 GenomicPrediction.jl。"""
from typing import Dict

from julia import Julia
from julia import GenomicPrediction

jl = Julia(compiled_modules=False)

def simulate_and_predict(num_individuals: int = 100, num_markers: int = 200) -> Dict[str, float]:
    """调用 Julia 侧函数模拟数据并训练 GBLUP 模型。"""
    dataset = GenomicPrediction.simulate_genomic_data(num_individuals, num_markers)
    model = GenomicPrediction.GBLUPModel(λ=0.8)
    GenomicPrediction.fit!(model, dataset.genotype, dataset.phenotype)
    preds = GenomicPrediction.predict(model, dataset.genotype)
    metrics = GenomicPrediction.evaluate_metrics(dataset.phenotype, preds)
    return dict(metrics)

if __name__ == "__main__":
    print(simulate_and_predict())
