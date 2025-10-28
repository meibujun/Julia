# API 参考

```@meta
CurrentModule = GenomicPrediction
```

## 数据处理

```@docs
GenomicDataset
load_genomic_table
load_phenotype_table
merge_genomic_phenotype
quality_control!
impute_missing!
build_grm
kfold_split
make_holdout_split
simulate_genomic_data
```

## 核心算法

```@docs
PredictionModel
GBLUPModel
RidgeRegressionModel
LassoModel
ElasticNetModel
BayesAModel
BayesBModel
fit!
predict
```

## 深度学习

```@docs
TrainingHistory
build_mlp
build_cnn
build_transformer
train_deep_model!
```

## 评估与 AutoGS

```@docs
evaluate_metrics
compute_auc
cross_validate
summarize_cv
AutoGSPipeline
run_autogs
default_workflow
```

## FAIR 建模

```@docs
ModelMetadata
create_metadata
enrich_metadata!
save_model
load_model
```
