#' R 接口示例: 通过 JuliaCall 调用 GenomicPrediction.jl
#'
#' 运行前请在 R 中执行:
#' install.packages("JuliaCall")
#' JuliaCall::julia_setup()

library(JuliaCall)

julia_setup()
julia_library("GenomicPrediction")

simulate_and_predict <- function(num_individuals = 100L, num_markers = 200L) {
  dataset <- julia_call("simulate_genomic_data", num_individuals, num_markers)
  model <- julia_eval("GBLUPModel(λ = 0.8)")
  julia_call("fit!", model, dataset$genotype, dataset$phenotype)
  preds <- julia_call("predict", model, dataset$genotype)
  metrics <- julia_call("evaluate_metrics", dataset$phenotype, preds)
  return(metrics)
}

print(simulate_and_predict())
