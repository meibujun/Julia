# ============================================================================
# 数据模块 - 基因型数据处理
# AnimalBreeding.jl
# ============================================================================

"""
    load_genotypes(filepath::String; format::Symbol=:csv, quality_control::Bool=true) -> DataFrame

从文件加载基因型数据。

# 参数
- `filepath::String`: 基因型文件的路径。
- `format::Symbol`: 文件格式，目前支持 `:csv`。未来可扩展至 `:plink`, `:vcf`。
- `quality_control::Bool`: 是否执行基础的质量控制。

# 返回
- `DataFrame`: 包含基因型数据的数据框，第一列应为动物ID。
"""
function load_genotypes(filepath::String; format::Symbol=:csv, quality_control::Bool=true)
    @info "加载基因型数据: $filepath (格式: $format)"

    if format == :csv
        genotypes = CSV.read(filepath, DataFrame)

        # 标准化动物ID列名
        id_col_name = names(genotypes)[1]
        if !(id_col_name in ["animal_id", "id", "animal"])
            @warn "将基因型文件的第一列 '$id_col_name' 视为动物ID。"
        end
        rename!(genotypes, id_col_name => "animal_id")

        n_animals = nrow(genotypes)
        n_markers = ncol(genotypes) - 1

        @info "成功加载 $n_animals 个个体的 $n_markers 个SNP标记。"

        if quality_control
            genotypes = perform_genotype_qc(genotypes)
        end

        return genotypes
    else
        error("不支持的基因型文件格式: $format")
    end
end

"""
    perform_genotype_qc(genotypes::DataFrame;
                        min_maf::Float64=0.01,
                        min_call_rate::Float64=0.90) -> DataFrame

对基因型数据执行质量控制。

# 质控步骤
1.  **标记检出率 (Marker Call Rate)**: 移除检出率低于阈值的标记。
2.  **次等位基因频率 (MAF)**: 移除MAF低于阈值的标记。
3.  **个体检出率 (Individual Call Rate)**: （未来可添加）移除检出率过低的个体。

# 参数
- `genotypes::DataFrame`: 原始基因型数据。
- `min_maf::Float64`: 最小允许的MAF。
- `min_call_rate::Float64`: 最小允许的标记检出率。

# 返回
- `DataFrame`: 经过质量控制后的基因型数据。
"""
function perform_genotype_qc(genotypes::DataFrame;
                            min_maf::Float64=0.01,
                            min_call_rate::Float64=0.90)

    @info "执行基因型质量控制 (MAF > $min_maf, Call Rate > $min_call_rate)..."

    n_markers_before = ncol(genotypes) - 1

    # 提取标记矩阵
    marker_matrix = Matrix(genotypes[:, 2:end])

    # 计算每个标记的统计数据
    mafs = Float64[]
    call_rates = Float64[]

    for j in 1:n_markers_before
        col = marker_matrix[:, j]
        non_missing = .!ismissing.(col)
        call_rate = mean(non_missing)
        push!(call_rates, call_rate)

        if call_rate > 0
            # 假设编码为0, 1, 2
            freq = mean(skipmissing(col)) / 2.0
            push!(mafs, min(freq, 1 - freq))
        else
            push!(mafs, 0.0)
        end
    end

    # 筛选标记
    markers_to_keep = (mafs .>= min_maf) .& (call_rates .>= min_call_rate)

    n_markers_after = sum(markers_to_keep)
    n_removed = n_markers_before - n_markers_after

    if n_removed > 0
        @info "移除了 $n_removed 个不符合质量控制标准的标记。"

        # +2 是因为第一列是ID，而markers_to_keep的索引是从1开始的
        indices_to_keep = [1; findall(markers_to_keep) .+ 1]
        return genotypes[:, indices_to_keep]
    else
        @info "所有标记均通过质量控制。"
        return genotypes
    end
end