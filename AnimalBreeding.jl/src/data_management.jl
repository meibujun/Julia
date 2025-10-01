# --- 数据管理模块 ---
# 该模块负责所有数据的导入、验证、预处理和统一管理。
# 它是整个系统的入口，确保后续分析模块获得高质量、标准化的数据。

"""
    DataManager

一个可变结构体(mutable struct)，用于存储遗传评估所需的所有数据。
它扮演一个中央数据容器的角色，集中管理谱系、基因型、表型数据以及计算出的关系矩阵。

# 字段
- `pedigree::DataFrame`: 存储谱系信息的DataFrame。应包含 `animal`, `sire`, `dam` 列。
- `genotypes::DataFrame`: 存储基因型信息的DataFrame。第一列应为动物ID，其余列为SNP标记。
- `phenotypes::DataFrame`: 存储表型信息的DataFrame。包含动物ID、性状记录和固定效应因子。
- `A::SparseMatrixCSC{Float64, Int}`: 谱系关系矩阵 (A矩阵)，以稀疏矩阵格式存储。
- `G::Matrix{Float64}`: 基因组关系矩阵 (G矩阵)。
- `H::Matrix{Float64}`: 单步法关系矩阵 (H矩阵)。

# 构造函数
- `DataManager()`: 创建一个空的DataManager实例，所有字段初始化为空的DataFrame或未定义。
"""
mutable struct DataManager
    pedigree::DataFrame
    genotypes::DataFrame
    phenotypes::DataFrame
    A::SparseMatrixCSC{Float64, Int} # 谱系关系矩阵
    G::Matrix{Float64}               # 基因组关系矩阵
    H::Matrix{Float64}               # 单步关系矩阵

    # 默认构造函数，初始化为空
    DataManager() = new(DataFrame(), DataFrame(), DataFrame())
end

"""
    load_pedigree(filepath::String) -> DataFrame

从CSV文件加载谱系数据。

该函数读取标准的谱系文件，文件必须包含 `animal` (个体ID), `sire` (父本ID), 和 `dam` (母本ID) 三列。
缺失的父本或母本信息应使用 `0`, `NA`, 或空字符串表示，函数会自动将其处理为`missing`。

# 参数
- `filepath::String`: 谱系数据文件的路径。

# 返回
- `DataFrame`: 一个包含谱系数据的DataFrame对象。

# 异常
- 如果文件路径无效或文件格式不正确（如缺少必要的列），将抛出错误。
"""
function load_pedigree(filepath::String)
    try
        # 读取CSV文件，并将指定的字符串视作缺失值
        ped = CSV.read(filepath, DataFrame; missingstrings=["", "NA", "0"])
        # 将所有列名转换为小写，以实现标准化
        rename!(ped, lowercase.(names(ped)))
        # 检查必需的列是否存在
        required_cols = ["animal", "sire", "dam"]
        if !all(col -> col in names(ped), required_cols)
            error("谱系文件必须包含 'animal', 'sire', 'dam' 列。")
        end
        println("谱系文件 '$filepath' 加载成功。")
        return ped
    catch e
        error("加载谱系文件 '$filepath' 失败。原因: $e")
    end
end

"""
    load_genotypes(filepath::String; animal_id_col::String="animal") -> DataFrame

从CSV文件加载基因型数据。

函数假定文件的第一列是个体ID，其余列是SNP标记数据。
基因型通常编码为 `0`, `1`, `2`，代表三种基因型。

# 参数
- `filepath::String`: 基因型数据文件的路径。
- `animal_id_col::String`: 指定个体ID所在的列名，默认为 "animal"。

# 返回
- `DataFrame`: 一个包含基因型数据的DataFrame对象。

# 异常
- 如果文件路径无效，将抛出错误。
"""
function load_genotypes(filepath::String; animal_id_col::String="animal")
    try
        geno = CSV.read(filepath, DataFrame)
        # 将用户指定的动物ID列名重命名为标准的 'animal'
        rename!(geno, Symbol(animal_id_col) => :animal, makeunique=true)
        println("基因型文件 '$filepath' 加载成功。")
        return geno
    catch e
        error("加载基因型文件 '$filepath' 失败。原因: $e")
    end
end

"""
    load_phenotypes(filepath::String; trait_cols::Vector{String}, fixed_cols::Vector{String}=String[]) -> DataFrame

从CSV文件加载表型数据。

此函数不仅加载数据，还允许用户指定哪些列是需要分析的“性状”，哪些是模型中的“固定效应”。
文件必须包含一个 `animal` 列，用于关联个体。

# 参数
- `filepath::String`: 表型数据文件的路径。
- `trait_cols::Vector{String}`: 一个字符串向量，包含所有性状列的名称。
- `fixed_cols::Vector{String}`: 一个可选的字符串向量，包含所有固定效应列的名称。

# 返回
- `DataFrame`: 一个包含表型数据的DataFrame对象，其元数据中标注了性状和固定效应列。

# 异常
- 如果文件路径无效或缺少 `animal` 列，将抛出错误。
"""
function load_phenotypes(filepath::String; trait_cols::Vector{String}, fixed_cols::Vector{String}=String[])
    try
        pheno = CSV.read(filepath, DataFrame)
        rename!(pheno, lowercase.(names(pheno)))
        if "animal" ∉ names(pheno)
            error("表型文件必须包含 'animal' 列。")
        end

        # 将性状和固定效应的列名信息存储在DataFrame的元数据中，方便后续模块调用
        metadata!(pheno, "trait_cols", trait_cols, style=:note)
        metadata!(pheno, "fixed_cols", fixed_cols, style=:note)

        println("表型文件 '$filepath' 加载成功。")
        return pheno
    catch e
        error("加载表型文件 '$filepath' 失败。原因: $e")
    end
end


"""
    validate_data(dm::DataManager) -> Bool

对加载到DataManager中的所有数据进行一致性验证。

这是一个关键的质量控制步骤，确保不同数据源之间的ID能够对应，并且数据结构完整、有效。
主要检查内容包括：
- 核心数据（谱系、表型）是否已加载。
- 表型和基因型数据中的个体是否都能在谱系中找到。
- 谱系结构是否有效（例如，个体不能是自己的亲本）。

# 参数
- `dm::DataManager`: 包含所有待验证数据的DataManager对象。

# 返回
- `Bool`: 如果所有检查通过，返回 `true`。否则，打印警告或错误信息，并最终抛出错误。

# 异常
- 如果发现严重的数据不一致问题，将抛出错误，终止后续分析。
"""
function validate_data(dm::DataManager)
    println("--- 开始数据验证 ---")
    valid = true

    # 检查核心数据是否存在
    if isempty(dm.pedigree)
        @error "谱系数据是必需的，但尚未加载。"
        valid = false
    end
    if isempty(dm.phenotypes)
        @error "表型数据是必需的，但尚未加载。"
        valid = false
    end
    # 如果核心数据缺失，则无法继续验证
    if !valid
        error("数据验证失败：缺少核心数据文件。")
    end

    ped_animals = Set(dm.pedigree.animal)
    pheno_animals = Set(dm.phenotypes.animal)

    # 检查1：表型数据中的个体是否都能在谱系中找到
    missing_in_ped = setdiff(pheno_animals, ped_animals)
    if !isempty(missing_in_ped)
        @warn "$(length(missing_in_ped)) 个体存在于表型数据中，但在谱系中未找到。这些个体将被从分析中忽略。"
        # 注意：这里也可以选择报错，取决于业务需求。当前设为警告。
    end

    # 检查2：谱系结构完整性
    for row in eachrow(dm.pedigree)
        # ismissing 在这里用于处理可能的 missing 值
        if !ismissing(row.sire) && row.animal == row.sire
            @error "谱系错误：个体 $(row.animal) 不能是自己的父本。"
            valid = false
        end
        if !ismissing(row.dam) && row.animal == row.dam
            @error "谱系错误：个体 $(row.animal) 不能是自己的母本。"
            valid = false
        end
    end

    # 检查3：如果存在基因型数据，检查其个体是否在谱系中
    if !isempty(dm.genotypes)
        geno_animals = Set(dm.genotypes.animal)
        missing_in_ped_geno = setdiff(geno_animals, ped_animals)
        if !isempty(missing_in_ped_geno)
            # 这通常是允许的，这些动物会被当作基础群体成员
            @warn "$(length(missing_in_ped_geno)) 个基因分型个体不在谱系中。在构建A矩阵时，它们将被视为无亲本的基础动物。"
        end
    end

    if valid
        println("数据验证成功：数据一致，准备进行分析。")
    else
        error("数据验证失败。请检查上面的错误和警告信息。")
    end
    println("--- 数据验证完成 ---")
    return true
end