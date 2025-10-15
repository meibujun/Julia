module Threading

using Base.Threads
using LinearAlgebra

"""
    threaded_map(f, collection; chunksize = nothing)

在多线程环境下对集合执行映射操作，自动分块以适应 90K 级别 SNP 规模。
"""
function threaded_map(f, collection; chunksize = nothing)
    n = length(collection)
    n == 0 && return similar(collection, 0)
    chunksize = isnothing(chunksize) ? max(1, fld(n, nthreads())) : chunksize
    result = Vector{Any}(undef, n)
    @threads for i in 1:n
        result[i] = f(collection[i])
    end
    return result
end

"""
    chunk_indices(total, chunks)

生成分块索引用于手动切片，支撑高性能矩阵运算。
"""
function chunk_indices(total::Integer, chunks::Integer)
    step = cld(total, chunks)
    ranges = Vector{UnitRange{Int}}()
    start = 1
    while start <= total
        stop = min(total, start + step - 1)
        push!(ranges, start:stop)
        start = stop + 1
    end
    return ranges
end

end # module
