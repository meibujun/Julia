# src/GenomicProMultiOmics/fusion.jl

using Lux, Random, NNlib

struct CrossAttentionFusion <: Lux.AbstractExplicitLayer
    d_model::Int
    n_heads::Int
end

function CrossAttentionFusion(d_model::Int; n_heads::Int=8)
    return CrossAttentionFusion(d_model, n_heads)
end

function Lux.initialparameters(rng::AbstractRNG, l::CrossAttentionFusion)
    d_head = l.d_model ÷ l.n_heads
    return (
        query = Lux.Dense(l.d_model => l.d_model),
        key = Lux.Dense(l.d_model => l.d_model),
        value = Lux.Dense(l.d_model => l.d_model),
        out = Lux.Dense(l.d_model => l.d_model)
    )
end

function (l::CrossAttentionFusion)(x, y, ps, st)
    d_head = l.d_model ÷ l.n_heads

    q = ps.query(x)
    k = ps.key(y)
    v = ps.value(y)

    q = reshape(q, d_head, l.n_heads, :)
    k = reshape(k, d_head, l.n_heads, :)
    v = reshape(v, d_head, l.n_heads, :)

    scores = batched_mul(permutedims(k, (2, 1, 3)), q) ./ sqrt(d_head)
    attention_weights = softmax(scores, dims=1)

    attention_output = batched_mul(v, attention_weights)
    attention_output = reshape(attention_output, l.d_model, :)

    return ps.out(attention_output), st
end
