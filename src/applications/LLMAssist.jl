module LLMAssist

using HTTP
using JSON3
using DataFrames
using Statistics

const LLM_DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
const LLM_STATE = Dict(:api_key => "", :endpoint => LLM_DEFAULT_ENDPOINT, :model => "gpt-4o-mini")

"""
    configure_llm!(; api_key, endpoint = LLM_DEFAULT_ENDPOINT, model = "gpt-4o-mini")

配置用于自动化解释、候选基因筛选与育种方案优化的大语言模型接口。
"""
function configure_llm!(; api_key::AbstractString, endpoint::AbstractString = LLM_DEFAULT_ENDPOINT,
                          model::AbstractString = "gpt-4o-mini")
    LLM_STATE[:api_key] = api_key
    LLM_STATE[:endpoint] = endpoint
    LLM_STATE[:model] = model
    return nothing
end

function llm_request(messages; temperature = 0.2)
    isempty(LLM_STATE[:api_key]) && error("请先使用 configure_llm! 设置 API Key")
    payload = JSON3.write(Dict(
        "model" => LLM_STATE[:model],
        "temperature" => temperature,
        "messages" => messages
    ))
    headers = [
        "Content-Type" => "application/json",
        "Authorization" => "Bearer " * LLM_STATE[:api_key]
    ]
    try
        response = HTTP.post(LLM_STATE[:endpoint], headers, payload)
        json = JSON3.read(String(response.body))
        return json["choices"][1]["message"]["content"]
    catch err
        return "LLM 请求失败: $(err)"
    end
end

"""
    llm_explain_results(rvat, epistasis; trait = "DailyGain")

自动生成中文报告，总结稀有变异和上位性 Meta 分析的关键发现，并提出育种建议。
"""
function llm_explain_results(rvat::DataFrame, epistasis::DataFrame; trait::AbstractString = "DailyGain")
    prompt = """
    你是一位资深家畜遗传育种专家，请根据以下稀有变异与上位性检验结果，
    为性状 $trait 撰写简洁但信息丰富的中文报告。报告需包含：
    1. 研究概览
    2. 主要显著结果（列出前 5 个基因或互作）
    3. 对肉羊/肉牛产肉与繁殖性能的潜在影响
    4. 育种与管理建议
    请突出多组学信息的综合价值。
    """
    messages = [
        Dict("role" => "system", "content" => "You are an expert livestock geneticist."),
        Dict("role" => "user", "content" => prompt *
            "\n稀有变异结果前 5 行:\n" * sprint(show, MIME("text/plain"), first(rvat, min(5, nrow(rvat)))) *
            "\n上位性结果前 5 行:\n" * sprint(show, MIME("text/plain"), first(epistasis, min(5, nrow(epistasis)))))
    ]
    return llm_request(messages)
end

"""
    llm_rank_candidate_genes(annotations; top_k = 10)

利用 LLM 综合文献知识，为候选基因排序并提供理由。
"""
function llm_rank_candidate_genes(annotations::DataFrame; top_k::Int = 10)
    prompt = """
    根据以下候选基因注释信息，结合最新文献，总结与肉羊/肉牛产肉、繁殖性能相关的证据，
    并按重要性排序列出前 $top_k 个基因，给出推荐理由。
    """
    messages = [
        Dict("role" => "system", "content" => "You are an expert in functional genomics."),
        Dict("role" => "user", "content" => prompt * "\n候选基因表:\n" * sprint(show, MIME("text/plain"), annotations))
    ]
    return llm_request(messages; temperature = 0.1)
end

"""
    llm_optimize_scheme(metrics; constraints = Dict())

调用 LLM 综合统计指标，提出育种方案参数优化建议。
"""
function llm_optimize_scheme(metrics::DataFrame; constraints = Dict())
    constraint_text = join(["$k: $v" for (k, v) in constraints], ", ")
    prompt = """
    我们计划优化肉羊/肉牛育种方案，请基于以下指标与约束提出 3 套参数组合，
    并解释各自的优势与风险。请关注多组学整合、稀有变异上位性利用及育种值可靠性。
    约束: $constraint_text
    """
    messages = [
        Dict("role" => "system", "content" => "You are an operations research specialist in animal breeding."),
        Dict("role" => "user", "content" => prompt * "\n评估指标表:\n" * sprint(show, MIME("text/plain"), metrics))
    ]
    return llm_request(messages; temperature = 0.3)
end

end # module
