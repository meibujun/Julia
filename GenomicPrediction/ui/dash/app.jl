# ui/dash/app.jl
#
# A web-based interactive dashboard for GenomicPrediction.jl using Dash.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..")) # Activate the project environment

using Dash
using DashCoreComponents
using DashHtmlComponents
using DashTable
using Base64
using CSV
using DataFrames
using Plots
using GenomicPrediction

# --- Dash App Initialization ---
app = dash("GenomicPrediction.jl Dashboard")

# --- App Layout ---
app.layout = html_div() do
    html_h1("GenomicPrediction.jl - 交互式仪表盘", style=Dict("textAlign" => "center")),

    html_hr(),

    # -- Data Upload Section --
    html_h2("1. 上传数据"),
    dcc_upload(
        id="upload-geno",
        children=html_div(["拖放或 ", html_a("选择基因型文件 (CSV)")])
    ),
    dcc_upload(
        id="upload-pheno",
        children=html_div(["拖放或 ", html_a("选择表型文件 (CSV)")])
    ),
    html_div(id="output-data-upload"),

    html_hr(),

    # -- Model Configuration Section --
    html_h2("2. 模型配置"),
    html_div() do
        html_label("GBLUP 正则化参数 (Lambda):"),
        dcc_slider(
            id="lambda-slider",
            min=0, max=500, step=5,
            value=50,
            marks=Dict(i => string(i) for i in 0:50:500)
        )
    end,

    html_hr(),

    # -- Run and Results Section --
    html_h2("3. 运行分析并查看结果"),
    html_button("运行 GBLUP 分析", id="run-button", n_clicks=0),
    dcc_loading(id="loading-spinner", children=html_div(id="results-output"), type="circle")
end

# --- Callbacks ---

# Callback to show uploaded file names
callback!(
    app,
    Output("output-data-upload", "children"),
    Input("upload-geno", "filename"),
    Input("upload-pheno", "filename"),
) do geno_name, pheno_name
    if isnothing(geno_name) && isnothing(pheno_name)
        return "请上传文件。"
    end
    return html_ul([
        html_li("基因型文件: $(isnothing(geno_name) ? \"未上传\" : geno_name)"),
        html_li("表型文件: $(isnothing(pheno_name) ? \"未上传\" : pheno_name)")
    ])
end

# Callback to run the analysis
callback!(
    app,
    Output("results-output", "children"),
    Input("run-button", "n_clicks"),
    State("upload-geno", "contents"),
    State("upload-pheno", "contents"),
    State("lambda-slider", "value"),
) do n_clicks, geno_contents, pheno_contents, lambda
    if n_clicks > 0 && !isnothing(geno_contents) && !isnothing(pheno_contents)
        try
            # 1. Parse uploaded data
            geno_df = CSV.read(IOBuffer(base64decode(split(geno_contents, ',')[2])), DataFrame)
            pheno_df = CSV.read(IOBuffer(base64decode(split(pheno_contents, ',')[2])), DataFrame)

            # Ensure column names are symbols for compatibility
            rename!(geno_df, Symbol.(names(geno_df)))
            rename!(pheno_df, Symbol.(names(pheno_df)))

            data = GenomicData(geno_df, pheno_df)

            # 2. Train model
            model = GBLUPModel(lambda=Float64(lambda))
            fit!(model, data)

            # 3. Predict
            predictions = predict(model, data.genotypes)

            # 4. Create results table and plot
            results_df = DataFrame(
                ID=data.genotypes[!,1],
                True=data.phenotypes[!,2],
                Predicted=predictions
            )

            corr_acc = cor(results_df.True, results_df.Predicted)

            scatter_plot = dcc_graph(
                figure=(
                    data=[
                        (x=results_df.True, y=results_df.Predicted, type="scatter", mode="markers")
                    ],
                    layout=(
                        title="预测值 vs. 真实值 (准确率: $(round(corr_acc, digits=4)))",
                        xaxis=Dict("title" => "真实表型"),
                        yaxis=Dict("title" => "预测表型"),
                    )
                )
            )

            results_table = dash_datatable(
                data=Dict.(pairs.(eachrow(results_df))),
                columns=[Dict("name" => i, "id" => i) for i in names(results_df)],
                page_size=10,
                style_table=Dict("overflowX" => "auto")
            )

            return html_div([
                html_h3("分析完成"),
                scatter_plot,
                html_h4("结果表格"),
                results_table
            ])

        catch e
            return html_div(html_pre("发生错误: $(sprint(showerror, e))"))
        end
    end
    return ""
end


# --- Run the App ---
# To run this app, execute this file from the Julia REPL:
# include("ui/dash/app.jl")
run_server(app, "0.0.0.0", 8050, debug=true)
