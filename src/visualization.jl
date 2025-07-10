# ===== src/visualization.jl =====
"""
Advanced visualization tools for epistatic genomic analysis within DynamicEpistasisGBLUP.
Includes network plots for interactions, heatmaps for GRMs, variance decomposition charts,
interactive Manhattan plots, and more.
Uses various plotting backends like Plots.jl, StatsPlots.jl, PlotlyJS.jl, Makie.jl.
"""

module Visualization

# Standard plotting packages
using Plots # Generic plotting interface
using StatsPlots # For statistical plots, e.g., Manhattan, QQ
# PlotlyJS might be good for interactive web-based plots
# using PlotlyJS
# Makie (GLMakie for interactive desktop, CairoMakie for static) for advanced/3D plots
# using GLMakie # Or CairoMakie

# Other utilities
using Colors # For custom color schemes
# using NetworkLayout # For graph layout algorithms if not in Graphs.jl directly
using Graphs      # For graph structures (e.g., SimpleGraph for interaction networks)
using DataFrames  # For handling results data for plotting
using Statistics  # For mean, std in plot summaries
using StatsBase   # For quantile, hclust (if needed and not in base Stats)

# Assuming types.jl and other core modules are accessible if their data is directly plotted.
# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

export plot_epistasis_interaction_network, # Renamed from plot_epistasis_network
       plot_grm_heatmap_viz, # Renamed
       plot_variance_components_pie_bar, # Renamed
       # interactive_manhattan_plot_epistasis, # Renamed (PlotlyJS specific, might be optional)
       # plot_accuracy_surface_3D, # Renamed (Makie specific, might be optional)
       animate_genetic_progress_viz # Renamed

"""
    plot_epistasis_interaction_network(...)

Plots a network graph of epistatic interactions.
Stronger interactions can be represented by thicker/darker edges.
Node size/color can represent SNP properties (e.g., main effect size, degree).
Requires a plotting backend that supports graph visualization (e.g., Plots.jl with GraphRecipes).
"""
function plot_epistasis_interaction_network(
    interaction_tuples::Vector{Tuple{Int32, Int32}}, # List of (SNP1_idx, SNP2_idx)
    interaction_scores::Vector{T};                 # Scores for each interaction
    top_n_interactions_to_plot::Int = 100,
    layout_alg::Symbol = :spring, # :spring, :circular, :spectral
    # node_attribute_values::Union{Nothing, Vector} = nothing, # e.g., main effects of SNPs
    # node_size_by_degree::Bool = true,
    edge_width_scale::Float64 = 5.0,
    figure_title::String = "Epistatic Interaction Network",
    save_plot_path::Union{Nothing, String} = nothing
) where T <: AbstractFloat

    if isempty(interaction_tuples)
        # println("No interactions to plot.")
        return nothing
    end

    # Select top N interactions by absolute score
    num_to_plot = min(top_n_interactions_to_plot, length(interaction_scores))
    if num_to_plot == 0 return nothing end

    sorted_indices = partialsortperm(abs.(interaction_scores), 1:num_to_plot, rev=true)
    plot_interactions = interaction_tuples[sorted_indices]
    plot_scores_abs = abs.(interaction_scores[sorted_indices]) # Use absolute for width/color intensity

    # Create a graph
    unique_snps_in_plot = sort(unique(vcat([pair[1] for pair in plot_interactions],
                                           [pair[2] for pair in plot_interactions])))
    if isempty(unique_snps_in_plot) return nothing end

    snp_to_node_map = Dict(snp_idx => i for (i, snp_idx) in enumerate(unique_snps_in_plot))

    graph_obj = SimpleGraph(length(unique_snps_in_plot))
    edge_weights_for_layout = Float64[] # For layout algorithm if it uses weights

    for (idx, pair) in enumerate(plot_interactions)
        u_node = get(snp_to_node_map, pair[1], -1)
        v_node = get(snp_to_node_map, pair[2], -1)
        if u_node != -1 && v_node != -1 && u_node != v_node
            add_edge!(graph_obj, u_node, v_node)
            # Using score for layout weight, can be scaled
            push!(edge_weights_for_layout, plot_scores_abs[idx])
        end
    end
    if ne(graph_obj) == 0 return nothing end # No valid edges added

    # Prepare for Plots.jl graph recipe (using default backend, e.g., GR)
    # Ensure Plots.jl is the active backend or use specific backend commands.
    # Requires `GraphRecipes` to be available in the environment if using its specific recipes.
    # Default Plots.jl can plot graphs from Graphs.jl with some limitations.

    # Layout using NetworkLayout.jl if available, or simple spring from Plots/GraphRecipes
    # For now, simple spring layout from Plots.jl if it supports it.
    # NetworkLayout.jl offers more options: spring, stress, etc.
    # `layout = NetworkLayout.Spring(dim=2)`
    # `node_positions = NetworkLayout.layout(layout, graph_obj)`
    # For Plots.jl direct graph plotting:
    # `plot(graph_obj, layout=layout_alg_plots, ...)`
    # The `layout_alg` symbol needs to map to Plots.jl options.
    # For now, this is a conceptual call.

    # Using Plots.jl with its default graph layouting (often spring-like)
    # Node sizes by degree
    node_degrees = degree(graph_obj)
    node_sizes_plot = 5 .+ (node_degrees ./ max(1,maximum(node_degrees))) .* 15 # Scale sizes

    # Edge widths by interaction score
    # This requires getting edge list in same order as plot_scores_abs or mapping.
    # For now, not directly setting varying edge widths in simple Plots.jl call.
    # Varying edge color/alpha by score
    max_score = maximum(plot_scores_abs)
    if max_score < eps(T) max_score = one(T) end # Avoid division by zero
    edge_alphas = plot_scores_abs ./ max_score

    # This is a simplified graph plot. GraphRecipes provides more control.
    # plot_obj = plot(graph_obj, layout=layout_alg, markersize=node_sizes_plot,
    #                 node_z=node_degrees, markerstrokewidth=1,
    #                 # linealpha=edge_alphas, # Needs mapping edge alphas to edges
    #                 title=figure_title, size=(800,700), framestyle=:none)

    # Placeholder: Actual plotting requires a backend to be active.
    # This function describes what to plot.
    # The original code used Makie for this. If Makie is the target:
    # fig = Figure(...)
    # ax = Axis(fig[1,1], ...)
    # graphplot!(ax, graph_obj, layout=..., node_size=..., edge_width=...)
    # This is a stub.
    # println("Visualization: Epistasis network plot generation requested (stubbed).")
    # println("  Would plot $(ne(graph_obj)) interactions among $(nv(graph_obj)) SNPs.")

    if save_plot_path !== nothing
        # try savefig(plot_obj, save_plot_path) catch e; println("Error saving plot: $e") end
    end

    return nothing # Return plot object if created
end


"""
    plot_grm_heatmap_viz(...)

Plots a heatmap of a Genomic Relationship Matrix (GRM).
Can optionally cluster individuals to reveal population structure.
Uses Plots.jl heatmap or PlotlyJS for interactive version.
"""
function plot_grm_heatmap_viz(
    G_matrix::Matrix{T}; # GRM on CPU
    plot_title::String = "Genomic Relationship Matrix",
    apply_clustering::Bool = true,
    # use_interactive::Bool = false, # To switch to PlotlyJS
    save_plot_path::Union{Nothing, String} = nothing
) where T <: AbstractFloat

    n_indiv = size(G_matrix,1)
    if n_indiv == 0 return nothing end

    G_to_plot = G_matrix
    if apply_clustering && n_indiv > 1
        try
            # Hierarchical clustering (requires StatsBase.hclust or similar)
            # Distance matrix D = 1 - G (or max(G) - G for similarity to distance)
            # For relationships, G_ii approx 1, G_ij smaller.
            # A simple distance: sqrt(2*(1-G_ij))
            dist_matrix = sqrt.(max.(zero(T), T(2) .* (one(T) .- G_matrix)))
            diagm(dist_matrix) .= zero(T) # Zero distance to self

            if suami(dist_matrix) # Check if any non-diagonal element is finite for hclust
              hc = hclust(dist_matrix, linkage=:average, branchorder=:optimal)
              row_order = hc.order
              G_to_plot = G_matrix[row_order, row_order]
            end
        catch e
            # println("Clustering for GRM heatmap failed: $e. Plotting unordered matrix.")
            # Fallback to unordered if clustering fails (e.g. package not available)
        end
    end

    # Using Plots.jl heatmap
    # plot_obj = heatmap(G_to_plot, aspect_ratio=:equal, color=:vik, # Diverging colormap
    #                    title=plot_title, xlabel="Individuals", ylabel="Individuals",
    #                    size=(700,600), clims=(-maximum(abs.(G_to_plot)), maximum(abs.(G_to_plot)))) # Symmetrize color limits

    # Placeholder:
    # println("Visualization: GRM heatmap generation requested (stubbed).")
    # println("  Matrix size: $n_indiv x $n_indiv. Clustering: $apply_clustering.")

    if save_plot_path !== nothing
        # try savefig(plot_obj, save_plot_path) catch e; println("Error saving plot: $e") end
    end

    return nothing # Return plot object
end

"""
    plot_variance_components_pie_bar(...)

Visualizes variance components (additive, epistatic, residual) using
a pie chart for proportions and a bar chart for heritability estimates.
"""
function plot_variance_components_pie_bar(
    var_comps::Main.DynamicEpistasisGBLUP.VarianceComponents{T}; # VarianceComponents struct
    plot_title_suffix::String = "",
    save_plot_path::Union{Nothing, String} = nothing
) where T <: AbstractFloat

    labels = ["Additive (σ²a)", "Epistatic (σ²aa)", "Residual (σ²e)"]
    values_abs = [var_comps.σ²_a, var_comps.σ²_aa, var_comps.σ²_e]

    # Filter out zero components for pie chart if they are truly zero (not just small)
    non_zero_indices = findall(v -> v > eps(T) * var_comps.σ²_p, values_abs)
    if isempty(non_zero_indices) && var_comps.σ²_p <= eps(T) # All zero or total is zero
        # println("All variance components are zero or near zero. Skipping plot.")
        return nothing
    elseif isempty(non_zero_indices) # All effectively zero, but total was non-zero (should not happen if σ²_p is sum)
        # This case means σ²_p was positive but components were filtered. Use original values.
        non_zero_indices = 1:length(values_abs)
    end

    pie_labels = labels[non_zero_indices]
    pie_values = values_abs[non_zero_indices]

    # Pie chart for proportions
    # p1 = pie(pie_labels, pie_values, title="Variance Proportions" * plot_title_suffix,
    #          autopct="%1.1f%%", legend=:outerright)

    # Bar chart for heritabilities
    h_labels = ["h² (Narrow)", "H² (Broad)", "h²_epi (H²-h²)"]
    h_values = [var_comps.h², var_comps.H², max(zero(T), var_comps.H² - var_comps.h²)]
    # p2 = bar(h_labels, h_values, title="Heritability Estimates" * plot_title_suffix,
    #          ylabel="Heritability", legend=false, ylims=(0,1.05), bar_width=0.6)
    # annotate!(twinx(p2), h_labels, h_values .+ 0.02, text.(round.(h_values, digits=3), :center, 8))


    # Combine plots if using Plots.jl layout
    # final_plot = plot(p1, p2, layout=(1,2), size=(1000, 400))
    # println("Visualization: Variance decomposition plot generation requested (stubbed).")

    if save_plot_path !== nothing
        # try savefig(final_plot, save_plot_path) catch e; println("Error saving plot: $e") end
    end

    return nothing # Return combined plot object
end


"""
    animate_genetic_progress_viz(...)

Creates an animation showing genetic progress over generations.
Can include distributions of phenotypes/GEBVs, trend lines for mean genetic value,
and changes in genetic variance or heritability.
"""
function animate_genetic_progress_viz(
    populations_history_list::Vector{Main.DynamicEpistasisGBLUP.PopulationData{T}},
    models_history_list::Vector{Main.DynamicEpistasisGBLUP.OrthogonalGBLUP{T}};
    trait_name_for_plot::String = "Simulated Trait",
    animation_save_path::String = "genetic_progress_animation.gif"
) where T <: AbstractFloat

    num_generations = length(populations_history_list)
    if num_generations == 0 return nothing end

    # Data for animation frames
    mean_phenotypes_over_gens = [mean(pop.phenotypes.values) for pop in populations_history_list]
    # Ensure models_history_list aligns with populations_history_list for heritabilities
    # (e.g., models_history_list[i] is model for populations_history_list[i])
    # If model is fitted for gen_i based on pop_i, then lengths match.
    # If model for gen_i is based on pop_{i-1}, adjust indexing.
    # Assuming lengths match for simplicity here.
    h2_narrow_over_gens = [mod.variance.h² for mod in models_history_list[1:min(end,num_generations)]] # Handle if models list shorter
    H2_broad_over_gens = [mod.variance.H² for mod in models_history_list[1:min(end,num_generations)]]

    # Create animation using Plots.jl @animate macro
    # anim = @animate for gen_idx in 1:num_generations
    #     current_pop_phenos = populations_history_list[gen_idx].phenotypes.values

    #     # Layout for 2x2 plots per frame
    #     l = @layout [a b; c d]

    #     # Plot 1: Phenotype distribution for current generation
    #     p1 = histogram(current_pop_phenos, bins=30, normalize=:probability,
    #                    title="Gen $(gen_idx-1): Phenotype Distribution", xlabel=trait_name_for_plot,
    #                    legend=false, xlims=(minimum(mean_phenotypes_over_gens)-2*std(current_pop_phenos), maximum(mean_phenotypes_over_gens)+2*std(current_pop_phenos) )) # Dynamic xlims

    #     # Plot 2: Mean genetic trend up to current generation
    #     p2 = plot(0:(gen_idx-1), mean_phenotypes_over_gens[1:gen_idx], marker=:circle,
    #               xlabel="Generation", ylabel="Mean Phenotype", title="Genetic Trend", legend=false)

    #     # Plot 3: Heritability trends
    #     p3 = plot(0:(gen_idx-1), h2_narrow_over_gens[1:gen_idx], label="h² (Narrow)", color=:blue, marker=:o, ylims=(0,1.05))
    #     plot!(p3, 0:(gen_idx-1), H2_broad_over_gens[1:gen_idx], label="H² (Broad)", color=:red, marker=:s,
    #           xlabel="Generation", ylabel="Heritability", title="Heritability Trends")

    #     # Plot 4: Placeholder (e.g., genetic variance trend, or specific interaction effects)
    #     p4 = plot(title="Additional Info (Placeholder)", framestyle=:none) # Empty plot

    #     plot(p1,p2,p3,p4, layout=l, size=(1000,800))
    # end

    # println("Visualization: Genetic progress animation generation requested (stubbed).")
    # println("  Would animate over $num_generations generations.")

    # if save_plot_path !== nothing && num_generations > 0
    #     # try gif(anim, save_plot_path, fps=2) catch e; println("Error saving animation: $e") end
    # end

    return nothing # Return animation object
end


# Interactive plots (PlotlyJS, Makie) are more involved and might be optional features.
# The stubs for `interactive_manhattan_plot_epistasis` and `plot_accuracy_surface_3D`
# from original `visualization.jl` are omitted for now to focus on Plots.jl examples.
# They can be added if those libraries are firm dependencies.

# Export functions if this file were a module
# export plot_epistasis_interaction_network, plot_grm_heatmap_viz,
#        plot_variance_components_pie_bar, animate_genetic_progress_viz

end # module Visualization
