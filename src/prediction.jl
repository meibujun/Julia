# ===== src/prediction.jl =====
"""
Genomic prediction functions and cross-validation utilities.
Handles prediction for new individuals given a fitted GBLUP model.
"""

using CUDA
using Random # For shuffle in cross_validation
using Statistics # For cor, mean in cross_validation
using LinearAlgebra # For I in inverse_gpu, norm
using DataFrames # For cross_validation results
# Assuming types.jl (OrthogonalGBLUP, GenotypeMatrix, PopulationData, PhenotypeData)
# and grm_computation.jl (for compute_grm_cross!, compute_epistatic_grm_cross!)
# and epistasis_core.jl (for orthogonal_epistasis_gblup)
# and utils.jl (for regression_coefficient) are accessible.

# Helper to get Float type
_Float() = DynamicEpistasisGBLUP.Float

"""
    genomic_prediction(
        model::OrthogonalGBLUP{T},
        new_genotypes::GenotypeMatrix{T},
        training_genotypes::GenotypeMatrix{T}, # Genotypes of the training population
        u_hat_additive_train::CuVector{T},     # Additive BLUPs for training population
        u_hat_epistatic_train::Union{Nothing, CuVector{T}} = nothing; # Epistatic BLUPs for training
        include_epistasis::Bool = true
    ) where T <: AbstractFloat -> Vector{T}

Predicts Genomic Estimated Breeding Values (GEBVs) for a new set of individuals (`new_genotypes`)
using a pre-fitted `model` (`OrthogonalGBLUP` object), the `training_genotypes` data that
was used to fit the model, and the BLUPs (`u_hats`) obtained for the training population.

The prediction formula is:
  `GEBV_new = X_new*β + G_new_train * G_train⁻¹ * u_hat_train_add + G_aa_new_train * G_aa_train⁻¹ * u_hat_train_epi`
where:
- `X_new*β` represents the fixed effect contribution (e.g., overall mean).
- `G_new_train` is the cross-GRM between new and training individuals.
- `G_train⁻¹` is the inverse of the training population's GRM (from `model`).
- `u_hat_train` are the BLUPs of the training individuals.

# Arguments
- `model::OrthogonalGBLUP{T}`: The fitted GBLUP model object containing GRMs of the training population (`model.G`, `model.G_aa`) and fixed effect estimates (`model.fixed_effects`).
- `new_genotypes::GenotypeMatrix{T}`: Genotype data for the individuals to be predicted.
- `training_genotypes::GenotypeMatrix{T}`: Genotype data of the original training population. This is crucial for computing cross-GRMs correctly using training population allele frequencies.
- `u_hat_additive_train::CuVector{T}`: Vector of additive BLUPs (random genetic effects, `u_a`) for the training population individuals (must be on GPU).
- `u_hat_epistatic_train::Union{Nothing, CuVector{T}} = nothing`: Optional vector of epistatic BLUPs (`u_aa`) for the training population (GPU). Required if `include_epistasis` is true and `model.G_aa` is present.
- `include_epistasis::Bool = true`: If `true` and epistatic components are available in the model and `u_hat_epistatic_train`, epistatic predictions are included.

# Returns
- `Vector{T}`: A vector of predicted GEBVs for the `new_genotypes` individuals (on CPU).

# Important Notes
- This function assumes `training_genotypes.allele_freq` contains the correct allele frequencies that were used to derive `model.G` and `model.G_aa`. These frequencies will be used to center/standardize `new_genotypes` for cross-GRM computation.
- The `u_hat_additive_train` and `u_hat_epistatic_train` are the solutions for random effects from the MME solved during model training, not the final GEBVs (which also include fixed effects).
"""
function genomic_prediction(
    model::OrthogonalGBLUP{T},
    new_genotypes::GenotypeMatrix{T},
    training_genotypes::GenotypeMatrix{T}, # Genotypes of the training population
    u_hat_additive_train::CuVector{T},     # Additive BLUPs for training population
    u_hat_epistatic_train::Union{Nothing, CuVector{T}} = nothing; # Epistatic BLUPs for training
    include_epistasis::Bool = true
) where T <: AbstractFloat

    n_new = new_genotypes.n_individuals
    n_train = training_genotypes.n_individuals

    if size(model.G, 1) != n_train || length(u_hat_additive_train) != n_train
        error("Training population size mismatch between model GRM ($(size(model.G,1))), training genotypes ($n_train), and additive BLUPs ($(length(u_hat_additive_train))).")
    end
    if include_epistasis && model.G_aa !== nothing
        if u_hat_epistatic_train === nothing || size(model.G_aa,1) != n_train || length(u_hat_epistatic_train) != n_train
             error("Epistatic components mismatch for training population when include_epistasis is true.")
        end
    end

    # --- Additive Component ---
    # 1. Compute G_new_train_additive (N_new x N_train)
    G_new_train_add_gpu = CuArray{T}(undef, n_new, n_train)
    # compute_grm_cross! uses allele freqs from training_genotypes (second arg) to center both.
    Main.DynamicEpistasisGBLUP.compute_grm_cross!(G_new_train_add_gpu, new_genotypes, training_genotypes, use_gpu=true)

    # 2. Compute G_train_additive_inv (N_train x N_train)
    ridge_G_add = T(1e-6) * (abs(tr(model.G))/n_train + T(1e-9)) # Relative ridge
    G_train_add_stable = model.G + CuMatrix{T}(I, n_train, n_train) * ridge_G_add
    G_train_add_inv = CUDA.zeros(T,0,0) # Ensure defined for scope
    try
        ch_Gadd = cholesky(Symmetric(G_train_add_stable); check=true)
        G_train_add_inv = inv(ch_Gadd)
    catch e
        if isa(e, PosDefException); G_train_add_inv = inv(G_train_add_stable) # Fallback
        else rethrow(e) end
    end
    if size(G_train_add_inv,1) == 0; error("Failed to invert training additive GRM during prediction."); end

    # 3. Predicted additive values for new individuals: G_new_train * G_train_inv * u_hat_train_add
    predictions_add_gpu = G_new_train_add_gpu * (G_train_add_inv * u_hat_additive_train)

    total_predictions_gpu = predictions_add_gpu

    # --- Epistatic Component (if included) ---
    if include_epistasis && model.G_aa !== nothing && u_hat_epistatic_train !== nothing
        G_aa_new_train_gpu = CuArray{T}(undef, n_new, n_train)
        Main.DynamicEpistasisGBLUP.compute_epistatic_grm_cross!(G_aa_new_train_gpu, new_genotypes, training_genotypes, use_gpu=true)

        ridge_G_aa = T(1e-6) * (abs(tr(model.G_aa))/n_train + T(1e-9))
        G_aa_train_stable = model.G_aa + CuMatrix{T}(I, n_train, n_train) * ridge_G_aa
        G_aa_train_inv = CUDA.zeros(T,0,0) # Ensure defined
        try
            ch_Gaa = cholesky(Symmetric(G_aa_train_stable); check=true)
            G_aa_train_inv = inv(ch_Gaa)
        catch e
            if isa(e, PosDefException); G_aa_train_inv = inv(G_aa_train_stable)
            else rethrow(e) end
        end
        if size(G_aa_train_inv,1) == 0; error("Failed to invert training epistatic GRM during prediction."); end

        predictions_epi_gpu = G_aa_new_train_gpu * (G_aa_train_inv * u_hat_epistatic_train)
        total_predictions_gpu = total_predictions_gpu .+ predictions_epi_gpu
    end

    final_predictions_cpu = Array(total_predictions_gpu)

    if model.fixed_effects !== nothing
        # Assuming fixed_effects in model is [beta_intercept; beta_other_effects...]
        # And new_genotypes needs a corresponding design matrix X_new for these fixed effects.
        # For simplicity, if only an intercept was fitted (model.fixed_effects is 1x1 matrix or single element vector):
        if length(model.fixed_effects) == 1
            final_predictions_cpu .+= model.fixed_effects[1] # Add global intercept
        else
            # TODO: Handle multiple fixed effects. Requires X_new for new_genotypes.
            @warn "Prediction with multiple fixed effects requires a design matrix for new individuals (not implemented here). Only intercept applied if available."
            if size(model.fixed_effects,2) == 1 && size(model.fixed_effects,1) == 1 # Check if it's essentially an intercept
                 final_predictions_cpu .+= model.fixed_effects[1,1]
            end
        end
    end

    return final_predictions_cpu
end


"""
    cross_validation(population::PopulationData{T}; n_folds=5, include_epistasis=true, seed=123) -> DataFrame

Performs k-fold cross-validation for the GBLUP model on a given population.
Returns a DataFrame with accuracy, bias, and MSE for each fold.
"""
function cross_validation(
    population::PopulationData{T};
    n_folds::Int = 5,
    include_epistasis::Bool = true,
    # update_frequencies for each fold's training? Usually CV uses fixed state from full data for freqs.
    # For dynamic model testing, CV might be within each generation of a larger simulation.
    # Here, assume freqs are from the `population` data as a whole for splitting.
    seed::Int = 123
) where T <: AbstractFloat
    Random.seed!(seed) # For reproducible folds

    n_individuals = population.genotypes.n_individuals
    indices = shuffle(1:n_individuals)

    fold_size = n_individuals ÷ n_folds
    remainder = n_individuals % n_folds

    results_list = [] # Store NamedTuples or similar for each fold

    for k_fold in 1:n_folds
        val_start_idx = (k_fold - 1) * fold_size + 1 + min(k_fold - 1, remainder)
        current_fold_size = fold_size + (k_fold <= remainder ? 1 : 0)
        val_end_idx = val_start_idx + current_fold_size - 1

        val_indices = indices[val_start_idx:val_end_idx]
        train_indices = setdiff(indices, val_indices)

        if isempty(train_indices) || isempty(val_indices)
            # println("Warning: Skipping fold $k_fold due to empty train/validation set.")
            continue
        end

        # Split population data into training and validation sets
        # This `split_population` function was defined in the original `utils.jl` / demo.
        # It needs to be accessible here.
        train_pop, val_pop = split_population_data_cv(population, train_indices, val_indices)

        # Fit model on training data
        # `orthogonal_epistasis_gblup` returns (model_object, gebv_vector_for_training_data)
        # We need the `model_object` for prediction, and potentially `gebv_vector_training` if prediction logic requires it.
        fitted_model_train, gebv_train = orthogonal_epistasis_gblup(
            train_pop; # Pass the training subset of the population
            include_epistasis = include_epistasis,
            update_frequencies_per_generation = true # Freqs from train_pop
        )

        # Predict on validation set
        # The `genomic_prediction` function needs the fitted model and new genotypes.
        # It also critically depends on how it accesses training population info for cross-GRMs.
        # As discussed, this is a stub until cross-GRMs are done.
        predictions_val = genomic_prediction(
            fitted_model_train, # The model fitted on the training data
            val_pop.genotypes;  # Genotypes of the validation individuals
            # reference_genotypes = train_pop.genotypes, # This kind of argument might be needed
            # u_hat_additive_train = # BLUPs from training, if needed by prediction function
            include_epistasis = include_epistasis
        )

        # True values for validation set
        # If simulation, true BVs might be stored in metadata. Otherwise, use phenotypes.
        # For GEBV accuracy, usually correlate with true BVs if known, or phenotypes.
        # Let's use phenotypes for now, common in real data CV.
        true_values_val = val_pop.phenotypes.values

        # Evaluate predictions
        # Ensure predictions_val and true_values_val are not empty and have variation for cor.
        accuracy = zero(T)
        bias_reg_coeff = zero(T) # Regression of true on predicted
        mse_val = zero(T)

        if length(predictions_val) > 1 && length(true_values_val) > 1 && var(predictions_val) > eps(T) && var(true_values_val) > eps(T)
            accuracy = cor(predictions_val, true_values_val)
            # Bias using regression_coefficient from utils.jl
            bias_reg_coeff = regression_coefficient(true_values_val, predictions_val)
        end
        if length(predictions_val) == length(true_values_val) && length(true_values_val) > 0
             mse_val = mean((predictions_val .- true_values_val).^2)
        end

        push!(results_list, (fold=k_fold, accuracy=accuracy, bias=bias_reg_coeff, mse=mse_val))
    end

    return DataFrame(results_list) # Convert list of NamedTuples/tuples to DataFrame
end


"""
    split_population_data_cv(population, train_indices, val_indices)

Helper function to split `PopulationData` into training and validation sets for CV.
This was named `split_population` in the original demo/utils.
Renamed to avoid conflict if there's a general `split_population` for other purposes.
"""
function split_population_data_cv(
    full_population::PopulationData{T},
    train_indices::Vector{Int},
    val_indices::Vector{Int}
) where T <: AbstractFloat

    # Split genotype data
    # Assuming genotypes.data is CuArray (individuals x SNPs)
    geno_data_host = Array(full_population.genotypes.data) # Move to CPU for slicing by indices

    train_geno_data = geno_data_host[train_indices, :]
    val_geno_data = geno_data_host[val_indices, :]

    # Create new GenotypeMatrix objects for train and validation sets
    # Allele frequencies for these subsets could be recalculated or inherited from full_population.
    # For CV, typically use allele freqs from the full training set for consistency.
    # Here, `full_population.genotypes.allele_freq` are used.

    # Train GenotypeMatrix
    # Missing mask needs to be subsetted too. This is complex for SparseCuMatrixCSR.
    # For now, creating new empty missing masks.
    # TODO: Properly subset sparse missing_mask.
    train_missing_mask = CuSparseMatrixCSR(spzeros(Bool, Int32, size(train_geno_data,1), size(train_geno_data,2)))
    train_genotypes = GenotypeMatrix(
        CuArray(train_geno_data),
        train_missing_mask, # Placeholder for subsetted mask
        full_population.genotypes.allele_freq, # Use original allele freqs
        Int32(length(train_indices)),
        full_population.genotypes.n_snps,
        full_population.genotypes.ploidy
    )

    # Validation GenotypeMatrix
    val_missing_mask = CuSparseMatrixCSR(spzeros(Bool, Int32, size(val_geno_data,1), size(val_geno_data,2)))
    val_genotypes = GenotypeMatrix(
        CuArray(val_geno_data),
        val_missing_mask, # Placeholder for subsetted mask
        full_population.genotypes.allele_freq, # Use original allele freqs
        Int32(length(val_indices)),
        full_population.genotypes.n_snps,
        full_population.genotypes.ploidy
    )

    # Split phenotype data
    train_pheno_values = full_population.phenotypes.values[train_indices]
    val_pheno_values = full_population.phenotypes.values[val_indices]

    # Assuming fixed/random effects are not used or are handled globally if present.
    # Subsetting DataFrames for fixed/random effects also needed if they exist.
    train_phenotypes = PhenotypeData(
        train_pheno_values,
        full_population.phenotypes.trait_names,
        # Subset fixed_effects DataFrame: full_population.phenotypes.fixed_effects[train_indices, :]
        full_population.phenotypes.fixed_effects === nothing ? nothing : full_population.phenotypes.fixed_effects[train_indices,:],
        full_population.phenotypes.random_effects === nothing ? nothing : full_population.phenotypes.random_effects[train_indices,:]
    )
    val_phenotypes = PhenotypeData(
        val_pheno_values,
        full_population.phenotypes.trait_names,
        full_population.phenotypes.fixed_effects === nothing ? nothing : full_population.phenotypes.fixed_effects[val_indices,:],
        full_population.phenotypes.random_effects === nothing ? nothing : full_population.phenotypes.random_effects[val_indices,:]
    )

    # Create new PopulationData objects
    # Pedigree is not typically subsetted this way for CV, usually handled by relationship matrix.
    train_pop = PopulationData(
        train_genotypes,
        train_phenotypes,
        nothing, # Pedigree usually not passed for GBLUP with GRM
        full_population.generation, # Same generation
        full_population.metadata    # Inherit metadata (e.g., architecture)
    )
    val_pop = PopulationData(
        val_genotypes,
        val_phenotypes,
        nothing,
        full_population.generation,
        full_population.metadata
    )

    return train_pop, val_pop
end

# Functions `compute_grm_cross!` and `compute_epistatic_grm_cross!` are MISSING.
# They were part of the original `prediction.jl` but without implementation.
# They are crucial for `genomic_prediction` to work.
# Their implementation is part of a later plan step.

# Export functions if this file were a module
# export genomic_prediction, cross_validation
