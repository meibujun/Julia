# src/GenomicProMultiOmics/training.jl

using Optimisers, Zygote, Random

function train_multiomics_model(model, multi_omics_data_train::MultiOmicsData, y_train,
                              multi_omics_data_val::MultiOmicsData, y_val,
                              n_epochs, batch_size, early_stopping_patience)

    ps, st = Lux.setup(Random.default_rng(), model)
    opt_state = Optimisers.setup(Adam(0.001), ps)

    best_val_loss = Inf
    patience_counter = 0

    for epoch in 1:n_epochs
        # This is a simplified batching strategy. A real implementation would need to
        # handle the different data structures in the MultiOmicsData object.
        for batch_indices in Iterators.partition(1:length(y_train), batch_size)

            loss, grads = Zygote.withgradient(ps) do p
                y_pred, _ = forward_pass(model, multi_omics_data_train, p, st)
                mse(y_pred, y_train[batch_indices])
            end

            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        end

        # Validation
        y_pred_val, _ = forward_pass(model, multi_omics_data_val, ps, st)
        val_loss = mse(y_pred_val, y_val)

        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
        else
            patience_counter += 1
        end

        if patience_counter >= early_stopping_patience
            break
        end
    end

    return ps, st
end

function forward_pass(model, multi_omics_data::MultiOmicsData, ps, st)
    # This is a simplified forward pass for two modalities. A real implementation
    # would need to be more general.
    snp_data = multi_omics_data.data[:genotypes]
    rnaseq_data = multi_omics_data.data[:expression]

    snp_encoded, st_snp = model.snp_encoder(snp_data, ps.snp_encoder, st.snp_encoder)
    rnaseq_encoded, st_rnaseq = model.expression_encoder(rnaseq_data, ps.expression_encoder, st.expression_encoder)
    fused, st_fusion = model.fusion_layer(snp_encoded, rnaseq_encoded, ps.fusion_layer, st.fusion_layer)
    y_pred, st_pred = model.predictor(fused, ps.predictor, st.predictor)

    st_new = (snp_encoder=st_snp, expression_encoder=st_rnaseq, fusion_layer=st_fusion, predictor=st_pred)
    return y_pred, st_new
end
