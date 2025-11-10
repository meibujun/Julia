# src/GenomicProMultiOmics/training.jl

using Optimisers, Zygote, Random

function train_multiomics_model(model, snp_data_train, rnaseq_data_train, y_train,
                              snp_data_val, rnaseq_data_val, y_val,
                              n_epochs, batch_size, early_stopping_patience)

    ps, st = Lux.setup(Random.default_rng(), model)
    opt_state = Optimisers.setup(Adam(0.001), ps)

    best_val_loss = Inf
    patience_counter = 0

    for epoch in 1:n_epochs
        for batch_indices in Iterators.partition(1:size(snp_data_train,1), batch_size)
            snp_batch = snp_data_train[batch_indices, :]
            rnaseq_batch = rnaseq_data_train[batch_indices, :]
            y_batch = y_train[batch_indices]

            loss, grads = Zygote.withgradient(ps) do p
                y_pred, _ = forward_pass(model, snp_batch, rnaseq_batch, p, st)
                mse(y_pred, y_batch)
            end

            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        end

        # Validation
        y_pred_val, _ = forward_pass(model, snp_data_val, rnaseq_data_val, ps, st)
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

function forward_pass(model, snp_data, rnaseq_data, ps, st)
    snp_encoded, st_snp = model.snp_encoder(snp_data, ps.snp_encoder, st.snp_encoder)
    rnaseq_encoded, st_rnaseq = model.expression_encoder(rnaseq_data, ps.expression_encoder, st.expression_encoder)
    fused, st_fusion = model.fusion_layer(snp_encoded, rnaseq_encoded, ps.fusion_layer, st.fusion_layer)
    y_pred, st_pred = model.predictor(fused, ps.predictor, st.predictor)

    st_new = (snp_encoder=st_snp, expression_encoder=st_rnaseq, fusion_layer=st_fusion, predictor=st_pred)
    return y_pred, st_new
end
