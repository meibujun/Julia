import numpy as np
import pandas as pd
import sys
import os

# Adjust path
try:
    from pyjwas.core import MixedModelEquations, VarianceCovariance, MCMCInfo
    from pyjwas.core.model_builder import build_model, get_mme_components
    from pyjwas.pedigree import read_pedigree, calculate_A_inverse
    from pyjwas.mcmc import run_mcmc
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from pyjwas.core import MixedModelEquations, VarianceCovariance, MCMCInfo
    from pyjwas.core.model_builder import build_model, get_mme_components
    from pyjwas.pedigree import read_pedigree, calculate_A_inverse
    from pyjwas.mcmc import run_mcmc

def simulate_animal_model_data(ped_filepath="dummy_animal_ped.csv",
                               n_extra_founders=2, n_offspring_gen=2, offspring_per_pairing=2,
                               intercept_true=50.0, sigma_a_true=10.0, sigma_e_true=20.0):
    """Simulates a pedigree and phenotypes for a simple animal model."""
    print(f"Simulating data for Animal Model...")

    # Create a slightly more complex pedigree than just 3 animals
    current_id = 0
    ped_rows = []

    def next_id():
        nonlocal current_id
        current_id += 1
        return f"Anim{current_id}"

    founders = [next_id() for _ in range(n_extra_founders + 2)] # At least 2 founders
    for f_id in founders:
        ped_rows.append([f_id, "0", "0"])

    last_gen_parents = list(founders)
    all_animals_in_ped = list(founders)

    for gen in range(n_offspring_gen):
        current_gen_offspring = []
        if len(last_gen_parents) < 2: break
        for i in range(0, len(last_gen_parents) -1, 2): # Pair them up
            sire = last_gen_parents[i]
            dam = last_gen_parents[i+1]
            for _ in range(offspring_per_pairing):
                offspring_id = next_id()
                ped_rows.append([offspring_id, sire, dam])
                current_gen_offspring.append(offspring_id)
                all_animals_in_ped.append(offspring_id)
        if not current_gen_offspring: break
        last_gen_parents = current_gen_offspring
        if len(last_gen_parents) < 2 and gen < n_offspring_gen -1 : # Need pairs for next gen
            # Add some more "random" animals to allow pairings if list is too small
            new_mates = [next_id() for _ in range(len(last_gen_parents))]
            for mate_id in new_mates: ped_rows.append([mate_id, "0", "0"]); all_animals_in_ped.append(mate_id)
            last_gen_parents.extend(new_mates)


    with open(ped_filepath, "w") as f:
        for row in ped_rows:
            f.write(",".join(row) + "\n")

    pedigree = read_pedigree(ped_filepath)
    A_inv = calculate_A_inverse(pedigree) # Needed for MME, not direct simulation here
    # For simulation, ideally use A matrix, but for simplicity, simulate IID u ~ N(0, sigma_a_true)

    ordered_animal_ids = pedigree.get_ordered_ids_str()
    n_animals = len(ordered_animal_ids)

    true_animal_effects_u = np.random.randn(n_animals) * np.sqrt(sigma_a_true)
    true_animal_effects_map = {aid: u_val for aid, u_val in zip(ordered_animal_ids, true_animal_effects_u)}

    pheno_df = pd.DataFrame({
        'obs_id': ordered_animal_ids, # Assume one obs per animal
        'yield': [intercept_true + true_animal_effects_map[aid] + np.random.randn() * np.sqrt(sigma_e_true)
                  for aid in ordered_animal_ids]
    })
    pheno_df.set_index('obs_id', inplace=True)

    print(f"  Simulated pedigree with {n_animals} animals written to {ped_filepath}")
    print("  Animal effects simulated IID for simplicity (not from A*sigma_a^2).")
    return pheno_df, pedigree, A_inv, ordered_animal_ids, intercept_true, sigma_a_true, sigma_e_true

def main():
    print("--- PyJWAS Animal Model (Pedigree BLUP) Simulation Example ---")
    ped_file = "animal_model_example_ped.csv"

    # 1. Simulate Data
    pheno_df, pedigree_obj, A_inv_matrix, ordered_animal_ids, \
    intercept_true, sigma_a_true, sigma_e_true = \
        simulate_animal_model_data(ped_filepath=ped_file, n_extra_founders=3, n_offspring_gen=3, offspring_per_pairing=2)

    # 2. Build Model
    model_equation = "yield = intercept + animal" # 'animal' is the polygenic effect

    R_vc_obj = VarianceCovariance(value=sigma_e_true*0.9, df=4.0, scale=sigma_e_true*0.9*0.5, estimate_variance=True)
    mme = build_model(model_equation, R_value=R_vc_obj.value, R_df=R_vc_obj.df)
    mme.R_variance = R_vc_obj

    # Add the polygenic animal effect
    G0_prior = VarianceCovariance(value=sigma_a_true*0.9, df=4.0, scale=sigma_a_true*0.9*0.5, estimate_variance=True)
    mme.add_structured_random_effect(
        base_term_name="animal",
        V_inv_matrix=A_inv_matrix,
        level_ids=ordered_animal_ids,
        prior_vc_info=G0_prior,
        random_type_code="A" # "A" for pedigree-based additive genetic effect
    )
    mme.pedigree_data = pedigree_obj # Store pedigree for reference if needed
    print(f"Model built: {mme.model_equations_str}")

    # 3. Set MCMC Parameters
    mme.mcmc_info = MCMCInfo(
        chain_length=10000, burnin=2000,
        output_samples_frequency=100, printout_frequency=1000, seed=101112
    )
    print(f"MCMC parameters set: Chain={mme.mcmc_info.chain_length}, Burn-in={mme.mcmc_info.burnin}")

    # 4. Construct MME Components
    # Phenotype DataFrame index must match the order of animals in H_inv (which is pedigree_obj.ordered_nodes)
    mme.obs_ids = ordered_animal_ids # Set obs_ids for MME based on H_inv order
    pheno_df_ordered = pheno_df.reindex(ordered_animal_ids)

    try:
        print("Building MME components...")
        get_mme_components(mme, pheno_df_ordered)
    except Exception as e:
        print(f"Error during MME component building: {e}"); import traceback; traceback.print_exc(); return

    # 5. Run MCMC
    print("\nStarting Animal Model MCMC run...")
    try:
        results = run_mcmc(mme, pheno_df_ordered)
        print("\nAnimal Model MCMC run completed.")

        # 6. Analyze Results
        print("\n--- Animal Model Results ---")
        intercept_idx = mme.model_term_dict["yield:intercept"].start_pos
        mean_intercept_est = results.get("mean_solutions")[intercept_idx]
        print(f"  True Intercept: {intercept_true:.4f}")
        print(f"  Estimated Mean Intercept: {mean_intercept_est:.4f}")

        mean_res_var_est = results.get("mean_residual_variance")
        print(f"  True Residual Variance (sigma_e^2): {sigma_e_true:.4f}")
        print(f"  Estimated Mean Residual Variance: {mean_res_var_est:.4f}")

        # Polygenic variance (sigma_a^2)
        # Access it from the RandomEffectTerm stored in MME, from its .Gi.value (which is G0_inv)
        animal_re_term = next((rt for rt in mme.random_effect_terms if rt.random_type=="A"), None)
        if animal_re_term and animal_re_term.Gi and animal_re_term.Gi.value is not None:
            # This assumes accumulation of G0_inv's mean is not yet in results dict,
            # so we take the last sampled value.
            # TODO: Accumulate mean of G0_inv (or G0) in MCMC results.
            final_G0_inv_est = animal_re_term.Gi.value
            if isinstance(final_G0_inv_est, (float, np.floating)): # Scalar for ST
                final_G0_est = 1.0 / final_G0_inv_est if final_G0_inv_est > 1e-9 else np.inf
                print(f"  True Polygenic Variance (sigma_a^2): {sigma_a_true:.4f}")
                print(f"  Estimated Polygenic Variance (from last sample of G0_inv): {final_G0_est:.4f}")
            else: # Matrix for MT
                print(f"  Estimated Polygenic Covariance G0_inv (last sample):\n{final_G0_inv_est}")
        else:
            print("  Could not retrieve estimated polygenic variance.")

    except Exception as e:
        print(f"An error occurred during Animal Model MCMC run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(ped_file): os.remove(ped_file)

if __name__ == "__main__":
    import traceback # To see full trace for errors in __main__
    main()

```
