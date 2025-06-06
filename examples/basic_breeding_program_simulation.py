# examples/basic_breeding_program_simulation.py

import pandas as pd
import numpy as np
import random

# Import from our sheep_breeding_genomics library
# Assuming the script is run from a location where 'sheep_breeding_genomics' is in PYTHONPATH
# or using an appropriate execution method (e.g., python -m examples.basic_breeding_program_simulation)
# For simplicity if running script directly from `examples` dir and `sheep_breeding_genomics` is one level up:
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sheep_breeding_genomics.data_management.data_structures import PedigreeData, PhenotypicData, GenomicData
from sheep_breeding_genomics.data_management.io_handlers import read_phenotypic_data # Placeholder, not used for generation here
from sheep_breeding_genomics.data_management.validators import validate_pedigree_data # Placeholder

from sheep_breeding_genomics.genetic_evaluation.relationship_matrix import calculate_nrm, calculate_grm, calculate_h_inverse_matrix
from sheep_breeding_genomics.genetic_evaluation.blup_models import solve_animal_model_mme, solve_ssgblup_model_mme

from sheep_breeding_genomics.breeding_strategies.selection import rank_animals, select_top_n, select_top_percent
from sheep_breeding_genomics.breeding_strategies.mating_schemes import random_mating, calculate_progeny_inbreeding, generate_mating_list_with_inbreeding

def simulate_phenotype(true_bv, overall_mean, heritability, phenotypic_variance):
    """
    Simulates a phenotypic value.
    phenotype = overall_mean + true_bv + random_error
    var(error) = var(phenotype) * (1 - heritability)
    true_bv is assumed to be deviation from mean.
    """
    # This is a simplified model. A more correct one:
    # Phenotype = mu + TBV + E, where E ~ N(0, sigma_e^2)
    # sigma_p^2 = sigma_g^2 + sigma_e^2
    # h^2 = sigma_g^2 / sigma_p^2
    # sigma_e^2 = sigma_g^2 * (1-h^2)/h^2
    # If TBV is provided, sigma_g^2 is var(TBV).
    # For simulation, if we have TBV, then E ~ N(0, phenotypic_variance * (1-heritability))
    # Or, if var_genetic is var(TBV): error_variance = var_genetic * (1-heritability) / heritability (if h2 > 0)

    if heritability > 0 and heritability < 1:
        error_variance = phenotypic_variance * (1 - heritability)
    elif heritability == 1:
        error_variance = 0
    else: # heritability == 0
        error_variance = phenotypic_variance

    random_error = np.random.normal(0, np.sqrt(error_variance)) if error_variance > 0 else 0
    return overall_mean + true_bv + random_error


def main_simulation():
    """
    Main function to run the basic breeding program simulation.
    """
    print("--- Basic Sheep Breeding Program Simulation ---")

    # --- 1. Simulation Parameters ---
    N_GENERATIONS = 3       # Number of generations to simulate
    N_SIRES_SELECT = 2      # Number of sires to select each generation
    N_DAMS_SELECT = 5       # Number of dams to select each generation
    N_PROGENY_PER_MATING = 2 # Number of progeny per mating pair

    # Phenotype simulation parameters
    TRAIT_MEAN = 10.0       # Overall mean for the trait
    HERITABILITY = 0.30     # Heritability of the trait
    PHENOTYPIC_SD = 2.0     # Phenotypic standard deviation for the trait
    PHENOTYPIC_VARIANCE = PHENOTYPIC_SD**2
    GENETIC_VARIANCE = PHENOTYPIC_VARIANCE * HERITABILITY
    RESIDUAL_VARIANCE = PHENOTYPIC_VARIANCE * (1 - HERITABILITY)

    # Initial animal ID counter
    animal_id_counter = 0

    # Lists to store results per generation
    avg_ebv_selected_sires_gens = []
    avg_ebv_selected_dams_gens = []
    avg_ebv_progeny_gens = []
    avg_inbreeding_progeny_gens = []

    # --- 2. Initialization (Generation 0) ---
    print("\n--- Initializing Generation 0 ---")

    base_pop_size = N_SIRES_SELECT + N_DAMS_SELECT # Keep it small for example
    current_pedigree_records = []
    current_phenotype_records = []

    # Create base animals (founders)
    base_animals_ids = []
    base_animals_true_bvs = {} # Store true breeding values for simulation reference

    for i in range(base_pop_size):
        animal_id_counter += 1
        animal_id = f"A{animal_id_counter}"
        base_animals_ids.append(animal_id)

        # Simulate True BV for founders (drawn from N(0, GENETIC_VARIANCE))
        true_bv = np.random.normal(0, np.sqrt(GENETIC_VARIANCE))
        base_animals_true_bvs[animal_id] = true_bv

        # Pedigree: founders have unknown parents (0 or None)
        current_pedigree_records.append({'AnimalID': animal_id, 'SireID': 0, 'DamID': 0})

        # Phenotype for founders
        pheno_val = simulate_phenotype(true_bv, TRAIT_MEAN, HERITABILITY, PHENOTYPIC_VARIANCE)
        current_phenotype_records.append({'AnimalID': animal_id, 'TraitValue': pheno_val})

    pedigree_df = pd.DataFrame(current_pedigree_records)
    phenotypic_df = pd.DataFrame(current_phenotype_records)

    # Using PedigreeData and PhenotypicData objects (optional for internal sim, but good practice)
    ped_obj = PedigreeData(pedigree_df)
    phen_obj = PhenotypicData(phenotypic_df)

    print(f"Base population created with {len(base_animals_ids)} animals.")
    print("Base Pedigree:")
    print(ped_obj.data.head())
    print("\nBase Phenotypes:")
    print(phen_obj.data.head())

    # For simplicity, this basic simulation will use PBLUP with NRM.
    # GRM/ssGBLUP could be added later.

    # --- 3. Simulation Loop ---
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n--- Simulating Generation {gen} ---")

        # --- 3.1 Genetic Evaluation (PBLUP) ---
        print(f"Running genetic evaluation for Gen {gen-1} animals...")
        if ped_obj.data.empty or phen_obj.data.empty:
            print("Pedigree or phenotypic data is empty. Skipping evaluation.")
            # Potentially stop simulation or handle this state
            break

        nrm_df = calculate_nrm(ped_obj.data, animal_col='AnimalID', sire_col='SireID', dam_col='DamID', founder_val=0)
        if nrm_df.empty:
            print("NRM calculation failed. Skipping evaluation.")
            break

        # Ensure all animals in phenotype_df are in nrm_df
        # This requires aligning animals that have phenotypes with NRM for MME solver
        # The solver itself handles this alignment.

        # For this simulation, variances are known (from parameters)
        ebv_df = solve_animal_model_mme(
            phenotypic_df=phen_obj.data,
            relationship_matrix_df=nrm_df,
            trait_col='TraitValue',
            animal_id_col='AnimalID',
            var_animal=GENETIC_VARIANCE,
            var_residual=RESIDUAL_VARIANCE
        )

        if ebv_df.empty:
            print("EBV calculation failed. Stopping simulation.")
            break

        print(f"EBVs calculated for {len(ebv_df)} animals.")
        # print(ebv_df.head())


        # --- 3.2 Selection ---
        print("Selecting parents...")
        # Rank all animals with EBVs
        # The EBV df contains all animals from NRM. We need to select from those alive and evaluated.
        # For simplicity, assume all animals in ped_obj.data are candidates if they have EBVs.

        # Merge EBVs with current animal list (ped_obj.data) to ensure we only rank relevant animals
        # Or, more simply, the ebv_df from solver contains all relevant IDs.

        ranked_animals_df = rank_animals(ebv_df, ebv_col='EBV', animal_id_col='AnimalID')

        # For selection, we might only consider animals from the most recent generation(s)
        # or all available. Here, select from all for simplicity.

        # Simplistic: Assume sires and dams can be selected from the same pool of ranked animals.
        # In reality, sex would matter. For this example, we just take top N.
        selected_sires = select_top_n(ranked_animals_df, N_SIRES_SELECT)
        # To select dams, ensure they are different from sires if that's a rule.
        # Here, we select from the same ranked list, but after removing already selected sires.
        remaining_for_dams = ranked_animals_df[~ranked_animals_df['AnimalID'].isin(selected_sires['AnimalID'])]
        selected_dams = select_top_n(remaining_for_dams, N_DAMS_SELECT)

        if selected_sires.empty or selected_dams.empty:
            print("Selection failed to yield enough sires or dams. Stopping simulation.")
            break

        print(f"Selected {len(selected_sires)} sires and {len(selected_dams)} dams.")
        avg_ebv_sires = selected_sires['EBV'].mean()
        avg_ebv_dams = selected_dams['EBV'].mean()
        avg_ebv_selected_sires_gens.append(avg_ebv_sires)
        avg_ebv_selected_dams_gens.append(avg_ebv_dams)
        print(f"Average EBV of selected sires: {avg_ebv_sires:.3f}")
        print(f"Average EBV of selected dams: {avg_ebv_dams:.3f}")


        # --- 3.3 Mating ---
        print("Generating mating list...")
        # Using animal IDs directly from the selected DataFrames
        mating_pairs_progeny_info = random_mating(
            selected_sires_df=selected_sires[['AnimalID']], # Pass DataFrame with ID column
            selected_dams_df=selected_dams[['AnimalID']],   # Pass DataFrame with ID column
            sire_id_col='AnimalID',
            dam_id_col='AnimalID',
            n_progeny_per_mating=N_PROGENY_PER_MATING,
            # n_matings_per_sire can be set if desired
        )

        if not mating_pairs_progeny_info:
            print("No mating pairs generated. Stopping simulation.")
            break

        # Calculate expected inbreeding of progeny
        mating_list_inbreeding_df = generate_mating_list_with_inbreeding(
            mating_pairs_progeny_info, nrm_df
        )
        avg_prog_inbreeding = mating_list_inbreeding_df['ProgenyInbreeding'].mean()
        avg_inbreeding_progeny_gens.append(avg_prog_inbreeding)
        print(f"Generated {len(mating_list_inbreeding_df)} potential progeny.")
        print(f"Average expected progeny inbreeding: {avg_prog_inbreeding:.4f}")
        # print(mating_list_inbreeding_df.head())


        # --- 3.4 Progeny Generation ---
        print("Generating progeny...")
        new_pedigree_records = []
        new_phenotype_records = []
        progeny_true_bvs = {} # Store True BVs for the new generation

        for index, mating in mating_list_inbreeding_df.iterrows():
            sire_id = mating['SireID']
            dam_id = mating['DamID']
            # progeny_num_in_mating = mating['Info1'] # Assuming 'Info1' is progeny_number

            animal_id_counter += 1
            progeny_id = f"A{animal_id_counter}"

            new_pedigree_records.append({'AnimalID': progeny_id, 'SireID': sire_id, 'DamID': dam_id})

            # Simulate True BV for progeny: 0.5*(TBV_sire + TBV_dam) + Mendelian_sampling_term
            # For simplicity in simulation, we use parent EBVs as proxy for their genetic merit passed on.
            # A proper simulation would use underlying True BVs.
            # Here, let's use the True BVs of parents if available (founders), or their EBVs as estimate.
            sire_true_bv = base_animals_true_bvs.get(sire_id, selected_sires[selected_sires['AnimalID']==sire_id]['EBV'].iloc[0])
            dam_true_bv = base_animals_true_bvs.get(dam_id, selected_dams[selected_dams['AnimalID']==dam_id]['EBV'].iloc[0])

            parent_avg_bv = 0.5 * (sire_true_bv + dam_true_bv)

            # Mendelian sampling term: N(0, 0.5 * (1 - avg_F_parents) * GENETIC_VARIANCE)
            # For simplicity, let F_sire and F_dam be 0 for this basic simulation part.
            # So, variance of Mendelian term is ~0.5 * GENETIC_VARIANCE if parents unrelated/not inbred.
            # A more accurate F_parent_avg would use parent inbreeding from NRM diagonal.
            mendelian_sampling_variance = 0.5 * GENETIC_VARIANCE # Simplified
            mendelian_sample_effect = np.random.normal(0, np.sqrt(mendelian_sampling_variance))

            progeny_true_bv = parent_avg_bv + mendelian_sample_effect
            progeny_true_bvs[progeny_id] = progeny_true_bv # Store for next gen reference

            # Simulate phenotype for progeny
            progeny_pheno = simulate_phenotype(progeny_true_bv, TRAIT_MEAN, HERITABILITY, PHENOTYPIC_VARIANCE)
            new_phenotype_records.append({'AnimalID': progeny_id, 'TraitValue': progeny_pheno})

        # Update master pedigree and phenotype lists/DataFrames for the next generation
        ped_obj.data = pd.concat([ped_obj.data, pd.DataFrame(new_pedigree_records)], ignore_index=True)
        phen_obj.data = pd.concat([phen_obj.data, pd.DataFrame(new_phenotype_records)], ignore_index=True)

        # Update the store of true BVs (for next generation's parent selection reference)
        base_animals_true_bvs.update(progeny_true_bvs)

        # Report average EBV of this generation's progeny (using their True BVs for this simulation metric)
        avg_progeny_true_bv = np.mean(list(progeny_true_bvs.values()))
        avg_ebv_progeny_gens.append(avg_progeny_true_bv)
        print(f"Progeny generated for Gen {gen}. Average True BV of progeny: {avg_progeny_true_bv:.3f}")

        # Clean up: Keep only current and maybe parent generation for phenotype data to keep it manageable
        # For this example, we'll let it grow. In a real sim, manage data size.
        # phen_obj.data = phen_obj.data[phen_obj.data['AnimalID'].isin(ped_obj.data['AnimalID'].unique())]


    # --- 4. Post-Simulation Summary ---
    print("\n--- Simulation Complete ---")
    print("Summary of Genetic Trend (Average True BV of Progeny per Generation):")
    for i, avg_bv in enumerate(avg_ebv_progeny_gens):
        print(f"Generation {i+1} Progeny Avg True BV: {avg_bv:.3f}")

    print("\nSummary of Selected Parent EBVs (Sire EBVs):")
    for i, avg_bv in enumerate(avg_ebv_selected_sires_gens):
        print(f"Generation {i+1} Selected Sires Avg EBV: {avg_bv:.3f}")

    print("\nSummary of Average Progeny Inbreeding Coeff per Generation:")
    for i, avg_inb in enumerate(avg_inbreeding_progeny_gens):
        print(f"Generation {i+1} Avg Progeny Inbreeding: {avg_inb:.4f}")

    # Further analysis could involve plotting these trends.

if __name__ == '__main__':
    # Set a seed for reproducibility of the simulation's random parts
    np.random.seed(42)
    random.seed(42)

    main_simulation()
