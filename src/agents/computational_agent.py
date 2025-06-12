# src/agents/computational_agent.py

import numpy as np
import pandas as pd
from src.core.datastructures import PhenotypeData, PedigreeData, GenotypeData

class ComputationalAgent:
    def __init__(self):
        pass

    def run_analysis(self, model_representation: dict) -> dict:
        """
        Runs the statistical analysis based on the model representation.
        For this initial version, it simulates results for an animal model.

        Args:
            model_representation: A dictionary describing the model,
                                  as produced by ModelBuildingAgent.

        Returns:
            A dictionary containing the analysis results.
            Example for a simple animal model:
            {
                'trait': 'T1',
                'fixed_effects_estimates': {
                    'herd_H1': 10.5, # Simulated
                    'herd_H2': 11.2  # Simulated
                },
                'random_effects_estimates': {
                    'animal_breeding_values': { # animal_id: ebv
                        1: 0.5,
                        2: -0.2,
                        # ...
                    }
                },
                'variance_components': { # Placeholder
                    'animal_additive': 1.0, # Simulated
                    'residual': 2.5       # Simulated
                },
                'log_likelihood': -150.0 # Simulated
            }
        """
        print(f"ComputationalAgent: Received model for trait '{model_representation.get('trait', 'N/A')}'")

        trait = model_representation.get('trait')
        phenotype_data_obj: PhenotypeData = model_representation.get('data_references', {}).get('phenotypes')
        pedigree_data_obj: PedigreeData = model_representation.get('data_references', {}).get('pedigree')
        # genotype_data_obj: GenotypeData = model_representation.get('data_references', {}).get('genotypes') # For later use

        if not phenotype_data_obj or not isinstance(phenotype_data_obj.data, pd.DataFrame):
            raise ValueError("Valid phenotype data not found in model representation.")

        phenotypes_df = phenotype_data_obj.data

        # Filter phenotypes for the specific trait
        if trait:
            phenotypes_df = phenotypes_df[phenotypes_df['trait_id'] == trait].copy()
        if phenotypes_df.empty:
            raise ValueError(f"No phenotype data found for trait '{trait}'.")

        results = {
            'trait': trait,
            'fixed_effects_estimates': {},
            'random_effects_estimates': {
                'animal_breeding_values': {}
            },
            'variance_components': {},
            'log_likelihood': np.random.uniform(-200, -100) # Simulated
        }

        # 1. Simulate Fixed Effects Estimates
        # For each fixed effect specified in the model, calculate simple averages.
        # This is a gross simplification.
        fixed_effects_names = model_representation.get('fixed_effects', [])
        for effect_name in fixed_effects_names:
            if effect_name in phenotypes_df.columns:
                # Calculate mean phenotype value per level of the fixed effect
                means = phenotypes_df.groupby(effect_name)['value'].mean()
                for level, mean_val in means.items():
                    results['fixed_effects_estimates'][f'{effect_name}_{level}'] = round(mean_val, 3)
            else:
                print(f"Warning: Fixed effect '{effect_name}' not found in phenotype data columns.")

        # 2. Simulate Random Effects Estimates (Breeding Values)
        # Assign random EBVs to all animals present in the pedigree or phenotype data.
        all_animals = pd.Index([])
        if pedigree_data_obj and isinstance(pedigree_data_obj.data, pd.DataFrame):
            all_animals = all_animals.union(pd.Index(pedigree_data_obj.data['animal_id'].unique()))

        all_animals = all_animals.union(pd.Index(phenotypes_df['animal_id'].unique()))

        for animal_id in all_animals.unique():
            # Ensure animal_id is a Python native type for JSON serialization if it comes from numpy
            animal_id_py = animal_id.item() if isinstance(animal_id, np.generic) else animal_id
            results['random_effects_estimates']['animal_breeding_values'][animal_id_py] = round(np.random.normal(0, 0.5), 3) # Simulated EBV

        # 3. Simulate Variance Components
        if 'animal' in model_representation.get('random_effects', {}):
            results['variance_components']['animal_additive'] = round(np.random.uniform(0.5, 1.5), 3) # Simulated

        # Check if genomic component is expected
        if 'genomic_animal' in model_representation.get('random_effects', {}):
             results['variance_components']['genomic_animal_effect'] = round(np.random.uniform(0.4, 1.2), 3) # Simulated genomic variance

        results['variance_components']['residual'] = round(np.random.uniform(1.0, 3.0), 3) # Simulated

        print(f"ComputationalAgent: Simulated analysis complete for trait '{trait}'.")
        return results

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Dummy data and model representation (normally from other agents)
    pheno_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'trait_id': ['T1', 'T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2', 'T2'],
        'value': [10.1, 11.5, 9.8, 12.0, 10.5, 5.0, 5.5, 4.9, 6.0, 5.2],
        'herd': ['H1', 'H2', 'H1', 'H2', 'H1', 'H1', 'H2', 'H1', 'H2', 'H1'],
        'sex': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'M']
    })
    dummy_phenotypes = PhenotypeData(pheno_df)

    ped_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5, 6], # Animal 6 only in pedigree
        'sire_id': [0, 0, 1, 1, 3, 0],
        'dam_id': [0, 0, 2, 2, 4, 0]
    })
    dummy_pedigree = PedigreeData(ped_df)

    # This would come from ModelBuildingAgent
    dummy_model_repr = {
        'trait': 'T1',
        'fixed_effects': ['herd', 'sex'],
        'random_effects': {
            'animal': {
                'type': 'additive_genetic',
                'covariance_structure': 'A_matrix_from_pedigree'
            }
        },
        'data_references': {
            'phenotypes': dummy_phenotypes,
            'pedigree': dummy_pedigree
        }
    }

    agent = ComputationalAgent()

    print("Running Simulated Analysis...")
    try:
        analysis_results = agent.run_analysis(dummy_model_repr)
        print("\nSimulated Analysis Results:")
        import json
        # Custom encoder for pandas/numpy types if they slip through
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        print(json.dumps(analysis_results, indent=2, cls=NpEncoder))
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
