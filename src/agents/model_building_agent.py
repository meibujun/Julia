# src/agents/model_building_agent.py

from src.core.datastructures import PhenotypeData, PedigreeData, GenotypeData, ModelParameters

class ModelBuildingAgent:
    def __init__(self):
        pass

    def build_animal_model(self,
                           phenotype_data: PhenotypeData,
                           pedigree_data: PedigreeData,
                           model_parameters: ModelParameters) -> dict:
        """
        Builds a representation of a mixed effects animal model.

        Args:
            phenotype_data: An instance of PhenotypeData.
            pedigree_data: An instance of PedigreeData.
            model_parameters: An instance of ModelParameters specifying effects.

        Returns:
            A dictionary representing the model to be fitted.
            Example:
            {
                'trait': 'trait_id_from_phenotypes_or_params',
                'fixed_effects': ['effect1_name', 'effect2_name'], # from model_parameters
                'random_effects': {
                    'animal': {
                        'type': 'additive_genetic',
                        'covariance_matrix': 'A_matrix_from_pedigree' # Symbolically, actual matrix built later
                    }
                    # Potentially other random effects like 'permanent_environment'
                },
                'data_references': { # References to the data to be used
                    'phenotypes': phenotype_data,
                    'pedigree': pedigree_data
                }
            }
        """
        if not isinstance(phenotype_data, PhenotypeData):
            raise TypeError("phenotype_data must be an instance of PhenotypeData.")
        if not isinstance(pedigree_data, PedigreeData):
            raise TypeError("pedigree_data must be an instance of PedigreeData.")
        if not isinstance(model_parameters, ModelParameters):
            raise TypeError("model_parameters must be an instance of ModelParameters.")

        model_representation = {}

        # Determine the trait to analyze.
        # This could come from model_parameters or by inspecting phenotype_data if only one trait exists.
        # For simplicity, let's assume model_parameters specifies the target trait.
        # Or, if phenotype_data.data['trait_id'] has only one unique value, use that.
        unique_traits = phenotype_data.data['trait_id'].unique()
        if len(unique_traits) == 1:
            model_representation['trait'] = unique_traits[0]
        elif 'target_trait' in model_parameters.parameters:
            model_representation['trait'] = model_parameters.parameters['target_trait']
        else:
            raise ValueError("Target trait for the model is ambiguous or not specified.")

        # Get fixed effects from model_parameters
        model_representation['fixed_effects'] = model_parameters.parameters.get('fixed_effects', [])

        # Define random effects
        model_representation['random_effects'] = {}
        random_effects_specs = model_parameters.parameters.get('random_effects', [])

        if 'animal_additive' in random_effects_specs:
            model_representation['random_effects']['animal'] = {
                'type': 'additive_genetic',
                'covariance_structure': 'A_matrix_from_pedigree' # Placeholder
                # Variance components (priors or estimates) would also go here later
            }
        # We could add more types of random effects here based on model_parameters
        # e.g., permanent environment, other genetic effects.

        # Store references to the data that will be needed for this model
        model_representation['data_references'] = {
            'phenotypes': phenotype_data,
            'pedigree': pedigree_data
            # If GBLUP, genotype_data would be referenced here too.
        }

        # Potentially include genotype data reference if specified for GBLUP etc.
        if 'genomic' in random_effects_specs and 'genotype_data' in model_parameters.parameters:
            genotype_data_ref = model_parameters.parameters['genotype_data']
            if not isinstance(genotype_data_ref, GenotypeData):
                raise TypeError("genotype_data in model_parameters must be an instance of GenotypeData.")
            model_representation['random_effects']['genomic_animal'] = {
                'type': 'genomic_relationship',
                'covariance_structure': 'G_matrix_from_genotypes' # Placeholder
            }
            model_representation['data_references']['genotypes'] = genotype_data_ref


        print(f"Model representation for trait '{model_representation['trait']}' built successfully.")
        return model_representation

    # We can add other model building methods later, e.g., build_gblup_model, build_bayesian_model

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Dummy data (in a real scenario, this would come from DataProcessingAgent)
    import pandas as pd
    pheno_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5],
        'trait_id': ['T1', 'T1', 'T1', 'T1', 'T1'],
        'value': [10.1, 11.5, 9.8, 12.0, 10.5],
        'herd': ['H1', 'H2', 'H1', 'H2', 'H1'] # Example fixed effect
    })
    dummy_phenotypes = PhenotypeData(pheno_df)

    ped_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5],
        'sire_id': [0, 0, 1, 1, 3],
        'dam_id': [0, 0, 2, 2, 4]
    })
    dummy_pedigree = PedigreeData(ped_df)

    # Define model parameters
    # Scenario 1: Simple animal model
    params_animal_model = ModelParameters({
        'target_trait': 'T1',
        'fixed_effects': ['herd'],
        'random_effects': ['animal_additive']
    })

    agent = ModelBuildingAgent()

    print("Building Animal Model...")
    try:
        animal_model_repr = agent.build_animal_model(dummy_phenotypes, dummy_pedigree, params_animal_model)
        print("Animal Model Representation:")
        import json
        print(json.dumps(animal_model_repr, default=lambda o: '<object>', indent=2))
    except Exception as e:
        print(f"Error building animal model: {e}")

    # Scenario 2: Model with genomic data (placeholder for GBLUP)
    geno_df = pd.DataFrame({
        'animal_id': [1,2,3,4,5],
        'marker1': [0,1,2,0,1],
        'marker2': [1,1,0,1,0]
    })
    dummy_genotypes = GenotypeData(geno_df, marker_info=None)

    params_gblup_model = ModelParameters({
        'target_trait': 'T1',
        'fixed_effects': ['herd'],
        'random_effects': ['animal_additive', 'genomic'], # 'genomic' implies GBLUP/ssGBLUP context
        'genotype_data': dummy_genotypes # Pass the genotype data object
    })

    print("\nBuilding GBLUP-like Model...")
    try:
        gblup_model_repr = agent.build_animal_model(dummy_phenotypes, dummy_pedigree, params_gblup_model)
        print("GBLUP Model Representation:")
        print(json.dumps(gblup_model_repr, default=lambda o: '<object>', indent=2))
    except Exception as e:
        print(f"Error building GBLUP model: {e}")
