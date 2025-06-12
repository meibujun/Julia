# src/agents/coordinator_agent.py

from src.agents.data_processing_agent import DataProcessingAgent
from src.agents.model_building_agent import ModelBuildingAgent
from src.agents.computational_agent import ComputationalAgent
from src.agents.results_analysis_agent import ResultsAnalysisAgent
from src.core.datastructures import ModelParameters, PhenotypeData, PedigreeData, GenotypeData

class CoordinatorAgent:
    def __init__(self):
        self.data_processing_agent = DataProcessingAgent()
        self.model_building_agent = ModelBuildingAgent()
        self.computational_agent = ComputationalAgent()
        self.results_analysis_agent = ResultsAnalysisAgent()
        print("CoordinatorAgent initialized with all sub-agents.")

    def run_complete_analysis(self,
                              phenotype_filepath: str,
                              pedigree_filepath: str,
                              model_options: dict,
                              genotype_filepath_data: str = None,
                              genotype_filepath_marker_info: str = None
                              ) -> dict:
        """
        Orchestrates a complete analysis workflow.

        Args:
            phenotype_filepath: Path to the phenotype CSV file.
            pedigree_filepath: Path to the pedigree CSV file.
            model_options: A dictionary defining model parameters, e.g.,
                           {
                               'target_trait': 'T1',
                               'fixed_effects': ['herd', 'sex'],
                               'random_effects': ['animal_additive']
                               # Potentially 'genomic' if genotype data is provided
                           }
            genotype_filepath_data: Optional path to genotype data CSV.
            genotype_filepath_marker_info: Optional path to genotype marker info CSV.


        Returns:
            A dictionary containing the analysis results from the ComputationalAgent.
        """
        print("CoordinatorAgent: Starting complete analysis workflow...")

        try:
            # 1. Load Data
            print("CoordinatorAgent: Loading phenotype data...")
            phenotype_data = self.data_processing_agent.load_phenotype_data(phenotype_filepath)
            print("CoordinatorAgent: Loading pedigree data...")
            pedigree_data = self.data_processing_agent.load_pedigree_data(pedigree_filepath)

            genotype_data_obj = None
            if genotype_filepath_data:
                print("CoordinatorAgent: Loading genotype data...")
                genotype_data_obj = self.data_processing_agent.load_genotype_data(
                    filepath_data=genotype_filepath_data,
                    filepath_marker_info=genotype_filepath_marker_info
                )
                # Add genotype_data to model_options if it's to be used by ModelBuildingAgent
                model_options['genotype_data'] = genotype_data_obj


            # 2. Create ModelParameters object
            model_parameters = ModelParameters(model_options)

            # 3. Build Model
            print("CoordinatorAgent: Building model...")
            # The ModelBuildingAgent might need to handle cases with or without genotype_data
            # The current build_animal_model can conditionally add genomic parts
            model_representation = self.model_building_agent.build_animal_model(
                phenotype_data=phenotype_data,
                pedigree_data=pedigree_data,
                model_parameters=model_parameters # This now includes genotype_data if loaded
            )

            # 4. Run Computation
            print("CoordinatorAgent: Running computation...")
            raw_results = self.computational_agent.run_analysis(model_representation)

            # 5. Analyze/Format Results
            print("CoordinatorAgent: Analyzing and formatting results...")
            final_results = self.results_analysis_agent.process_results(raw_results)

            print("CoordinatorAgent: Workflow completed.")
            return final_results

        except FileNotFoundError as fnf_error:
            print(f"CoordinatorAgent Error: Data file not found. {fnf_error}")
            raise
        except ValueError as val_error:
            print(f"CoordinatorAgent Error: Invalid input or data. {val_error}")
            raise
        except Exception as e:
            print(f"CoordinatorAgent Error: An unexpected error occurred during the analysis. {e}")
            raise

# Example Usage (for testing purposes)
if __name__ == '__main__':
    coordinator = CoordinatorAgent()

    # Create dummy CSV files for testing (similar to DataProcessingAgent's test)
    import pandas as pd
    import os
    import numpy as np # Added for NpEncoder if needed here

    # Phenotype
    pheno_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5],
        'trait_id': ['T1', 'T1', 'T1', 'T1', 'T1'],
        'value': [10.1, 11.5, 9.8, 12.0, 10.5],
        'herd': ['H1', 'H2', 'H1', 'H2', 'H1'],
        'sex': ['M', 'F', 'M', 'F', 'M']
    })
    pheno_file = 'dummy_phenotypes_coord.csv'
    pheno_df.to_csv(pheno_file, index=False)

    # Pedigree
    ped_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5, 6],
        'sire_id': [0, 0, 1, 1, 3, 0],
        'dam_id': [0, 0, 2, 2, 4, 0]
    })
    ped_file = 'dummy_pedigree_coord.csv'
    ped_df.to_csv(ped_file, index=False)

    # Genotypes (Optional)
    geno_df = pd.DataFrame({
        'animal_id': [1,2,3,4,5], # Subset of animals
        'markerA': [0,1,2,0,1],
        'markerB': [1,1,0,1,0]
    })
    geno_file = 'dummy_genotypes_coord.csv'
    geno_df.to_csv(geno_file, index=False)


    model_options_animal_model = {
        'target_trait': 'T1', # Should be handled by ModelBuildingAgent if not specified & only one trait
        'fixed_effects': ['herd', 'sex'],
        'random_effects': ['animal_additive']
    }

    model_options_gblup_like = {
        'target_trait': 'T1',
        'fixed_effects': ['herd', 'sex'],
        'random_effects': ['animal_additive', 'genomic'] # 'genomic' will trigger genomic part in model builder
    }

    print("--- Running Analysis (Animal Model) ---")
    try:
        results_animal = coordinator.run_complete_analysis(
            phenotype_filepath=pheno_file,
            pedigree_filepath=ped_file,
            model_options=model_options_animal_model
        )
        print("\nCoordinator - Animal Model Results:")
        import json
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (PhenotypeData, PedigreeData, GenotypeData)): return str(obj) # Avoid circular refs
                return super(NpEncoder, self).default(obj)
        print(json.dumps(results_animal, indent=2, cls=NpEncoder))
    except Exception as e:
        print(f"Error in Coordinator example (Animal Model): {e}")

    print("\n--- Running Analysis (GBLUP-like Model) ---")
    try:
        results_gblup = coordinator.run_complete_analysis(
            phenotype_filepath=pheno_file,
            pedigree_filepath=ped_file,
            model_options=model_options_gblup_like,
            genotype_filepath_data=geno_file
            # marker info file could be added here if needed by DataProcessingAgent & GenotypeData
        )
        print("\nCoordinator - GBLUP-like Model Results:")
        print(json.dumps(results_gblup, indent=2, cls=NpEncoder))

    except Exception as e:
        print(f"Error in Coordinator example (GBLUP-like): {e}")
    finally:
        # Clean up dummy files
        if os.path.exists(pheno_file): os.remove(pheno_file)
        if os.path.exists(ped_file): os.remove(ped_file)
        if os.path.exists(geno_file): os.remove(geno_file)
