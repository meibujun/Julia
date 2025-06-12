# src/main.py

import argparse
import json
import pandas as pd # For displaying DataFrame results nicely
import numpy as np # For NpEncoder
from src.agents.coordinator_agent import CoordinatorAgent
from src.core.datastructures import PhenotypeData, PedigreeData, GenotypeData # For NpEncoder if needed for other parts

# For prettier printing of DataFrames if they are part of the final output
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 15)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records') # Or other suitable format
        if isinstance(obj, (PhenotypeData, PedigreeData, GenotypeData)): # Should not happen with ResultsAnalysisAgent
             return str(obj)
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="AI Agents for Livestock Breeding Analysis CLI")

    parser.add_argument("--phenotypes", type=str, required=True, help="Path to the phenotype CSV file.")
    parser.add_argument("--pedigree", type=str, required=True, help="Path to the pedigree CSV file.")
    parser.add_argument("--genotypes_data", type=str, help="Optional: Path to the genotype data CSV file.")
    parser.add_argument("--genotypes_marker_info", type=str, help="Optional: Path to the genotype marker info CSV file.")

    parser.add_argument("--trait", type=str, required=True, help="Name of the trait to analyze (must exist in phenotype file's 'trait_id' column).")
    parser.add_argument("--fixed_effects", type=str, nargs='+', help="List of fixed effect column names from the phenotype file (e.g., herd sex).")
    parser.add_argument("--random_effects", type=str, nargs='+', default=['animal_additive'],
                        help="List of random effects to include (e.g., animal_additive genomic). Default: animal_additive.")

    args = parser.parse_args()

    print("Starting AI Agent System for Livestock Breeding Analysis...")

    coordinator = CoordinatorAgent()

    model_options = {
        'target_trait': args.trait,
        'fixed_effects': args.fixed_effects if args.fixed_effects else [],
        'random_effects': args.random_effects
    }

    try:
        print(f"Analysis Parameters:")
        print(f"  Phenotype File: {args.phenotypes}")
        print(f"  Pedigree File: {args.pedigree}")
        if args.genotypes_data:
            print(f"  Genotype Data File: {args.genotypes_data}")
        if args.genotypes_marker_info:
            print(f"  Genotype Marker Info File: {args.genotypes_marker_info}")
        print(f"  Trait: {args.trait}")
        print(f"  Fixed Effects: {model_options['fixed_effects']}")
        print(f"  Random Effects: {model_options['random_effects']}")
        print("-" * 30)

        results = coordinator.run_complete_analysis(
            phenotype_filepath=args.phenotypes,
            pedigree_filepath=args.pedigree,
            model_options=model_options,
            genotype_filepath_data=args.genotypes_data,
            genotype_filepath_marker_info=args.genotypes_marker_info
        )

        print("\n" + "="*20 + " ANALYSIS RESULTS " + "="*20)
        if results and 'summary_text' in results:
            print(results['summary_text'])

            if 'top_animals_ebv' in results and not results['top_animals_ebv'].empty:
                print("\nTop Animals by EBV:")
                print(results['top_animals_ebv'].to_string(index=False))
            else:
                print("\nNo EBV data to display for top animals.")
        else:
            print("Analysis completed, but no summary text found in results.")
            print("Full results object (JSON):")
            print(json.dumps(results, indent=2, cls=NpEncoder))

        print("\n" + "="*58)
        print("CLI analysis run finished.")

    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e.filename}")
    except ValueError as e:
        print(f"ERROR: Invalid value or configuration: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed debugging if needed

if __name__ == '__main__':
    # Create dummy files for CLI testing convenience
    # These would be the same files used by CoordinatorAgent's example,
    # but CLI needs them to be present before running.

    import os
    pheno_file_cli = 'dummy_phenotypes_cli.csv'
    ped_file_cli = 'dummy_pedigree_cli.csv'
    geno_file_cli = 'dummy_genotypes_cli.csv'

    if not os.path.exists(pheno_file_cli):
        pheno_df_cli = pd.DataFrame({
            'animal_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'trait_id': ['T1', 'T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2', 'T2'],
            'value': [10.1, 11.5, 9.8, 12.0, 10.5, 200, 210, 190, 220, 195],
            'herd': ['H1', 'H2', 'H1', 'H2', 'H1', 'H1', 'H2', 'H1', 'H2', 'H1'],
            'sex': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'M']
        })
        pheno_df_cli.to_csv(pheno_file_cli, index=False)
        print(f"Created dummy phenotype file: {pheno_file_cli}")

    if not os.path.exists(ped_file_cli):
        ped_df_cli = pd.DataFrame({
            'animal_id': [1, 2, 3, 4, 5, 6],
            'sire_id': [0, 0, 1, 1, 3, 0],
            'dam_id': [0, 0, 2, 2, 4, 0]
        })
        ped_df_cli.to_csv(ped_file_cli, index=False)
        print(f"Created dummy pedigree file: {ped_file_cli}")

    if not os.path.exists(geno_file_cli):
        geno_df_cli = pd.DataFrame({
            'animal_id': [1,2,3,4,5], # Subset of animals
            'SNP1': [0,1,2,0,1],
            'SNP2': [1,1,0,1,0],
            'SNP3': [2,2,1,0,1]
        })
        geno_df_cli.to_csv(geno_file_cli, index=False)
        print(f"Created dummy genotype file: {geno_file_cli}")

    print("\nTo run an example analysis from the command line:")
    print(f"python src/main.py --phenotypes {pheno_file_cli} --pedigree {ped_file_cli} --trait T1 --fixed_effects herd sex")
    print("Or for a GBLUP-like (simulated) analysis:")
    print(f"python src/main.py --phenotypes {pheno_file_cli} --pedigree {ped_file_cli} --genotypes_data {geno_file_cli} --trait T1 --fixed_effects herd --random_effects animal_additive genomic")
    print("\nRunning main() function now with default example if no args provided (for dev testing):")

    # This part allows running directly in an IDE for quick tests without CLI args,
    # but true CLI testing should be from the terminal.
    import sys
    if len(sys.argv) == 1: # No CLI arguments provided
        print("\nSimulating a CLI run within main.py for development testing...")
        sys.argv.extend([
            '--phenotypes', pheno_file_cli,
            '--pedigree', ped_file_cli,
            '--trait', 'T1',
            '--fixed_effects', 'herd', 'sex'
        ])
        # To test GBLUP like, uncomment below and comment above
        # sys.argv.extend([
        #     '--phenotypes', pheno_file_cli,
        #     '--pedigree', ped_file_cli,
        #     '--genotypes_data', geno_file_cli,
        #     '--trait', 'T2',
        #     '--fixed_effects', 'herd',
        #     '--random_effects', 'animal_additive', 'genomic'
        # ])

    main()

    # Clean up dummy files after main() if they were created by this script block
    # This is tricky because main() could be called with external files.
    # For simplicity, we'll leave them for now. User can delete dummy_*.csv files.
    # Consider adding a --cleanup_dummy_files argument for CLI tests if needed.
