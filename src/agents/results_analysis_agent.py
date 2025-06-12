# src/agents/results_analysis_agent.py

import pandas as pd

class ResultsAnalysisAgent:
    def __init__(self):
        print("ResultsAnalysisAgent initialized.")

    def process_results(self, raw_results: dict, top_n_ebv: int = 5) -> dict:
        """
        Processes raw analysis results from the ComputationalAgent.
        For this basic version, it extracts key information and can provide a summary
        of breeding values.

        Args:
            raw_results: The dictionary output from ComputationalAgent.run_analysis.
            top_n_ebv: Number of top animals by EBV to list in the summary.

        Returns:
            A dictionary containing processed or summarized results.
            Example:
            {
                'trait': 'T1',
                'summary_text': "Analysis for trait T1 completed. ...",
                'fixed_effects': { ... },
                'variance_components': { ... },
                'top_animals_ebv': pd.DataFrame(...) or list of tuples,
                'all_breeding_values': { ... }
            }
        """
        print(f"ResultsAnalysisAgent: Processing results for trait '{raw_results.get('trait', 'N/A')}'")

        processed_output = {
            'trait': raw_results.get('trait'),
            'fixed_effects_estimates': raw_results.get('fixed_effects_estimates', {}),
            'variance_components': raw_results.get('variance_components', {}),
            'log_likelihood': raw_results.get('log_likelihood')
        }

        ebv_estimates = raw_results.get('random_effects_estimates', {}).get('animal_breeding_values', {})
        processed_output['all_breeding_values'] = ebv_estimates

        summary_lines = [
            f"Analysis Summary for Trait: {processed_output['trait']}",
            f"Log-Likelihood: {processed_output['log_likelihood']:.2f}",
            "Estimated Variance Components:"
        ]
        for component, value in processed_output['variance_components'].items():
            summary_lines.append(f"  - {component}: {value:.3f}")

        summary_lines.append("Estimated Fixed Effects:")
        for effect, value in processed_output['fixed_effects_estimates'].items():
            summary_lines.append(f"  - {effect}: {value:.3f}")

        if ebv_estimates:
            # Sort EBVs to find top N
            sorted_ebvs = sorted(ebv_estimates.items(), key=lambda item: item[1], reverse=True)

            top_ebvs_list = []
            summary_lines.append(f"Top {min(top_n_ebv, len(sorted_ebvs))} Animals by Estimated Breeding Value (EBV):")
            for i, (animal_id, ebv) in enumerate(sorted_ebvs):
                if i < top_n_ebv:
                    summary_lines.append(f"  - Animal ID {animal_id}: {ebv:.3f}")
                    top_ebvs_list.append({'animal_id': animal_id, 'ebv': ebv})

            # Store as a DataFrame or list of dicts
            processed_output['top_animals_ebv'] = pd.DataFrame(top_ebvs_list) if top_ebvs_list else pd.DataFrame()

            # Add mean EBV to summary
            mean_ebv = pd.Series(list(ebv_estimates.values())).mean()
            summary_lines.append(f"Mean EBV for all animals: {mean_ebv:.3f}")

        else:
            summary_lines.append("No breeding value estimates found.")
            processed_output['top_animals_ebv'] = pd.DataFrame()

        processed_output['summary_text'] = "\n".join(summary_lines)

        print(f"ResultsAnalysisAgent: Processing complete for trait '{processed_output['trait']}'.")
        print(processed_output['summary_text']) # Print summary for now

        return processed_output

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Dummy raw results (normally from ComputationalAgent)
    dummy_raw_results = {
        'trait': 'T1',
        'fixed_effects_estimates': {
            'herd_H1': 10.512,
            'herd_H2': 11.198,
            'sex_M': 0.123,
            'sex_F': -0.111
        },
        'random_effects_estimates': {
            'animal_breeding_values': {
                1: 0.532, 2: -0.211, 3: 0.789, 4: 0.011, 5: -0.500, 6: 1.23
            }
        },
        'variance_components': {
            'animal_additive': 0.987,
            'residual': 2.512
        },
        'log_likelihood': -155.789
    }

    agent = ResultsAnalysisAgent()

    print("Processing Simulated Raw Results...")
    try:
        processed_data = agent.process_results(dummy_raw_results, top_n_ebv=3)
        print("\nProcessed Results (details):")
        # print(processed_data) # This will print the whole dict including summary text again
        print(f"Trait: {processed_data['trait']}")
        print("Fixed Effects:")
        for k,v in processed_data['fixed_effects_estimates'].items(): print(f"  {k}: {v}")
        print("Variance Components:")
        for k,v in processed_data['variance_components'].items(): print(f"  {k}: {v}")
        print("Top Animals by EBV (DataFrame):")
        print(processed_data['top_animals_ebv'])

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
