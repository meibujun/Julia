from typing import Optional, Union, Any

class MCMCInfo:
    """
    Stores MCMC run parameters and settings.
    Translated from the Julia JWAS.jl MCMCinfo struct.
    """
    def __init__(self,
                 chain_length: int = 1000,
                 burnin: int = 100,
                 output_samples_frequency: int = 10,
                 output_folder: str = "pyjwas_results",
                 seed: Optional[int] = None,
                 heterogeneous_residuals: bool = False,
                 printout_model_info: bool = True,
                 printout_frequency: int = 100,
                 single_step_analysis: bool = False,
                 fitting_J_vector: bool = True, # Specific to SSBR
                 missing_phenotypes: bool = True, # Allow missing phenotypes in multi-trait
                 update_priors_frequency: int = 0, # 0 means never update
                 output_ebv: bool = True,
                 output_heritability: bool = True,
                 prediction_equation: Optional[str] = None,
                 double_precision: bool = True, # Python floats are typically double
                 rrm: bool = False, # Random Regression Model flag or parameters
                 fast_blocks: Union[bool, int, list] = False # For marker block updates
                 ):
        """
        Initializes an MCMCInfo instance.

        Args:
            chain_length: Total number of MCMC iterations.
            burnin: Number of burn-in iterations to discard.
            output_samples_frequency: Frequency to save MCMC samples.
            output_folder: Directory to save results.
            seed: Random seed for reproducibility.
            heterogeneous_residuals: Flag for heterogeneous residual variances.
            printout_model_info: Flag to print model information.
            printout_frequency: Frequency to print MCMC progress/summaries.
            single_step_analysis: Flag for single-step genomic analysis.
            fitting_J_vector: Flag used in single-step Bayesian regression.
            missing_phenotypes: Flag to allow missing phenotypes in multi-trait models.
            update_priors_frequency: Frequency to update priors empirically (if > 0).
            output_ebv: Flag to output Estimated Breeding Values.
            output_heritability: Flag to output heritability estimates.
            prediction_equation: String defining a custom prediction equation.
            double_precision: Flag to use double precision (Python default for float).
            rrm: Flag or parameters for Random Regression Models.
            fast_blocks: Configuration for fast block updates for markers.
                         bool: True to auto-calculate, int for block size, list for specific blocks.
        """
        self.heterogeneous_residuals: bool = heterogeneous_residuals
        self.chain_length: int = chain_length
        self.burnin: int = burnin
        self.output_samples_frequency: int = output_samples_frequency
        self.printout_model_info: bool = printout_model_info
        self.printout_frequency: int = printout_frequency
        self.single_step_analysis: bool = single_step_analysis
        self.fitting_J_vector: bool = fitting_J_vector
        self.missing_phenotypes: bool = missing_phenotypes
        self.update_priors_frequency: int = update_priors_frequency
        self.output_ebv: bool = output_ebv
        self.output_heritability: bool = output_heritability
        self.prediction_equation: Optional[str] = prediction_equation
        self.seed: Optional[int] = seed
        self.double_precision: bool = double_precision # Python floats are double by default
        self.output_folder: str = output_folder
        self.rrm: Any = rrm # Could be bool or more complex parameters for RRM
        self.fast_blocks: Any = fast_blocks # bool, int, or list

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"MCMCInfo({attrs})"

if __name__ == '__main__':
    # Example Usage:
    default_info = MCMCInfo()
    print(default_info)

    custom_info = MCMCInfo(chain_length=20000,
                           burnin=5000,
                           output_folder="my_analysis_run1",
                           seed=12345,
                           single_step_analysis=True)
    print(custom_info)

    minimal_info = MCMCInfo(chain_length=100, burnin=10, output_samples_frequency=1)
    print(minimal_info)
