"""
Provides classes and functions for performing Markov Chain Monte Carlo (MCMC)
sampling for genomic analyses, including a base sampler and a specialized
GBLUP (Genomic Best Linear Unbiased Prediction) sampler.
"""
import numpy as np
import pandas as pd # Added for get_ebv_summary
from scipy.stats import invgamma, multivariate_normal
# Assuming model.ModelDefinition and data.PhenotypeData will be imported when used.
# from ..model import ModelDefinition # Type hint
# from ..data import PhenotypeData, GenotypeData # Type hint


class BaseMCMCSampler:
    """
    A base class for MCMC samplers.

    This class provides the basic structure for an MCMC sampler, including
    initialization of parameters, running the sampling loop, storing samples,
    and retrieving posterior summaries. It is intended to be subclassed by
    specific sampler implementations.

    Attributes:
        model_definition (ModelDefinition): The model specification.
        phenotype_data (PhenotypeData): The phenotype data.
        n_iterations (int): Total number of MCMC iterations.
        burn_in (int): Number of burn-in iterations to discard.
        samples (dict): A dictionary to store posterior samples of parameters.
                        Keys are parameter names (e.g., 'beta', 'g', 'variance_components'),
                        values are lists of samples.
        beta (np.ndarray): Current sample of fixed effect coefficients.
        g (np.ndarray): Current sample of random genetic effects (if applicable).
        variance_components (dict): Current samples of variance components.
    """
    def __init__(self, model_definition: 'ModelDefinition', phenotype_data: 'PhenotypeData', 
                 n_iterations: int, burn_in: int):
        """
        Initializes the BaseMCMCSampler.

        Args:
            model_definition (ModelDefinition): An instance of ModelDefinition
                                                containing the model specification.
            phenotype_data (PhenotypeData): An instance of PhenotypeData
                                            containing the phenotype observations.
            n_iterations (int): The total number of MCMC iterations to run.
            burn_in (int): The number of initial iterations to discard (burn-in period).
        
        Raises:
            TypeError: If `model_definition` or `phenotype_data` are not of the expected type.
            ValueError: If `n_iterations` or `burn_in` are invalid (e.g., non-positive,
                        burn_in >= n_iterations).
        """
        # These checks remain, but actual class types would be better
        if not hasattr(model_definition, 'dependent_vars'): 
            raise TypeError("model_definition does not appear to be a valid ModelDefinition instance.")
        if not hasattr(phenotype_data, 'get_y_vector'): # Check for a method expected in PhenotypeData
            raise TypeError("phenotype_data does not appear to be a valid PhenotypeData instance.")
        if not isinstance(n_iterations, int) or n_iterations <= 0:
            raise ValueError("n_iterations must be a positive integer.")
        if not isinstance(burn_in, int) or burn_in < 0:
            raise ValueError("burn_in must be a non-negative integer.")
        if burn_in >= n_iterations:
            raise ValueError("burn_in must be less than n_iterations.")

        self.model_definition = model_definition
        self.phenotype_data = phenotype_data
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        
        self.samples = {
            'beta': [],
            'g': [],
            'variance_components': []
        } 
        
        self.beta = None 
        self.g = None  
        self.variance_components = {}

    def _initialize_parameters(self):
        print("BaseMCMCSampler: _initialize_parameters called (no specific initialization).")
        pass

    def _sample_parameters(self):
        print(f"BaseMCMCSampler: _sample_parameters called (no specific sampling).")
        pass

    def _store_sample(self, iteration: int):
        if iteration > self.burn_in:
            if self.beta is not None:
                self.samples['beta'].append(np.copy(self.beta))
            if self.g is not None:
                 self.samples['g'].append(np.copy(self.g))
            if self.variance_components:
                self.samples['variance_components'].append(self.variance_components.copy())
        # No print here to avoid excessive output during run
        pass

    def run(self) -> dict:
        """
        Runs the MCMC sampling process.

        The process involves:
        1. Initializing parameters using `_initialize_parameters()`.
        2. Iterating `n_iterations` times:
            a. Sampling parameters using `_sample_parameters()`.
            b. Storing samples (if past burn-in) using `_store_sample()`.
        Prints progress messages during the run.

        Returns:
            dict: A dictionary containing the collected posterior samples for each
                  parameter, filtered to remove parameters with no stored samples.
                  The structure is `{'param_name': [sample1, sample2, ...]}`.
                  Note: This return is from the internal storage; `run_mcmc` wrapper
                  returns the sampler instance.
        """
        print(f"Starting MCMC sampling for {self.n_iterations} iterations with {self.burn_in} burn-in using {self.__class__.__name__}...")
        
        self._initialize_parameters()
        
        for i in range(1, self.n_iterations + 1):
            if i % (max(1, self.n_iterations // 10)) == 0 or i == 1 or i == self.n_iterations: # Print progress ~10 times
                 print(f"Iteration {i}/{self.n_iterations}...")
            self._sample_parameters()
            self._store_sample(iteration=i)
            
        print("MCMC sampling complete.")
        # Filter out empty sample lists if a parameter was never stored
        return {k: v for k, v in self.samples.items() if v}

    def get_posterior_summary(self, parameter_name: str) -> dict | None:
        """
        Calculates and returns posterior summary statistics for a specified parameter.

        Summaries include mean, standard deviation, and 2.5% and 97.5% percentiles
        (for a 95% credible interval).

        Args:
            parameter_name (str): The name of the parameter for which to get the summary
                                  (e.g., 'beta', 'g', 'variance_components').

        Returns:
            dict | None: A dictionary containing the summary statistics.
                         For 'variance_components', it's a nested dictionary where keys
                         are component names (e.g., 'genetic', 'residual') and values
                         are dicts of summaries. For 'beta' and 'g', it's a dict
                         where values are NumPy arrays (mean, std, ci_low, ci_high),
                         with statistics calculated column-wise if samples are multi-dimensional.
                         Returns None if no samples are found for the parameter.
        
        Raises:
            ValueError: If 'variance_components' samples are not in the expected format.
        """
        if parameter_name not in self.samples or not self.samples[parameter_name]:
            print(f"Warning: No samples found for parameter '{parameter_name}'.")
            return None

        raw_samples = self.samples[parameter_name]

        if parameter_name == 'variance_components':
            # Assumes raw_samples is a list of dicts, e.g., [{'genetic': v1, 'residual': v2}, ...]
            if not isinstance(raw_samples[0], dict):
                raise ValueError("Samples for 'variance_components' are not in the expected format (list of dicts).")
            
            summary = {}
            # Extract samples for each key in the variance components dictionary
            vc_keys = raw_samples[0].keys()
            for key in vc_keys:
                component_samples = np.array([s[key] for s in raw_samples])
                summary[key] = {
                    'mean': np.mean(component_samples),
                    'std': np.std(component_samples),
                    'ci_low': np.percentile(component_samples, 2.5),
                    'ci_high': np.percentile(component_samples, 97.5)
                }
            return summary
        else: # For 'beta', 'g' which are lists of arrays (or single arrays if 1D)
            samples_array = np.array(raw_samples)
            if samples_array.ndim == 1: # Handle case where parameter is 1D (e.g. single fixed effect, or error)
                samples_array = samples_array.reshape(-1,1)


            return {
                'mean': np.mean(samples_array, axis=0),
                'std': np.std(samples_array, axis=0),
                'ci_low': np.percentile(samples_array, 2.5, axis=0),
                'ci_high': np.percentile(samples_array, 97.5, axis=0)
            }


class GBLUPSampler(BaseMCMCSampler):
    """
    MCMC sampler for Genomic Best Linear Unbiased Prediction (GBLUP).

    This sampler implements a Gibbs sampling algorithm to estimate fixed effects (beta),
    genomic breeding values (g), and variance components (genetic and residual variance)
    for a GBLUP model.

    It requires phenotype data, genotype data (including a GRM), and a model definition.

    Attributes:
        genotype_data (GenotypeData): The genotype data, including the GRM.
        GRM (np.ndarray): The Genomic Relationship Matrix.
        X (np.ndarray): Design matrix for fixed effects.
        Z_g (np.ndarray): Design matrix for random genetic effects (g).
        y (np.ndarray): Phenotype vector (potentially subsetted if NaNs were present).
        prior_df (float): Prior degrees of freedom for variance components.
        prior_scale_g (float): Prior scale for genetic variance.
        prior_scale_e (float): Prior scale for residual variance.
    """
    def __init__(self, model_definition: 'ModelDefinition', 
                 phenotype_data: 'PhenotypeData', 
                 genotype_data: 'GenotypeData', 
                 n_iterations: int, burn_in: int):
        """
        Initializes the GBLUPSampler.

        Args:
            model_definition (ModelDefinition): The model specification.
            phenotype_data (PhenotypeData): Phenotype data.
            genotype_data (GenotypeData): Genotype data, expected to contain or be
                                          able to calculate a GRM.
            n_iterations (int): Total number of MCMC iterations.
            burn_in (int): Number of burn-in iterations.

        Raises:
            TypeError: If `genotype_data` is not a valid GenotypeData instance.
            ValueError: If GRM cannot be obtained from `genotype_data`.
        """
        super().__init__(model_definition, phenotype_data, n_iterations, burn_in)
        
        if not hasattr(genotype_data, 'calculate_grm'): # Basic check for GenotypeData
            raise TypeError("genotype_data does not appear to be a valid GenotypeData instance.")
        
        self.genotype_data = genotype_data
        if self.genotype_data.grm is None:
            print("GRM not found in genotype_data, calculating now...")
            self.genotype_data.calculate_grm() 
        self.GRM = self.genotype_data.grm
        if self.GRM is None: # Still None after calculation attempt
             raise ValueError("GRM could not be calculated or retrieved from genotype_data.")

        # Basic alignment check placeholder - actual alignment is complex.
        # Assumes phenotype_data.individual_ids and genotype_data.individual_ids exist for a proper check.
        if self.phenotype_data.individual_ids and self.genotype_data.individual_ids:
            pheno_ids_set = set(self.phenotype_data.individual_ids)
            geno_ids_set = set(self.genotype_data.individual_ids)
            if not pheno_ids_set.issubset(geno_ids_set):
                # This is a warning; depending on Z matrix construction, it might be handled.
                print("Warning: Not all phenotyped individuals are present in the genotype data based on IDs.")
        else:
            print("Warning: Individual IDs not available in phenotype or genotype data for full alignment check. Assuming data is ordered correctly.")

        # These will be constructed in _initialize_parameters
        self.X = None
        self.Z_g = None 
        self.y = None

    def _initialize_parameters(self):
        super()._initialize_parameters() # Call base class method if it does anything useful

        # 1. Determine dependent variable (y)
        # Assuming the first dependent variable in the model definition is the one to use.
        if not self.model_definition.dependent_vars:
            raise ValueError("No dependent variables specified in model_definition.")
        y_col_name = self.model_definition.dependent_vars[0] 
        self.y = self.phenotype_data.get_y_vector(y_col_name)
        
        # Handle NaNs in y: For GBLUP, individuals with NaN phenotypes are typically removed.
        # This also requires removing corresponding rows from X and Z.
        nan_y_indices = np.isnan(self.y)
        if np.any(nan_y_indices):
            print(f"Warning: {np.sum(nan_y_indices)} NaN values found in phenotype vector y. These records will be removed for GBLUP analysis.")
            self.y = self.y[~nan_y_indices]
            # Store original indices if needed for mapping results back, or manage ID lists.
            # For now, assume phenotype_data.data is not modified, so X and Z will be built on full data then subsetted.
        
        # 2. Construct X (fixed effects incidence matrix)
        # model_definition.fixed_effects should contain "intercept" if it was in the equation.
        self.X = self.phenotype_data.get_design_matrix_X(self.model_definition.fixed_effects)
        if np.any(nan_y_indices): # If y had NaNs, X needs to be subsetted
            self.X = self.X[~nan_y_indices, :]

        if self.X.shape[0] != len(self.y):
            raise ValueError(f"Mismatch in number of observations for y ({len(self.y)}) and X ({self.X.shape[0]}) after NaN handling.")

        n_fe = self.X.shape[1]
        self.beta = np.zeros(n_fe) # Initialize fixed effects (beta)
        
        # 3. Construct Z_g (random genetic effects incidence matrix)
        # For GBLUP, there's typically one random effect term representing 'g' or 'animal'.
        # We need to pass all genotyped individual IDs (in GRM order) to get_design_matrix_Z.
        if not self.genotype_data.individual_ids:
             raise ValueError("Genotype data must have individual_ids for Z_g matrix construction.")
        
        # Assuming the random effect term for GBLUP is implicitly 'g' or the main effect captured by GRM
        # This part might need refinement based on how random effects are specified in ModelDefinition for GBLUP
        gblup_random_term = ["g"] # Placeholder term, ModelDefinition might need to specify this
        self.Z_g = self.phenotype_data.get_design_matrix_Z(gblup_random_term, self.genotype_data.individual_ids)
        if np.any(nan_y_indices): # If y had NaNs, Z_g needs to be subsetted
            self.Z_g = self.Z_g[~nan_y_indices, :]
        
        if self.Z_g.shape[0] != len(self.y):
            raise ValueError(f"Mismatch in number of observations for y ({len(self.y)}) and Z_g ({self.Z_g.shape[0]}) after NaN handling.")
        if self.Z_g.shape[1] != self.GRM.shape[0]:
            raise ValueError(f"Z_g columns ({self.Z_g.shape[1]}) do not match GRM dimension ({self.GRM.shape[0]}).")

        n_geno = self.GRM.shape[0]
        self.g = np.zeros(n_geno) # Initialize genomic effects (g)

        # 4. Initialize variance components and priors
        self.prior_df = 5 
        self.prior_scale_g = 0.5 
        self.prior_scale_e = 0.5 
        sigma2_g = invgamma.rvs(a=self.prior_df/2, scale=(self.prior_df * self.prior_scale_g)/2) # Sample from prior
        sigma2_e = invgamma.rvs(a=self.prior_df/2, scale=(self.prior_df * self.prior_scale_e)/2) # Sample from prior
        self.variance_components = {'genetic': sigma2_g if sigma2_g > 0 else 1.0, 
                                    'residual': sigma2_e if sigma2_e > 0 else 1.0}
        
        print(f"GBLUPSampler: Initialization complete. y shape: {self.y.shape}, X shape: {self.X.shape}, Z_g shape: {self.Z_g.shape}, GRM shape: {self.GRM.shape}")
        print(f"Initial variance components: genetic={self.variance_components['genetic']:.4f}, residual={self.variance_components['residual']:.4f}")


    def _sample_parameters(self):
        # Ensure y, X, Z_g, GRM are available and correctly shaped from initialization
        y, X, Z_g, GRM = self.y, self.X, self.Z_g, self.GRM
        sigma2_g = self.variance_components['genetic']
        sigma2_e = self.variance_components['residual']

        # Inverse of GRM (with ridge for stability)
        GRM_inv = np.linalg.inv(GRM + np.eye(GRM.shape[0]) * 1e-8) # Small ridge

        # 1. Sample β (fixed effects)
        y_adj_beta = y - Z_g @ self.g
        XTX = X.T @ X
        # Add ridge for stability, especially if X is not full rank or has collinearity
        XTX_inv = np.linalg.inv(XTX + np.eye(XTX.shape[0]) * 1e-8) 
        beta_mean = XTX_inv @ X.T @ y_adj_beta
        beta_cov = XTX_inv * sigma2_e
        try:
            self.beta = multivariate_normal.rvs(mean=beta_mean, cov=beta_cov, allow_singular=True)
        except np.linalg.LinAlgError as e:
            print(f"Warning: Singular covariance matrix for beta sampling: {e}. Using pseudo-inverse.")
            beta_cov_pinv = np.linalg.pinv(beta_cov)
            # Re-sample mean or use pseudo-inverse for covariance if sampler allows
            # For simplicity, drawing from mean with diagonal variance if cov is problematic
            # This is a fallback and indicates issues with model/data (e.g. collinearity)
            try:
                self.beta = multivariate_normal.rvs(mean=beta_mean, cov=(np.diag(np.diag(beta_cov)) + np.eye(beta_cov.shape[0])*1e-9), allow_singular=True)
            except: # Final fallback
                 self.beta = beta_mean 
                 print("Warning: Beta sampling completely failed, using mean.")


        # 2. Sample g (genomic values)
        y_adj_g = y - X @ self.beta
        
        # Coefficient matrix for g: (Z_g.T @ Z_g / sigma2_e + GRM_inv / sigma2_g)
        # Equivalent to (Z_g.T @ Z_g * (1/sigma2_e) + GRM_inv * (1/sigma2_g))
        # Or (Z_g.T @ Z_g + GRM_inv * (sigma2_e / sigma2_g)) / sigma2_e
        # Let lambda_val = sigma2_e / sigma2_g
        # Cgg_scaled = Z_g.T @ Z_g + GRM_inv * lambda_val
        # Cgg_inv_scaled = np.linalg.inv(Cgg_scaled + np.eye(Cgg_scaled.shape[0]) * 1e-8)
        # g_mean = Cgg_inv_scaled @ Z_g.T @ y_adj_g
        # g_cov = Cgg_inv_scaled * sigma2_e
        # self.g = multivariate_normal.rvs(mean=g_mean, cov=g_cov, allow_singular=True)

        # Using the formulation from the prompt:
        # Cgg = Z_g.T @ Z_g / sigma2_e + GRM_inv / sigma2_g
        Cgg = (Z_g.T @ Z_g / sigma2_e) + (GRM_inv / sigma2_g)
        Cgg_inv = np.linalg.inv(Cgg + np.eye(Cgg.shape[0]) * 1e-8) # Add ridge for stability
        g_mean = Cgg_inv @ (Z_g.T @ y_adj_g / sigma2_e)
        try:
            self.g = multivariate_normal.rvs(mean=g_mean, cov=Cgg_inv, allow_singular=True)
        except np.linalg.LinAlgError as e:
            print(f"Warning: Singular covariance matrix for g sampling: {e}. Using pseudo-inverse.")
            # Fallback similar to beta
            try:
                self.g = multivariate_normal.rvs(mean=g_mean, cov=(np.diag(np.diag(Cgg_inv)) + np.eye(Cgg_inv.shape[0])*1e-9), allow_singular=True)
            except:
                self.g = g_mean
                print("Warning: g sampling completely failed, using mean.")


        # 3. Sample σ²g (genetic variance)
        # scale_g = (self.g.T @ GRM_inv @ self.g + self.prior_df * self.prior_scale_g)
        # df_g = GRM.shape[0] + self.prior_df
        # self.variance_components['genetic'] = invgamma.rvs(a=df_g/2, scale=scale_g/2)
        
        # Corrected InvGamma parameterization for scale (often rate parameter b = scale/2 for invgamma(a,b))
        # If invgamma uses shape (a) and scale (β) parameters directly:
        # Posterior shape a' = a_prior + n/2
        # Posterior scale β' = β_prior + S/2 
        # where S = g' G_inv g. Here, a_prior = self.prior_df/2, β_prior = (self.prior_df * self.prior_scale_g)/2
        
        post_df_g = self.prior_df + GRM.shape[0]
        post_scale_g_sum_sq = self.g.T @ GRM_inv @ self.g
        post_scale_g = self.prior_df * self.prior_scale_g + post_scale_g_sum_sq
        
        sampled_sigma2_g = invgamma.rvs(a=post_df_g/2, scale=post_scale_g/2)
        self.variance_components['genetic'] = max(sampled_sigma2_g, 1e-9) # Ensure positive variance


        # 4. Sample σ²e (residual variance)
        e = y - X @ self.beta - Z_g @ self.g
        # scale_e = (e.T @ e + self.prior_df * self.prior_scale_e)
        # df_e = len(y) + self.prior_df
        # self.variance_components['residual'] = invgamma.rvs(a=df_e/2, scale=scale_e/2)
        
        post_df_e = self.prior_df + len(y)
        post_scale_e_sum_sq = e.T @ e
        post_scale_e = self.prior_df * self.prior_scale_e + post_scale_e_sum_sq

        sampled_sigma2_e = invgamma.rvs(a=post_df_e/2, scale=post_scale_e/2)
        self.variance_components['residual'] = max(sampled_sigma2_e, 1e-9) # Ensure positive variance

    def _store_sample(self, iteration: int):
        # Overrides BaseMCMCSampler._store_sample to ensure correct data is stored
        if iteration > self.burn_in:
            self.samples['beta'].append(np.copy(self.beta))
            self.samples['g'].append(np.copy(self.g)) # Store copy of g
            self.samples['variance_components'].append(self.variance_components.copy())

    def get_ebv_summary(self) -> pd.DataFrame | None:
        """
        Generates a pandas DataFrame with Estimated Breeding Value (EBV) summaries.

        The EBVs are derived from the posterior samples of the genomic values 'g'.
        The summary includes mean, standard deviation, and a 95% credible interval
        for the EBV of each individual in the genotype data.

        Returns:
            pd.DataFrame | None: A DataFrame with columns 'individual_id', 'ebv_mean',
                                  'ebv_std', 'ebv_ci_low', 'ebv_ci_high'.
                                  Returns None if genotype data with IDs or 'g' samples
                                  are unavailable.
        """
        if not hasattr(self, 'genotype_data') or self.genotype_data.individual_ids is None:
            print("Warning: Genotype data with individual IDs is required for EBV summary.")
            return None

        g_summary = self.get_posterior_summary('g')
        if g_summary is None:
            print("Warning: Posterior summary for 'g' (genomic values) is not available.")
            return None

        individual_ids = self.genotype_data.individual_ids
        
        # Ensure g_summary outputs match the number of individuals
        num_ids = len(individual_ids)
        if not all(len(val) == num_ids for val_name, val in g_summary.items() if isinstance(val, np.ndarray)):
             # This can happen if g was stored as a list of 1D arrays and then get_posterior_summary handled it.
             # Or if g has fewer elements than individual_ids (e.g. subset of individuals in GRM that don't match original list)
             # This indicates a potential mismatch that should be investigated.
             # For now, we'll try to proceed if lengths seem compatible for DataFrame construction.
            g_mean_len = len(g_summary['mean'])
            if g_mean_len != num_ids:
                print(f"Warning: Length of mean g ({g_mean_len}) does not match number of individual IDs ({num_ids}). EBV summary might be incorrect or incomplete.")
                # Truncate or pad IDs/g_summary if necessary, or raise error
                # For now, we'll use the minimum length to avoid crashing
                min_len = min(num_ids, g_mean_len)
                individual_ids = individual_ids[:min_len]
                for key in g_summary:
                    g_summary[key] = g_summary[key][:min_len]


        ebv_df = pd.DataFrame({
            'individual_id': individual_ids,
            'ebv_mean': g_summary['mean'],
            'ebv_std': g_summary['std'],
            'ebv_ci_low': g_summary['ci_low'],
            'ebv_ci_high': g_summary['ci_high']
        })
        return ebv_df


def run_mcmc(model_definition: 'ModelDefinition', phenotype_data: 'PhenotypeData', 
             genotype_data: 'GenotypeData' = None, # Added genotype_data
             n_iterations: int = 1000, burn_in: int = 100, method: str = "default"):
    """
    Top-level function to configure and run an MCMC sampler.

    This function acts as a factory for different MCMC sampler types based on the
    `method` argument. It initializes the appropriate sampler and then calls its
    `run()` method. After the run completes, it returns the sampler instance itself,
    which contains the samples and can be used to access summary methods.

    Args:
        model_definition (ModelDefinition): The model specification.
        phenotype_data (PhenotypeData): The phenotype data.
        genotype_data (GenotypeData, optional): The genotype data. Required if
                                                `method` is "GBLUP". Defaults to None.
        n_iterations (int, optional): Total number of MCMC iterations. Defaults to 1000.
        burn_in (int, optional): Number of burn-in iterations. Defaults to 100.
        method (str, optional): Specifies the MCMC sampling method to use.
                                Currently supports "GBLUP" and "default" (BaseMCMCSampler).
                                Defaults to "default".

    Returns:
        BaseMCMCSampler: The sampler instance after the MCMC run has completed.
                         This instance contains the samples and provides methods for
                         their summarization (e.g., `get_posterior_summary`, `get_ebv_summary`).
    
    Raises:
        ValueError: If `genotype_data` is not provided when `method` is "GBLUP".
        NotImplementedError: If an unsupported `method` is specified.
    """
    if method.upper() == "GBLUP":
        if genotype_data is None:
            raise ValueError("GenotypeData must be provided for GBLUP method.")
        sampler = GBLUPSampler(model_definition, phenotype_data, genotype_data, 
                               n_iterations, burn_in)
    elif method == "default":
        # BaseMCMCSampler might not be runnable if its _sample_parameters is not implemented
        print("Warning: Using BaseMCMCSampler with placeholder sampling logic.")
        sampler = BaseMCMCSampler(model_definition, phenotype_data, n_iterations, burn_in)
    else:
        raise NotImplementedError(f"MCMC method '{method}' is not yet implemented.")
        
    sampler.run() # Run the sampler
    return sampler # Return the sampler instance itself
