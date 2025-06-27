from typing import List, Any, Optional, Union, Dict
import numpy as np

# Forward declaration or direct import
from .variance_covariance import VarianceCovariance

class GenotypesData:
    """
    Stores and manages genotype data and related parameters for a specific genomic category.
    Translated from the Julia JWAS.jl Genotypes struct.
    """
    def __init__(self,
                 name: str,
                 trait_names: Optional[List[str]] = None,
                 obs_ids: Optional[List[str]] = None,
                 marker_ids: Optional[List[str]] = None,
                 genotype_matrix: Optional[np.ndarray] = None, # n_obs x n_markers
                 allele_freqs: Optional[np.ndarray] = None, # n_markers
                 is_centered: bool = False,
                 is_grm: bool = False,
                 method: Optional[str] = None, # e.g., "BayesC", "GBLUP"
                 pi_value: Optional[Union[float, Dict[Any, float]]] = None, # Prior for marker inclusion
                 estimate_pi: bool = False
                 ):
        """
        Initializes a GenotypesData instance.

        Args:
            name: Name for this genotype category (e.g., "snps", "chip1").
            trait_names: Names of traits this genotype data applies to.
            obs_ids: List of individual IDs corresponding to rows in genotype_matrix.
            marker_ids: List of marker IDs corresponding to columns in genotype_matrix.
            genotype_matrix: The actual genotype data. Could be raw genotypes or centered.
            allele_freqs: Allele frequencies for each marker.
            is_centered: Boolean indicating if the genotype_matrix is already centered.
            is_grm: Boolean indicating if genotype_matrix is actually a Genomic Relationship Matrix.
            method: Bayesian method to be applied (e.g., "BayesC", "GBLUP").
            pi_value: Initial value for pi (marker inclusion probability).
            estimate_pi: Boolean indicating if pi should be estimated.
        """
        self.name: str = name
        self.trait_names: List[str] = trait_names if trait_names is not None else []

        self.obs_ids: List[str] = obs_ids if obs_ids is not None else []
        self.marker_ids: List[str] = marker_ids if marker_ids is not None else []

        self.n_obs: int = len(self.obs_ids)
        self.n_markers: int = len(self.marker_ids)

        self.allele_freqs: Optional[np.ndarray] = allele_freqs
        self.sum_2pq: Optional[float] = None # Can be calculated from allele_freqs
        if self.allele_freqs is not None:
            p = self.allele_freqs
            self.sum_2pq = float(np.sum(2 * p * (1 - p)))

        self.is_centered: bool = is_centered
        self.genotypes: Optional[np.ndarray] = genotype_matrix

        self.n_loci_model: int = self.n_markers # Number of markers/loci included in the model (can change)
        self.n_traits_model: int = len(self.trait_names) # Number of traits this genotype set is used for

        # VarianceCovariance objects
        self.genetic_variance: Optional[VarianceCovariance] = None # Overall genetic variance explained
        self.marker_effect_variance: Optional[VarianceCovariance] = None # Variance of marker effects (sigma_g^2)

        self.method: Optional[str] = method
        self.estimate_pi: bool = estimate_pi

        # Placeholders for MCMC-related arrays and matrices (Bayesian Alphabet)
        # These would be populated during MCMC initialization
        self.m_array: Any = None
        self.m_rinv_array: Any = None
        self.mp_rinvm: Any = None
        # ... other arrays like m_phi_phi_array, M_array, MRinvArray, MpRinvM, D_eigen_gblup, gamma_array_bayesl

        # MCMC samples - these are typically lists of arrays, one per trait if multi-trait
        self.alpha_samples: List[np.ndarray] = [] # Marker effects (beta * delta)
        self.beta_samples: List[np.ndarray] = []  # Actual marker effect sizes (continuous part)
        self.delta_samples: List[np.ndarray] = [] # Inclusion indicators (0 or 1)

        # Pi (marker inclusion probability) - can be scalar or more complex for multi-trait
        self.pi_value: Optional[Union[float, Dict[Any, float]]] = pi_value
        if self.pi_value is None and self.method not in ["GBLUP", "RR-BLUP"]: # Default for Bayesian alphabet methods
             self.pi_value = 0.5 # A common default if not specified

        # Posterior means and variances from MCMC
        self.mean_alpha: List[np.ndarray] = []
        self.mean_alpha_sq: List[np.ndarray] = [] # For variance calculation: E[x^2]
        self.mean_delta: List[np.ndarray] = []

        self.mean_pi: Optional[Union[float, Dict[Any, float]]] = None
        self.mean_pi_sq: Optional[Union[float, Dict[Any, float]]] = None # E[pi^2]

        self.mean_marker_variance: Optional[Union[float, np.ndarray]] = None # Posterior mean of Mi.G.val
        self.mean_marker_variance_sq: Optional[Union[float, np.ndarray]] = None

        self.mean_scale_marker_variance: Optional[Union[float, np.ndarray]] = None # Posterior mean of Mi.G.scale
        self.mean_scale_marker_variance_sq: Optional[Union[float, np.ndarray]] = None

        self.output_genotypes: bool = False # Flag to control if genotypes are written to output
        self.is_grm: bool = is_grm # Is this a Genomic Relationship Matrix?

    def __repr__(self) -> str:
        return (f"GenotypesData(name='{self.name}', method='{self.method}', "
                f"n_obs={self.n_obs}, n_markers={self.n_markers}, "
                f"n_traits_model={self.n_traits_model}, is_grm={self.is_grm})")

    def initialize_mcmc_storage(self, n_traits: int, n_markers_in_model: int):
        """Initializes storage for MCMC samples based on model configuration."""
        self.n_traits_model = n_traits
        self.n_loci_model = n_markers_in_model

        self.alpha_samples = [np.zeros(n_markers_in_model) for _ in range(n_traits)]
        self.beta_samples = [np.zeros(n_markers_in_model) for _ in range(n_traits)]
        self.delta_samples = [np.ones(n_markers_in_model) for _ in range(n_traits)] # Often start as all included

        self.mean_alpha = [np.zeros(n_markers_in_model) for _ in range(n_traits)]
        self.mean_alpha_sq = [np.zeros(n_markers_in_model) for _ in range(n_traits)]
        self.mean_delta = [np.zeros(n_markers_in_model) for _ in range(n_traits)]

        if isinstance(self.pi_value, float) or self.pi_value is None:
            self.mean_pi = 0.0
            self.mean_pi_sq = 0.0
        elif isinstance(self.pi_value, dict):
            self.mean_pi = {k: 0.0 for k in self.pi_value}
            self.mean_pi_sq = {k: 0.0 for k in self.pi_value}

        # Initialize variance storage based on expected type (scalar or array for multi-trait)
        if n_traits == 1:
            self.mean_marker_variance = 0.0
            self.mean_marker_variance_sq = 0.0
            if self.marker_effect_variance and isinstance(self.marker_effect_variance.scale, float):
                 self.mean_scale_marker_variance = 0.0
                 self.mean_scale_marker_variance_sq = 0.0
        else: # multi-trait
            self.mean_marker_variance = np.zeros((n_traits, n_traits))
            self.mean_marker_variance_sq = np.zeros((n_traits, n_traits))
            if self.marker_effect_variance and isinstance(self.marker_effect_variance.scale, np.ndarray):
                 self.mean_scale_marker_variance = np.zeros_like(self.marker_effect_variance.scale)
                 self.mean_scale_marker_variance_sq = np.zeros_like(self.marker_effect_variance.scale)


if __name__ == '__main__':
    # Example Usage
    geno_data = GenotypesData(name="chip_data",
                              obs_ids=[f"id_{i}" for i in range(100)],
                              marker_ids=[f"snp_{j}" for j in range(1000)],
                              method="BayesC",
                              estimate_pi=True)
    print(geno_data)

    # Simulate setting variance components
    geno_data.genetic_variance = VarianceCovariance(value=0.4, df=4.0)
    geno_data.marker_effect_variance = VarianceCovariance(value=0.001, df=4.0, estimate_variance=True)
    print(f"Genetic Variance: {geno_data.genetic_variance}")
    print(f"Marker Effect Variance: {geno_data.marker_effect_variance}")

    # Initialize MCMC storage for a single trait model
    geno_data.initialize_mcmc_storage(n_traits=1, n_markers_in_model=1000)
    print(f"Length of alpha_samples[0]: {len(geno_data.alpha_samples[0])}")
    print(f"Mean pi: {geno_data.mean_pi}")

    # Multi-trait example
    geno_data_mt = GenotypesData(name="gwas_panel", method="MTBayesC", trait_names=["trait1", "trait2"])
    geno_data_mt.marker_effect_variance = VarianceCovariance(value=np.diag([0.01, 0.01]), df=5.0)
    geno_data_mt.initialize_mcmc_storage(n_traits=2, n_markers_in_model=500)
    print(geno_data_mt)
    print(f"Mean marker variance shape: {geno_data_mt.mean_marker_variance.shape}")
