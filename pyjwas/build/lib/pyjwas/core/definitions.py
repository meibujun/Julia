from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Callable
import numpy as np
# Forward declaration for pandas DataFrame if needed, though not directly stored in these types yet
# import pandas as pd # Not strictly needed here yet, but will be for functions using these types

# It's good practice to define a type alias for float precision early on
# The Julia code uses Union{Array{Float64,1},Array{Float32,1}}
# We can decide on a default precision or make it configurable later.
# For now, let's default to np.float64 for wider compatibility,
# but acknowledge that np.float32 might be an optimization.
DefaultFloat = np.float64
DefaultInt = np.int64

@dataclass
class ModelTerm:
    """
    Represents a single term in a model equation (e.g., y1:A, y2:A*B).
    Corresponds to Julia struct ModelTerm.
    """
    i_model: int  # 1st (1) or 2nd (2) model_equation
    i_trait: str  # trait 1 ("y1") or trait 2 ("y2") (trait name)
    trm_str: str  # Full term string, e.g., "y1:A" or "y1:A*B"
    n_factors: int # Number of factors in the term, e.g., 1 for "A", 2 for "A*B"
    factors: List[str] # List of factor names as symbols/strings, e.g., [:A, :B]

    # Data associated with the term
    data: List[str] = field(default_factory=list) # String representation of levels/values for each obs
                                                 # e.g., covariate^2: ["A x B", "A X B", ...]
                                                 # factor^2: ["A1 x B1", "A2 X B2", ...]
                                                 # factor*covariate: ["A1 x B","A2 x B", ...]
    val: np.ndarray = field(default_factory=lambda: np.array([], dtype=DefaultFloat)) # Numerical values for each observation

    # Output attributes
    n_levels: int = 0 # Number of levels for this term (1 for covariate)
    names: List[Any] = field(default_factory=list) # Names of levels (e.g., "A1", "A2" for factor; "A" for covariate)

    start_pos: int = 0  # Start position for this term in the overall incidence matrix
    X: Optional[Any] = None  # Incidence matrix (likely a sparse matrix from scipy.sparse)

    random_type: str = "fixed" # "fixed", "I" (iid random), "A" (pedigree), "V" (general variance structure), "genotypes"

    # Julia constructor: ModelTerm(trmStr,m,traitname)
    # Python equivalent will be handled by direct instantiation or a helper factory if needed.

@dataclass
class Variance:
    """
    Structure to hold information about a variance component.
    Used for marker effect variances, residual variances, etc.
    Corresponds to Julia struct Variance.
    """
    val: Union[DefaultFloat, np.ndarray, bool] = False # Value of the variance (scalar or matrix for multi-trait)
    df: Union[DefaultFloat, bool] = False             # Degrees of freedom
    scale: Union[DefaultFloat, np.ndarray, bool] = False # Scale parameter

    estimate_variance: bool = True  # Estimate variance at each MCMC iteration
    estimate_scale: bool = False    # Estimate scale at each MCMC iteration
    constraint: bool = False        # For multi-trait, constraint=true means covariance is zero

@dataclass
class ResVar:
    """
    Specialized structure for residual covariance, particularly for missing data.
    Corresponds to Julia struct ResVar.
    """
    R0: Optional[np.ndarray] = None # Base residual covariance matrix (nob*nModel x nob*nModel or ntraits x ntraits)
    # Julia uses BitArray{1} for keys, List[bool] or tuple of bools might be Pythonic
    # For simplicity, using a tuple of bools as dict key.
    RiDict: Dict[tuple[bool, ...], np.ndarray] = field(default_factory=dict)

@dataclass
class RandomEffect:
    """
    Represents a general random effect in the model.
    Corresponds to Julia struct RandomEffect.
    """
    term_array: List[str] # Model terms constituting this random effect (e.g. ["y1:animal", "y2:animal"])
    Gi: Variance          # Covariance matrix for this effect (Variance object)
    GiOld: Optional[Variance] = None # Used for MCMC updates (lambda version of MME)
    GiNew: Optional[Variance] = None # Used for MCMC updates (lambda version of MME)

    Vinv: Optional[Any] = None # Inverse variance or relationship matrix (e.g., A-inverse, G-inverse), sparse matrix
    names: List[Any] = field(default_factory=list) # IDs associated with levels of this random effect
    random_type: str = "I" # Type of random effect (e.g., "I" for iid, "A" for pedigree, "V" for custom Vinv)

@dataclass
class Genotypes:
    """
    Manages genotype data for a specific category of markers.
    Corresponds to Julia struct Genotypes.
    """
    name: str = "" # Name for this genotype category, e.g., "geno1"
    trait_names: List[str] = field(default_factory=list) # Names of corresponding traits, e.g., ["y1", "y2"]

    obs_id: List[str] = field(default_factory=list) # Row IDs for genotyped (and phenotyped) individuals
    marker_id: List[str] = field(default_factory=list) # Marker IDs

    n_obs: int = 0
    n_markers: int = 0

    allele_freq: Optional[np.ndarray] = None # Allele frequencies
    sum_2pq: DefaultFloat = 0.0 # Sum of 2*p*q over markers
    centered: bool = False

    genotypes: Optional[np.ndarray] = None # Genotype matrix (n_obs x n_markers)

    n_loci: int = 0 # Number of markers included in the model (could be different from n_markers if selection occurs)
    n_traits: int = 0 # Number of traits this genotype set applies to

    genetic_variance: Variance = field(default_factory=lambda: Variance(val=False, df=4.0, scale=False, estimate_variance=True))
    G: Variance = field(default_factory=lambda: Variance(val=False, df=4.0, scale=False, estimate_variance=True)) # Marker effect variance

    method: str = "BayesC" # Prior for marker effects (e.g., "BayesC", "GBLUP", "BayesA", "BayesB", "BayesLasso")
    estimate_pi: bool = True # Estimate pi (proportion of non-zero effect markers)

    # Pre-calculated matrices or intermediate results for specific algorithms
    # These are often Optional and initialized to None or empty arrays
    m_array: Optional[Any] = None
    m_rinv_array: Optional[Any] = None
    mp_rinvm: Optional[Any] = None
    m_phi_phi_array: Optional[Any] = None # For RRM
    M_array: Optional[Any] = None # RHS approach
    M_rinv_array: Optional[Any] = None # RHS approach
    Mp_rinv_M: Optional[Any] = None # RHS approach
    D: Optional[np.ndarray] = None # Eigenvalues for GBLUP
    gamma_array: Optional[np.ndarray] = None # For Bayesian LASSO

    # MCMC samples
    alpha: Optional[np.ndarray] = None # Marker effects
    beta: Optional[np.ndarray] = None  # Other parameters if needed
    delta: Optional[np.ndarray] = None # Inclusion indicators (0/1)
    pi_val: Optional[DefaultFloat] = 0.5 # Current value of pi

    # Means of MCMC samples
    mean_alpha: Optional[np.ndarray] = None
    mean_alpha2: Optional[np.ndarray] = None
    mean_delta: Optional[np.ndarray] = None
    mean_pi: Optional[DefaultFloat] = None
    mean_pi2: Optional[DefaultFloat] = None
    mean_vara: Optional[Any] = None # Mean of genetic variance
    mean_vara2: Optional[Any] = None
    mean_scale_vara: Optional[Any] = None
    mean_scale_vara2: Optional[Any] = None

    output_genotypes: bool = False # Whether to output these genotypes
    is_grm: bool = False # True if a pre-computed GRM was provided instead of raw genotypes

@dataclass
class MCMCInfo:
    """
    Holds parameters and settings for an MCMC run.
    Corresponds to Julia struct MCMCinfo.
    """
    heterogeneous_residuals: bool = False
    chain_length: int = 1000
    burnin: int = 100
    output_samples_frequency: int = 10
    printout_model_info: bool = True
    printout_frequency: int = 100
    single_step_analysis: bool = False
    fitting_J_vector: bool = True # for single-step
    missing_phenotypes: bool = True # allow missing phenotypes in multi-trait
    update_priors_frequency: int = 0 # 0 means update based on all previous samples, 1 means current sample, >1 means block
    output_ebv: bool = True
    output_heritability: bool = True
    prediction_equation: Union[str, bool] = False
    seed: Union[int, bool] = False
    double_precision: bool = True # Default to True (np.float64)
    output_folder: str = "results"
    rrm: bool = False # Random Regression Model
    fast_blocks: Union[bool, int] = False # For marker processing in blocks

@dataclass
class MME:
    """
    Central class encapsulating Mixed Model Equations and related information.
    Corresponds to Julia struct MME.
    """
    n_models: int # Number of model equations (traits)
    model_vec: List[str] # List of model equation strings, e.g., ["y1 = A + B", "y2 = A + C"]
    model_terms: List[ModelTerm] # List of all ModelTerm objects
    model_term_dict: Dict[str, ModelTerm] # Dictionary mapping term string to ModelTerm object
    lhs_vec: List[str] # List of phenotype (left-hand side) names as strings (Julia uses Symbols)
    cov_vec: List[str] = field(default_factory=list) # List of variable names (strings) that are covariates

    # Mixed Model Equations components
    X: Optional[Any] = None # Full incidence matrix (sparse)
    y_sparse: Optional[Any] = None # Phenotype vector (sparse)
    obs_id: List[str] = field(default_factory=list) # IDs for phenotypes

    mme_lhs: Optional[Any] = None # Left-hand side of MME (X'RinvX + Ginv_block)
    mme_rhs: Optional[Any] = None # Right-hand side of MME (X'Rinvy)

    # Random effects - Pedigree
    ped_trm_vec: List[str] = field(default_factory=list) # Polygenic effect terms, e.g., ["1:Animal"]
    ped: Optional[Any] = None # Pedigree object (e.g., from a Pedigree library or custom class)
    # Gi, GiOld, GiNew for pedigree are often handled within a RandomEffect object in rnd_trm_vec
    # Or, if specific, could be:
    # ped_Gi: Optional[Variance] = None
    # ped_GiOld: Optional[Variance] = None
    # ped_GiNew: Optional[Variance] = None
    # scale_ped: Optional[Any] = None # Not clear from Julia, might be part of Variance
    # G0_mean: Optional[np.ndarray] = None
    # G0_mean2: Optional[np.ndarray] = None

    # Random effects - General
    rnd_trm_vec: List[RandomEffect] = field(default_factory=list)

    # Residual effects
    R: Variance = field(default_factory=lambda: Variance(val=1.0, df=4.0, scale=0.5, estimate_variance=True)) # Residual (co)variance
    missing_pattern: Optional[Any] = None # For imputation of missing residuals
    res_var: Optional[ResVar] = None      # For imputation of missing residuals
    R_old: Optional[Union[DefaultFloat, np.ndarray]] = None # Residual variance (single-trait) for MCMC

    mean_vare: Optional[Any] = None # Mean of residual variance samples
    mean_vare2: Optional[Any] = None

    inv_weights: Optional[np.ndarray] = None # For heterogeneous residuals (n_obs x 1)

    # Genotypes
    M: List[Genotypes] = field(default_factory=list) # List of Genotypes objects

    mme_pos: int = 0 # Temporary value to record term position during MME construction

    output_samples_vec: List[ModelTerm] = field(default_factory=list) # Which location parameters to save MCMC samples for

    output_id: Optional[List[str]] = None # IDs for which to output predictions
    output_genotypes_dict: Dict[str, Any] = field(default_factory=dict) # Store output genotypes if needed
    output_X_dict: Dict[str, Any] = field(default_factory=dict) # Store output incidence matrices if needed

    results: Dict[str, Any] = field(default_factory=dict) # To store analysis results (EBVs, variance components, etc.)
    mcmc_info: MCMCInfo = field(default_factory=MCMCInfo)

    # Solution vectors
    sol: Optional[np.ndarray] = None # Current solution vector for MME
    sol_mean: Optional[np.ndarray] = None # Mean of solution vector samples
    sol_mean2: Optional[np.ndarray] = None # Mean of squared solution vector samples

    # Structural Equation Model
    causal_structure: Union[bool, np.ndarray] = False

    # Nonlinear models / Neural Networks
    nonlinear_function: Union[bool, Callable] = False # User-provided function or string like "tanh"
    weights_NN: Optional[np.ndarray] = None
    sigma2_yobs: DefaultFloat = 1.0 # Variance of observed phenotype in NN models
    is_fully_connected: bool = True
    is_activation_fcn: bool = False # If nonlinear_function is a predefined activation like "tanh", "relu"
    latent_traits: Union[bool, List[str]] = False # ["z1", "z2"], for intermediate omics data
    y_obs_for_nn: Optional[np.ndarray] = None # For single observed trait when y_sparse is for latent traits
    y_obs_name_for_nn: Optional[str] = None
    sigma2_weights_NN: DefaultFloat = 1.0 # Variance of NN weights
    fixed_sigma2_NN: bool = False # If sigma2_yobs and sigma2_weights_NN are fixed by user
    incomplete_omics: bool = False # If intermediate omics data can be missing

    # Trait types (continuous, censored, categorical)
    traits_type: List[str] = field(default_factory=list) # List of strings, one per trait
    thresholds: Dict[int, List[DefaultFloat]] = field(default_factory=dict) # Thresholds for categorical/binary, e.g., {0: [-np.inf, 0, np.inf]} for 1st trait

    def __post_init__(self):
        # Initialize traits_type if not provided, based on n_models
        if not self.traits_type and self.n_models > 0:
            self.traits_type = ["continuous"] * self.n_models
        # Ensure MCMCInfo is always initialized if not passed
        if self.mcmc_info is None:
            self.mcmc_info = MCMCInfo()

if __name__ == '__main__':
    # Example Usage (illustrative)
    term = ModelTerm(i_model=1, i_trait="yield", trm_str="yield:A", n_factors=1, factors=["A"])
    print(term)

    var_comp = Variance(val=1.5, df=4.0, scale=1.0, estimate_variance=True)
    print(var_comp)

    mcmc_settings = MCMCInfo(chain_length=20000, burnin=5000)
    print(mcmc_settings)

    # Basic MME structure for 2 traits
    # Typically, MME objects would be built by a model_builder function
    mme_instance = MME(n_models=2,
                       model_vec=["y1=intercept+A", "y2=intercept+B"],
                       model_terms=[], # Would be populated by builder
                       model_term_dict={}, # Would be populated by builder
                       lhs_vec=["y1", "y2"],
                       mcmc_info=mcmc_settings)
    print(mme_instance)
    print(f"Trait types for MME: {mme_instance.traits_type}")

    geno_data = Genotypes(name="snp_chip1", n_markers=1000, method="BayesC")
    print(geno_data)

    # Check DefaultFloat
    arr = np.array([1,2,3], dtype=DefaultFloat)
    print(f"Array with DefaultFloat: {arr.dtype}")

    # Check that traits_type is initialized correctly
    mme_single_trait = MME(n_models=1, model_vec=["y=mu+X"], model_terms=[], model_term_dict={}, lhs_vec=["y"])
    print(f"Single trait MME trait types: {mme_single_trait.traits_type}")
