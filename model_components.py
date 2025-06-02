import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
# Assuming Pedigree class is available from pedigree_module
# from pedigree_module import Pedigree # This would be the actual import

# Forward declaration for MME_py to resolve circular dependency if MME_py uses these components
class MME_py:
    pass

# --- Component Classes ---
class VarianceComponent:
    """
    Represents a variance component with its value, prior degrees of freedom,
    prior scale, and estimation flags.
    Analogous to Julia's `Variance` struct.
    """
    def __init__(self,
                 value: Union[float, np.ndarray, None] = None, # Can be float or matrix for multi-trait
                 df: float = 4.0,
                 scale: Union[float, np.ndarray, None] = None,
                 estimate_variance: bool = True,
                 estimate_scale: bool = False, # Usually false in JWAS from what was seen
                 constraint: bool = False, # For multi-trait, true means zero covariance
                 is_g_component: bool = False # Special flag for G components of Genotypes
                ):
        self.value = value
        self.df = df
        self.scale = scale
        self.estimate_variance = estimate_variance
        self.estimate_scale = estimate_scale
        self.constraint = constraint
        self.is_g_component = is_g_component # True if this is G in Genotypes (marker var)

    def __repr__(self):
        return (f"VarianceComponent(value={self.value}, df={self.df}, scale={self.scale}, "
                f"estimate_variance={self.estimate_variance}, constraint={self.constraint})")

class RandomEffectComponent:
    """
    Represents a defined random effect in the model.
    Analogous to Julia's `RandomEffect` struct.
    """
    def __init__(self,
                 term_array: List[str], # e.g., ["y1:animal", "y2:animal"]
                 variance_prior: VarianceComponent, # Corresponds to Gi in Julia
                 # Vinv_obj is the actual matrix (e.g., A-inverse) or None for IID
                 Vinv_obj: Optional[np.ndarray] = None,
                 Vinv_names: Optional[List[str]] = None, # IDs corresponding to Vinv_obj rows/cols
                 random_type: str = "I" # "I" (IID), "A" (Pedigree), "V" (User Vinv)
                ):
        self.term_array = term_array
        self.variance_prior = variance_prior # This is Gi in Julia's RandomEffect
        # GiOld and GiNew for single-trait MME updates. Initialized same as Gi.
        self.variance_prior_old = VarianceComponent(value=variance_prior.value, df=variance_prior.df, scale=variance_prior.scale,
                                           estimate_variance=variance_prior.estimate_variance,
                                           estimate_scale=variance_prior.estimate_scale,
                                           constraint=variance_prior.constraint)
        self.variance_prior_new = VarianceComponent(value=variance_prior.value, df=variance_prior.df, scale=variance_prior.scale,
                                           estimate_variance=variance_prior.estimate_variance,
                                           estimate_scale=variance_prior.estimate_scale,
                                           constraint=variance_prior.constraint)
        self.Vinv_obj = Vinv_obj
        self.Vinv_names = Vinv_names
        self.random_type = random_type

    def __repr__(self):
        return (f"RandomEffectComponent(terms={self.term_array}, type='{self.random_type}', "
                f"variance={self.variance_prior})")

class GenotypesComponent:
    """
    Represents a set of genotypes and associated parameters/variances.
    Analogous to Julia's `Genotypes` struct, focusing on aspects relevant to model setup.
    Actual genotype matrix data might be handled separately or loaded on demand.
    """
    def __init__(self,
                 name: str,
                 method: str = "BayesB", # e.g., BayesB, RR-BLUP, GBLUP
                 # G: Prior for marker effect variance(s)
                 marker_variance_prior: Optional[VarianceComponent] = None,
                 # genetic_variance: Prior for total genetic variance (esp. for GBLUP)
                 total_genetic_variance_prior: Optional[VarianceComponent] = None,
                 pi_value: Optional[Union[float, Dict[str, float]]] = None, # Prior for pi (marker not having zero effect)
                 estimate_pi: bool = False,
                 obs_ids: Optional[List[str]] = None, # Individuals with genotypes
                 marker_ids: Optional[List[str]] = None,
                 n_markers: int = 0,
                 n_obs: int = 0,
                 # Placeholder for actual genotype data reference if needed here
                 # genotype_data_ref: Any = None
                 ):
        self.name = name
        self.method = method
        self.marker_variance_prior = marker_variance_prior if marker_variance_prior else VarianceComponent(is_g_component=True)
        self.total_genetic_variance_prior = total_genetic_variance_prior if total_genetic_variance_prior else VarianceComponent()
        self.pi_value = pi_value
        self.estimate_pi = estimate_pi
        self.obs_ids = obs_ids if obs_ids else []
        self.marker_ids = marker_ids if marker_ids else []
        self.n_markers = n_markers
        self.n_obs = n_obs
        self.ntraits = 1 # Default, can be updated

        # For MCMC state, similar to Julia struct
        self.alpha: Optional[List[np.ndarray]] = None # List of arrays (one per trait) for marker effects
        self.beta: Optional[List[np.ndarray]] = None  # For BayesB individual marker variances
        self.delta: Optional[List[np.ndarray]] = None # For BayesC/BayesB inclusion indicators
        self.gamma_array: Optional[np.ndarray] = None # For BayesL

    def __repr__(self):
        return (f"GenotypesComponent(name='{self.name}', method='{self.method}', "
                f"n_markers={self.n_markers}, n_obs={self.n_obs})")


# --- Placeholder for MME_py class if not defined elsewhere ---
# This is a simplified version focusing on attributes relevant to these components
if not hasattr(__import__('__main__'), 'MME_py'): # Avoid redefinition if MME_py is already structured
    class MME_py:
        def __init__(self):
            self.model_terms: List[Any] = [] # Should be List[ModelTermPy] from build_mme
            self.model_term_dict: Dict[str, Any] = {}
            self.lhs_vec: List[str] = [] # Trait names
            self.n_models: int = 0
            self.pedigree_obj: Optional[Any] = None # Optional[Pedigree] from pedigree_module

            # Components to be managed by this module's functions
            self.residual_variance: VarianceComponent = VarianceComponent()
            self.random_effect_components: List[RandomEffectComponent] = []
            self.genotype_components: List[GenotypesComponent] = [] # mme.M in Julia

            self.mcmc_info: Dict[str, Any] = {"output_samples_frequency": 10,
                                              "single_step_analysis": False,
                                              "RRM": False, # Random Regression Model flag
                                              "double_precision": True
                                              } # Simplified MCMC info
            self.output_id: Optional[List[str]] = None
            self.obs_id: Optional[List[str]] = None # IDs in training data
            self.ped_term_vec: List[str] = [] # Polygenic effect terms
            self.polygenic_variance_prior: Optional[VarianceComponent] = None # Corresponds to mme.Gi for polygenic
            self.traits_type: List[str] = [] # e.g. ["continuous", "categorical(binary)"]

            # For input_data_validation.jl -> init_mixed_model_equations
            self.mme_lhs: Optional[Any] = None # Will be sparse matrix
            self.mme_rhs: Optional[Any] = None
            self.solution_vector: Optional[np.ndarray] = None # mme.sol

            # For input_data_validation.jl -> make_incidence_matrices
            self.output_X: Dict[str, Any] = {} # Store Zout*X for prediction equations
            self.prediction_equation_terms: List[str] = [] # Terms used in prediction

        def is_polygenic_effect_defined(self) -> bool:
            return any(rec.random_type == "A" for rec in self.random_effect_components)

# --- Input Data Validation Functions (from input_data_validation.jl) ---

def check_model_arguments_py(mme: MME_py):
    """
    Validates MCMC arguments and model setup.
    Corresponds to errors_args in Julia.
    """
    if not hasattr(mme, 'mme_lhs') or mme.mme_lhs is None : # Simplified check for build_model completion
         # In Julia, this was `mme.mmePos != 1`. Here, we check if MME is built.
         # This check might be better placed before MCMC run.
         pass # Not raising error for now, as MME construction is separate.

    if mme.mcmc_info.get("output_samples_frequency", 0) <= 0:
        raise ValueError("output_samples_frequency should be an integer > 0.")

    for gc in mme.genotype_components:
        valid_methods = ["BayesL", "BayesC", "BayesB", "BayesA", "RR-BLUP", "GBLUP"]
        if gc.method not in valid_methods:
            raise ValueError(f"{gc.method} is not available. Valid methods: {valid_methods}")

        if gc.method in ["RR-BLUP", "BayesL", "GBLUP", "BayesA"]:
            if gc.pi_value is not None and gc.pi_value != 0.0: # Assuming 0.0 means effectively false for pi
                print(f"Warning: {gc.method} runs with pi_value effectively false. User value {gc.pi_value} ignored.")
                gc.pi_value = 0.0
            if gc.estimate_pi:
                print(f"Warning: {gc.method} runs with estimate_pi = False. User value ignored.")
                gc.estimate_pi = False

        if gc.method == "BayesA":
            gc.method = "BayesB" # BayesA is BayesB with pi=0
            print("Info: BayesA is equivalent to BayesB with known pi_value=0. Running as BayesB.")
            gc.pi_value = 0.0
            gc.estimate_pi = False

        if gc.method == "GBLUP":
            if gc.total_genetic_variance_prior.value is None and gc.marker_variance_prior.value is not None:
                 # Julia error: "Please provide values for the genetic variance for GBLUP analysis"
                 # This implies total_genetic_variance_prior should be set for GBLUP.
                 # G.val in Julia for GBLUP is usually marker variance, genetic_variance.val is total.
                 # Let's assume marker_variance_prior is for GBLUP's per-SNP variance if total is not set.
                 pass # Python logic might differ slightly on how these are named/used.
            if mme.mcmc_info.get("single_step_analysis", False):
                raise ValueError("SSGBLUP is not available (or handle differently).") # Based on Julia error

        if mme.n_models > 1 and gc.pi_value is not None and isinstance(gc.pi_value, dict):
            if abs(sum(gc.pi_value.values()) - 1.0) > 1e-2:
                raise ValueError("Summation of probabilities of Pi (dict) is not equal to one.")
        elif mme.n_models > 1 and gc.pi_value is not None and not isinstance(gc.pi_value, dict):
             raise ValueError("Pi must be a dictionary for multi-trait analysis if specified per trait/group.")


    if mme.mcmc_info.get("single_step_analysis", False) and not mme.genotype_components:
        raise ValueError("Genomic information is required for single-step analysis.")

    # Causal structure check would require mme.causal_structure attribute
    # if hasattr(mme, 'causal_structure') and mme.causal_structure and mme.n_models == 1:
    #     raise ValueError("Causal structures are only allowed in multi-trait analysis.")

    if mme.n_models > 1:
        for gc in mme.genotype_components:
            if gc.marker_variance_prior.estimate_scale:
                raise ValueError("estimate_scale=true for marker variance is only supported for single trait now.")


def check_output_id_py(mme: MME_py, df_phenotypes: pd.DataFrame):
    """
    Sets default output IDs for EBV estimation and validates them.
    Corresponds to check_outputID in Julia.
    """
    output_ebv = mme.mcmc_info.get("outputEBV", False)
    output_heritability = mme.mcmc_info.get("output_heritability", False)

    if output_ebv:
        if mme.output_id is None: # Equivalent to Julia's mme.output_ID == false
            if mme.genotype_components:
                mme.output_id = list(mme.genotype_components[0].obs_ids) # Use first genotype component's IDs
            elif mme.pedigree_obj and hasattr(mme.pedigree_obj, 'ordered_ids'):
                mme.output_id = list(mme.pedigree_obj.ordered_ids)
            else: # No genomic, no pedigree
                mme.mcmc_info["outputEBV"] = False # Cannot output EBV
    else: # output_ebv is False
        mme.output_id = None

    if output_heritability and mme.genotype_components:
        mme.mcmc_info["outputEBV"] = True # Heritability implies EBV output
        if not mme.mcmc_info.get("single_step_analysis", False):
            mme.output_id = list(mme.genotype_components[0].obs_ids)
        elif mme.pedigree_obj and hasattr(mme.pedigree_obj, 'ordered_ids'):
            mme.output_id = list(mme.pedigree_obj.ordered_ids)
        # else: heritability with no pedigree in SSGBLUP is problematic, but covered by output_id logic

    # Validate output_id against available data
    if mme.output_id:
        valid_ids_for_output = set()
        if not mme.mcmc_info.get("single_step_analysis", False) and mme.genotype_components:
            # Complete genomic data, non-single-step: output IDs must be in genotyped set
            valid_ids_for_output = set(mme.genotype_components[0].obs_ids)
            msg_context = "genotyped individuals (complete genomic data, non-single-step)"
        elif mme.pedigree_obj and hasattr(mme.pedigree_obj, 'id_map'):
            # Single-step or PBLUP: output IDs must be in pedigree
            valid_ids_for_output = set(mme.pedigree_obj.id_map.keys())
            msg_context = "individuals in pedigree (single-step or PBLUP)"

        if valid_ids_for_output:
            current_output_ids = set(mme.output_id)
            if not current_output_ids.issubset(valid_ids_for_output):
                print(f"Warning: Testing individuals are not a subset of {msg_context}. "
                      "Only outputting EBV for testing individuals found in the valid set.")
                mme.output_id = list(current_output_ids.intersection(valid_ids_for_output))

        if not mme.output_id : # If intersection is empty
             mme.mcmc_info["outputEBV"] = False
             print("Warning: No valid output IDs found after filtering. Disabling EBV output.")


def check_data_consistency_py(mme: MME_py, df_phenotypes: pd.DataFrame,
                              pedigree_obj: Optional[Any] = None) -> pd.DataFrame:
    """
    Checks consistency of pedigree, genotypes, and phenotypes.
    Modifies df_phenotypes by filtering rows.
    Corresponds to check_pedigree_genotypes_phenotypes in Julia.
    """
    df_filtered = df_phenotypes.copy()
    pheno_id_col = df_filtered.columns[0]
    df_filtered[pheno_id_col] = df_filtered[pheno_id_col].astype(str).str.strip()
    all_pheno_ids = set(df_filtered[pheno_id_col])

    if pedigree_obj:
        # print("Info: Checking pedigree...")
        mme.pedigree_obj = pedigree_obj # Assume pedigree_obj is already processed Pedigree

    if mme.genotype_components:
        # print("Info: Checking genotypes...")
        # Check all genotype components use same individuals (obs_ids)
        if len(mme.genotype_components) > 1:
            first_gc_obs_ids = set(mme.genotype_components[0].obs_ids)
            for i, gc in enumerate(mme.genotype_components[1:]):
                if set(gc.obs_ids) != first_gc_obs_ids:
                    raise ValueError(f"Genotype components {mme.genotype_components[0].name} and "
                                     f"{gc.name} do not have the same set of individuals.")

        # Check genotyped individuals are in pedigree if pedigree is used
        if mme.pedigree_obj and hasattr(mme.pedigree_obj, 'id_map'):
            ped_ids = set(mme.pedigree_obj.id_map.keys())
            for gc in mme.genotype_components:
                if not set(gc.obs_ids).issubset(ped_ids):
                    raise ValueError(f"Not all genotyped individuals in component {gc.name} are found in pedigree.")

        if mme.mcmc_info.get("single_step_analysis", False) and len(mme.genotype_components) != 1:
            raise ValueError("Only one genomic category (GenotypesComponent) is allowed in single-step analysis.")

    # print("Info: Checking phenotypes...")
    # Filter phenotypes
    if mme.genotype_components and not mme.mcmc_info.get("single_step_analysis", False):
        # Complete genomic data, non-SSGBLUP: phenotyped individuals must be genotyped
        geno_ids = set(mme.genotype_components[0].obs_ids)
        if not all_pheno_ids.issubset(geno_ids):
            original_len = len(df_filtered)
            df_filtered = df_filtered[df_filtered[pheno_id_col].isin(geno_ids)]
            removed_count = original_len - len(df_filtered)
            if removed_count > 0:
                print(f"Warning: {removed_count} phenotyped individuals not genotyped were removed (non-single-step analysis).")

    if mme.pedigree_obj and hasattr(mme.pedigree_obj, 'id_map'):
        # SSGBLUP or PBLUP: phenotyped individuals must be in pedigree
        ped_ids = set(mme.pedigree_obj.id_map.keys())
        if not all_pheno_ids.issubset(ped_ids):
            original_len = len(df_filtered)
            df_filtered = df_filtered[df_filtered[pheno_id_col].isin(ped_ids)]
            removed_count = original_len - len(df_filtered)
            if removed_count > 0:
                print(f"Warning: {removed_count} phenotyped individuals not in pedigree were removed.")

        if mme.genotype_components and mme.mcmc_info.get("single_step_analysis", False):
            pheno_ids_after_ped_filter = set(df_filtered[pheno_id_col])
            geno_ids = set(mme.genotype_components[0].obs_ids)
            if pheno_ids_after_ped_filter.issubset(geno_ids):
                 # This means all individuals with phenotypes (that are also in pedigree) are genotyped.
                 # SSGBLUP is typically for when some phenotyped individuals are *not* genotyped.
                 print("Warning: All phenotyped individuals (present in pedigree) are also genotyped. "
                       "Single-step analysis might not be necessary or standard GBLUP could be used.")


    # RRM sorting (simplified, assuming 'time' column exists if RRM is true)
    if mme.mcmc_info.get("RRM", False):
        if 'time' not in df_filtered.columns:
            raise ValueError("Column 'time' required for RRM models but not found in phenotype data.")
        df_filtered = df_filtered.sort_values(by=['time', pheno_id_col])

    # Categorical trait validation
    mme.traits_type = getattr(mme, 'traits_type', ['continuous'] * mme.n_models) # Ensure traits_type exists
    for i_trait, trait_name_symbol in enumerate(mme.lhs_vec): # lhs_vec should store trait names as strings
        trait_name = str(trait_name_symbol)
        if i_trait < len(mme.traits_type) and mme.traits_type[i_trait] == "categorical":
            if trait_name not in df_filtered.columns:
                print(f"Warning: Categorical trait '{trait_name}' not found in DataFrame columns. Skipping validation for it.")
                continue

            # Drop rows where this categorical trait is missing for validation purposes
            cat_series = df_filtered[trait_name].dropna()
            if cat_series.empty:
                print(f"Warning: Categorical trait '{trait_name}' has no non-missing observations.")
                continue

            try:
                # Attempt to convert to int, handling potential errors if non-integer strings exist
                category_obs = cat_series.astype(float).astype(int) # float first to handle "1.0"
            except ValueError:
                raise ValueError(f"Categorical trait '{trait_name}' contains non-integer values that cannot be converted.")

            unique_cats = sorted(category_obs.unique())
            n_unique = len(unique_cats)
            expected_cats = list(range(1, n_unique + 1))

            if unique_cats != expected_cats:
                raise ValueError(f"For categorical trait {trait_name}, categories should be {expected_cats}; "
                                 f"instead found {unique_cats} after sorting unique observations.")
            if n_unique == 2:
                mme.traits_type[i_trait] = "categorical(binary)"

    # Print trait type info
    # (Simplified printout compared to Julia, can be expanded if needed)
    if any(ttype != "continuous" for ttype in mme.traits_type):
        # print("Info: Trait types determined as:", mme.traits_type)
        pass

    df_filtered.columns = [str(col).strip() for col in df_filtered.columns]
    return df_filtered


def set_default_priors_for_variance_components_py(mme: MME_py, df_phenotypes: pd.DataFrame):
    """
    Sets default priors for variance components if not already specified.
    Corresponds to set_default_priors_for_variance_components in Julia.
    """
    if not mme.lhs_vec: return # Should not happen if MME is properly initialized

    pheno_vars_list = []
    for trait_name_symbol in mme.lhs_vec:
        trait_name = str(trait_name_symbol)
        if trait_name in df_phenotypes.columns:
            # Ensure column is numeric and drop NAs before var calculation
            numeric_series = pd.to_numeric(df_phenotypes[trait_name], errors='coerce').dropna()
            if not numeric_series.empty:
                pheno_vars_list.append(np.var(numeric_series))
            else:
                pheno_vars_list.append(1.0) # Default if no data or all NA
        else: # Trait not in df, use default variance of 1.0
            pheno_vars_list.append(1.0)

    pheno_var_matrix = np.diag(pheno_vars_list)
    h2 = 0.5 # Default heritability assumption

    num_genetic_random_effects = len(mme.genotype_components)
    # Count polygenic random effects (type "A")
    num_genetic_random_effects += sum(1 for rec in mme.random_effect_components if rec.random_type == "A")

    # Count non-genetic, non-residual random effects
    # Julia counts 1 initially, then adds non-polygenic, non-epsilon terms.
    # Let's count explicit non-genetic iid/V type random effects. Epsilon is residual.
    num_nongenetic_other_random = sum(1 for rec in mme.random_effect_components if rec.random_type in ["I", "V"])

    # Default variance partitioning
    # If no genetic effects at all, vare takes up all h2 part, which is fine.
    var_g_total_per_effect = pheno_var_matrix * h2 / num_genetic_random_effects if num_genetic_random_effects > 0 else np.zeros_like(pheno_var_matrix)

    # Residual variance takes remaining part of h2 if no other non-genetic random effects,
    # or shares it if there are. If num_nongenetic_other_random is 0, it implies residual takes the full (1-h2) part + its share of h2 if no genetic.
    # Julia logic: vare = phenovar*h2/nongenetic_random_count, where nongenetic_random_count starts at 1 (for residual)
    # then adds other non-genetic terms. This implies residual gets a share of the h2 budget too.
    # Let's simplify: residual gets (1-h2)*phenovar + its share if other non-genetic are absent.
    # More aligned with Julia:
    effective_nongenetic_divisor = 1 + num_nongenetic_other_random # 1 for residual
    var_e_component = pheno_var_matrix * (1.0 - h2) / 1.0 # Basic residual part
    # If genetic effects are zero, residual might implicitly take more.
    # The Julia code seems to give residual `phenovar*h2/nongenetic_random_count` if `nongenetic_random_count` includes residual.
    # Let's use `vare = phenovar * (1-h2)` as base, and then adjust if priors are not set.
    # The Julia code `vare = phenovar*h2/nongenetic_random_count` seems to imply residual variance also comes from the h2 portion.
    # This is unusual. Standard is VarP = VarG + VarE. If h2=VarG/VarP, then VarE = VarP(1-h2).
    # Let's stick to standard interpretation unless prior setting requires specific partitioning.
    # The Julia code `vare = phenovar*h2/nongenetic_random_count` seems to assume total variance is split among all components.
    # Let's assume total variance is phenovar. Genetic part is h2*phenovar. Residual part is (1-h2)*phenovar.
    # Priors are then set based on these.

    # Default for Genotype Components (Marker Variances G)
    for gc in mme.genotype_components:
        if gc.total_genetic_variance_prior.value is None and gc.marker_variance_prior.value is None:
            # print(f"Info: Setting default prior for genetic/marker variance for {gc.name} from data.")
            # GBLUP uses total_genetic_variance_prior. Bayesian methods use marker_variance_prior.
            if gc.method == "GBLUP":
                gc.total_genetic_variance_prior.value = var_g_total_per_effect if mme.n_models > 1 else var_g_total_per_effect[0,0]
            else: # Bayesian methods - prior on individual marker variances
                  # This is more complex, as it's per marker. Usually small.
                  # Julia sets Mi.genetic_variance.val = varg, then G.val is reset later.
                  # So, we set the component's "total genetic variance" idea first.
                  gc.total_genetic_variance_prior.value = var_g_total_per_effect if mme.n_models > 1 else var_g_total_per_effect[0,0]
                  # Actual marker_variance_prior.value (Mi.G.val) is set in set_marker_hyperparameters_variances_and_pi

    # Default for Residual Variance (R)
    if mme.residual_variance.value is None:
        # print("Info: Setting default prior for residual variance from data.")
        base_res_var = pheno_var_matrix * (1.0 - h2) # Standard partitioning
        if mme.n_models == 1:
            is_cat_bin = mme.traits_type and (mme.traits_type[0] == "categorical" or mme.traits_type[0] == "categorical(binary)")
            mme.residual_variance.value = 1.0 if is_cat_bin else base_res_var[0,0]
            if is_cat_bin: mme.residual_variance.estimate_variance = False
        else: # multi-trait
            # For binary traits in MT, their residual var is fixed to 1, no cov with others.
            res_val_mt = np.array(base_res_var, copy=True) # Start with calculated residual var matrix
            num_binary_traits = 0
            binary_indices = [i for i,ttype in enumerate(mme.traits_type) if ttype=="categorical(binary)"]
            if binary_indices:
                for idx in binary_indices:
                    res_val_mt[idx, :] = 0 # Zero out row
                    res_val_mt[:, idx] = 0 # Zero out col
                    res_val_mt[idx, idx] = 1.0 # Set diagonal to 1
                num_binary_traits = len(binary_indices)

            mme.residual_variance.value = res_val_mt
            if num_binary_traits == mme.n_models: # All traits are binary
                mme.residual_variance.constraint = True # No covariances (already done by reset above)
                mme.residual_variance.estimate_variance = False

        # Set scale based on value and df for residual
        if mme.residual_variance.value is not None:
            if mme.n_models == 1:
                 mme.residual_variance.scale = mme.residual_variance.value * (mme.residual_variance.df - 2.0) / mme.residual_variance.df
            else: # multi-trait Inverse Wishart scale E[Sigma] = Scale / (df - p - 1) -> Scale = E[Sigma]*(df-p-1)
                 p = mme.n_models
                 mme.residual_variance.scale = mme.residual_variance.value * (mme.residual_variance.df - p - 1.0)


    # Default for other Random Effect Components (non-marker, non-polygenic)
    var_nongenetic_other_per_effect = pheno_var_matrix * h2 / effective_nongenetic_divisor if num_genetic_random_effects == 0 else \
                                      pheno_var_matrix * (1.0-h2) / effective_nongenetic_divisor # Simplified share of residual if genetic effects exist

    for rec in mme.random_effect_components:
        if rec.variance_prior.value is None:
            # print(f"Info: Setting default prior for random effect {rec.term_array} from data.")
            # Determine if it's polygenic ("A") or other ("I", "V")
            if rec.random_type == "A":
                # This is polygenic, should use var_g_total_per_effect
                # Julia: G = diagm(Zdesign*diag(varg)). If Zdesign is identity for traits, then G = varg.
                # This means the prior for polygenic effect variance is a share of total genetic variance.
                rec_val = var_g_total_per_effect if mme.n_models > 1 else var_g_total_per_effect[0,0]
            else: # IID or user Vinv, share of residual or non-assigned variance
                rec_val = var_nongenetic_other_per_effect if mme.n_models > 1 else var_nongenetic_other_per_effect[0,0]

            rec.variance_prior.value = rec_val
            # Also set for GiOld, GiNew
            rec.variance_prior_old.value = rec_val
            rec.variance_prior_new.value = rec_val

            # Set scale for this random effect's variance prior
            if mme.n_models == 1:
                rec.variance_prior.scale = rec_val * (rec.variance_prior.df - 2.0) / rec.variance_prior.df
            else:
                p = len(rec.term_array) # Number of correlated effects within this random component
                if p == 0 and mme.n_models > 0 : p = mme.n_models # if terms span all traits implicitly
                elif p == 0 and mme.n_models == 0: p = 1

                # Ensure rec_val is a matrix for multi-trait scale calculation
                current_val_matrix = np.array(rec_val)
                if current_val_matrix.ndim == 0: # scalar
                    current_val_matrix = np.diag([current_val_matrix.item()] * p) if p > 0 else np.diag([current_val_matrix.item()])
                elif current_val_matrix.ndim == 1 and p > 0 : # 1d array
                     current_val_matrix = np.diag(current_val_matrix)

                if current_val_matrix.ndim == 2 and current_val_matrix.shape[0] != p: # matrix shape mismatch with p
                    # Heuristic: if p is based on term_array, but variance is for overall traits
                    if current_val_matrix.shape[0] == mme.n_models: p = mme.n_models
                    else: p = current_val_matrix.shape[0] # best guess for p

                rec.variance_prior.scale = current_val_matrix * (rec.variance_prior.df - p - 1.0)

            rec.variance_prior_old.scale = rec.variance_prior.scale
            rec.variance_prior_new.scale = rec.variance_prior.scale

            # If this is the primary polygenic effect, update mme.polygenic_variance_prior
            if rec.random_type == "A" and mme.polygenic_variance_prior is None : # Check if it's the main one
                mme.polygenic_variance_prior = rec.variance_prior


# --- Random Effects Setup Functions (from random_effects.jl) ---

def set_random_py(mme: MME_py,
                  random_str: str,
                  G_prior_value: Union[float, np.ndarray, List[List[float]], None] = None,
                  Vinv_obj: Optional[Any] = None, # Can be Pedigree object or matrix
                  Vinv_names: Optional[List[str]] = None,
                  df_prior: float = 4.0,
                  estimate_variance: bool = True,
                  estimate_scale: bool = False,
                  constraint: bool = False):
    """
    Sets up a random effect component.
    Corresponds to set_random in Julia.
    """
    if G_prior_value is not None:
        G_prior_value_np = np.array(G_prior_value)
        # Check for positive definiteness if G_prior_value is a matrix
        if G_prior_value_np.ndim == 2:
            if not np.all(np.linalg.eigvals(G_prior_value_np) > 1e-6): # Looser check for near-zero eigvals
                raise ValueError("The covariance matrix G_prior_value is not positive definite.")
        elif G_prior_value_np.ndim == 0: # Scalar
             if G_prior_value_np <= 0: raise ValueError("Variance G_prior_value must be positive.")
             G_prior_value_np = G_prior_value_np.reshape(1,1) # Convert scalar to 1x1 matrix for consistency
        # If 1D array, it's ambiguous - could be diagonal of matrix or vector for single factor over traits.
        # Assuming for now it's passed as 2D if matrix, or scalar.
    else: # G_prior_value is None (Julia's G=false)
        G_prior_value_np = None


    parsed_term_names = []
    raw_terms = random_str.strip().split() # e.g. ["animal", "animal*age"] or ["litter"]

    for term_base_name in raw_terms:
        found_in_model = False
        for i_model, model_eq_str_or_lhs_trait in enumerate(mme.lhs_vec): # Assuming lhs_vec has trait names
            # This part needs careful alignment with how model terms are stored in mme.model_term_dict
            # Julia: string(mme.lhsVec[m])*":"*trm
            # Example: if term_base_name is "animal", and traits are "y1", "y2",
            # it generates "y1:animal", "y2:animal" if "animal" is in model eqs for y1, y2.
            # For simplicity, assume model_term_dict keys are like "trait_name:term_base_name"
            # Or, the random_str already contains full term names like "y1:animal y2:animal"

            # Simpler approach: if random_str contains ":", assume full names, else apply to all traits for that base name
            if ":" in term_base_name: # Assume full name like "y1:animal"
                if term_base_name in mme.model_term_dict:
                    parsed_term_names.append(term_base_name)
                    found_in_model = True
                # else: error or warning if full name not found?
            else: # Base name like "animal", apply to all traits where 'animal' is a term
                trait_name = str(model_eq_str_or_lhs_trait)
                full_term_name = f"{trait_name}:{term_base_name}"
                if full_term_name in mme.model_term_dict:
                    parsed_term_names.append(full_term_name)
                    found_in_model = True
        if not found_in_model and ":" not in term_base_name:
             print(f"Warning: Base random term '{term_base_name}' not found in any model equation context.")
        elif not found_in_model and ":" in term_base_name:
             print(f"Warning: Specified random term '{term_base_name}' not found in model_term_dict.")

    if not parsed_term_names:
        raise ValueError(f"No valid model terms found for random effect string: {random_str}")

    # Determine random_type and process Vinv_obj
    actual_Vinv_matrix = None
    final_Vinv_names = Vinv_names
    random_type = "I" # Default to IID

    if Vinv_obj is not None:
        # Check if Vinv_obj is a Pedigree instance (requires Pedigree class to be importable)
        # To avoid direct import issues in this standalone block, use class name string check if necessary
        # For now, assume we can check type if Pedigree is properly imported/available
        # if isinstance(Vinv_obj, Pedigree): # This would be the proper check
        if type(Vinv_obj).__name__ == 'Pedigree': # Duck-typing check for now
            if not Vinv_obj.ordered_ids: # Need ordered IDs to build A-inv
                Vinv_obj.ordered_ids = _get_ordered_ids(Vinv_obj) # from pedigree_module
            actual_Vinv_matrix = Vinv_obj.calculate_A_inverse()
            final_Vinv_names = list(Vinv_obj.ordered_ids) # Names are the ordered IDs from pedigree
            random_type = "A"
            # Update mme.pedigree_obj if this is the first pedigree-based effect
            if mme.pedigree_obj is None: mme.pedigree_obj = Vinv_obj
            mme.ped_term_vec.extend(parsed_term_names) # Add these terms as polygenic
        elif isinstance(Vinv_obj, (np.ndarray, sps.spmatrix)): # User provided matrix
            if not Vinv_names:
                raise ValueError("Vinv_names must be provided if Vinv_obj is a matrix.")
            if Vinv_obj.shape[0] != Vinv_obj.shape[1] or Vinv_obj.shape[0] != len(Vinv_names):
                raise ValueError("Vinv_obj matrix must be square and its dimensions must match length of Vinv_names.")
            actual_Vinv_matrix = Vinv_obj
            random_type = "V"
        else:
            raise TypeError("Vinv_obj must be a Pedigree object, numpy array, or scipy sparse matrix.")

    # Update ModelTerms in MME
    for term_name in parsed_term_names:
        if term_name in mme.model_term_dict:
            mme.model_term_dict[term_name].random_type = random_type
            if random_type in ["A", "V"] and final_Vinv_names:
                mme.model_term_dict[term_name].names = final_Vinv_names # Set names for level alignment

    # Create VarianceComponent for this random effect's G
    # Scale calculation needs G_prior_value and df_prior
    # Scale for Inverse Wishart E[Sigma] = Scale / (df - p - 1) -> Scale = E[Sigma]*(df-p-1)
    # Scale for Inv ChiSq E[sigma^2] = scale_param / (df - 2) -> scale_param = E[sigma^2]*(df-2) (here scale_param is S0*nu0 from some derivations)
    # Julia uses: G*(df-length(term_array)-1) for InvWishart, or G*(df-2)/df for scalar (seems different from typical InvChiSq)
    # Let's use the InvWishart form for scale.

    num_correlated_effects = len(parsed_term_names) # p for Inverse Wishart
    if G_prior_value_np is not None and G_prior_value_np.ndim == 2 and G_prior_value_np.shape[0] != num_correlated_effects :
        # If G is a matrix but its dimension doesn't match number of terms, it's an issue.
        # This can happen if G is for traits, but terms are fewer (e.g. one effect over some traits)
        # For now, assume G_prior_value_np matches num_correlated_effects if it's a matrix.
        # If G_prior_value_np is scalar, it applies to all, or gets expanded.
        pass

    scale_val = None
    if G_prior_value_np is not None:
        if num_correlated_effects > 1: # Matrix G, use Inverse Wishart scale formula
            scale_val = G_prior_value_np * (df_prior - num_correlated_effects - 1.0) if df_prior > num_correlated_effects + 1 else G_prior_value_np
        else: # Scalar G (or 1x1 matrix), use Inv Gamma / Scaled Inv ChiSq like formula
              # Julia: G*(df-2)/df. Let's adapt to common S0*nu0 (scale_param = value * df) or value * (df-2)
            scale_val = G_prior_value_np * (df_prior - 2.0) if df_prior > 2.0 else G_prior_value_np

    # If G_prior_value is None, it means default prior will be set later by set_default_priors...
    # So, variance_prior_obj might have value=None initially.
    variance_prior_obj = VarianceComponent(value=G_prior_value_np, df=df_prior, scale=scale_val,
                                         estimate_variance=estimate_variance,
                                         estimate_scale=estimate_scale,
                                         constraint=constraint)

    # If this is the primary polygenic effect, store its variance prior separately for easy access
    if random_type == "A" and mme.polygenic_variance_prior is None: # First one defined
        mme.polygenic_variance_prior = variance_prior_obj


    random_effect_comp = RandomEffectComponent(
        term_array=parsed_term_names,
        variance_prior=variance_prior_obj,
        Vinv_obj=actual_Vinv_matrix,
        Vinv_names=final_Vinv_names,
        random_type=random_type
    )
    mme.random_effect_components.append(random_effect_comp)
