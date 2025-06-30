from typing import List, Dict, Any, Optional, Union
import numpy as np
from scipy.sparse import spmatrix, csc_matrix # Added csc_matrix for A_inv type hint

from .model_term import ModelTerm
from .variance_covariance import VarianceCovariance
from .residual_variance import ResidualVariance
from .random_effect import RandomEffectTerm
from .genotypes import GenotypesData
from .mcmc_info import MCMCInfo
# Assuming Pedigree class will be defined elsewhere, e.g., pyjwas.pedigree.PedigreeData
# from ..pedigree.pedigree_data import PedigreeData # Example path

class MixedModelEquations:
    """
    Main class to hold all components of the Mixed Model Equations, model parameters,
    data, and MCMC settings/results.
    Main class encapsulating all components of a mixed model analysis,
    including model definitions, data matrices (X, y), Mixed Model Equations (LHS, RHS),
    variance components, MCMC settings, and results.

    This class serves as the central hub for a PyJWAS analysis, analogous to the
    `MME` struct in the original Julia JWAS.jl.

    Attributes:
        n_models (int): Number of traits or model equations being analyzed.
        model_equations_str (List[str]): Original model equation strings provided by the user.
        model_terms (List[ModelTerm]): Parsed model terms (fixed effects, covariates, i.i.d random).
        model_term_dict (Dict[str, ModelTerm]): Dictionary mapping term strings to ModelTerm objects.
        lhs_variables (List[str]): Names of the phenotype (left-hand side) variables.
        covariate_variables (List[str]): Names of variables designated as covariates.
        X (Optional[Union[np.ndarray, spmatrix]]): The full incidence matrix for fixed and random effects
                                                   (excluding markers handled separately).
        y_sparse (Optional[Union[np.ndarray, spmatrix]]): Phenotype vector (N*n_traits x 1).
        obs_ids (List[str]): Observation IDs corresponding to rows of y_sparse and X.
        mme_LHS (Optional[Union[np.ndarray, spmatrix]]): Left-hand side of the Mixed Model Equations.
        mme_RHS (Optional[Union[np.ndarray, spmatrix]]): Right-hand side of the Mixed Model Equations.
        pedigree_effect_terms (List[str]): Names of terms treated as pedigree-based random effects.
        pedigree_data (Optional[Any]): `PedigreeData` object if pedigree is used. (Type hint later)
        pedigree_inv_covariance (Optional[VarianceCovariance]): VC object for polygenic (co)variances (G0).
        random_effect_terms (List[RandomEffectTerm]): List of general random effect terms.
        R_variance (VarianceCovariance): VarianceCovariance object for residual (co)variance (R).
        R_old_value (Optional[Union[float, np.ndarray]]): Previous iteration's residual variance (for ST MCMC).
        missing_pattern (Optional[Any]): Information on missing data patterns (for MT advanced handling).
        residual_variance_handler (Optional[ResidualVariance]): Handler for complex residual structures.
        mean_residual_variance (Optional[Union[float, np.ndarray]]): Posterior mean of residual (co)variance.
        mean_residual_variance_sq (Optional[Union[float, np.ndarray]]): Accumulator for variance of residual (co)variance.
        inverse_weights (Optional[np.ndarray]): Inverse observation weights for heterogeneous residuals.
        genotypes_data_list (List[GenotypesData]): List of GenotypesData objects for marker effects.
        mcmc_info (Optional[MCMCInfo]): MCMC settings and parameters.
        mme_pos_counter (int): Counter for assigning column positions in MME construction.
        output_samples_terms (List[ModelTerm]): Model terms for which to save MCMC samples.
        output_ids (Optional[List[str]]): IDs for which to generate predictions.
        output_genotypes_data (Optional[Dict[str, Any]]): Genotype data for output_ids.
        output_X_matrix (Optional[Union[np.ndarray, spmatrix]]): X matrix for output_ids.
        output_results (Optional[Dict[str, Any]]): Dictionary to store final analysis results.
        solutions (Optional[np.ndarray]): Current MCMC sample for fixed and random effect solutions.
        mean_solutions (Optional[np.ndarray]): Posterior mean of solutions.
        mean_solutions_sq (Optional[np.ndarray]): Accumulator for variance of solutions.
        causal_structure_matrix (Optional[np.ndarray]): Matrix defining causal structure for SEM.
        # ... (other attributes for NN, categorical traits, etc.) ...
        trait_types (List[str]): Type of each trait (e.g., "continuous", "categorical").
        thresholds (Optional[Dict[int, np.ndarray]]): Thresholds for categorical traits.
        current_Ri_matrix (Optional[spmatrix]): Current iteration's Ri matrix for MT MCMC.
        mme_LHS_base_X_Rinv_X (Optional[spmatrix]): Base X'R_invX part of LHS, used if Lambda is updated incrementally.
    """
    def __init__(self,
                 n_models: int,
                 model_equations_str: List[str],
                 model_terms: List[ModelTerm],
                 model_term_dict: Dict[str, ModelTerm],
                 lhs_variables: List[str],
                 residual_variance_info: VarianceCovariance):
        self.n_models: int = n_models
        self.model_equations_str: List[str] = model_equations_str
        self.model_terms: List[ModelTerm] = model_terms
        self.model_term_dict: Dict[str, ModelTerm] = model_term_dict
        self.lhs_variables: List[str] = lhs_variables
        self.covariate_variables: List[str] = []
        self.X: Optional[Union[np.ndarray, spmatrix]] = None
        self.y_sparse: Optional[Union[np.ndarray, spmatrix]] = None
        self.obs_ids: List[str] = []
        self.mme_LHS: Optional[Union[np.ndarray, spmatrix]] = None
        self.mme_RHS: Optional[Union[np.ndarray, spmatrix]] = None
        self.pedigree_effect_terms: List[str] = []
        self.pedigree_data: Optional[Any] = None
        self.pedigree_inv_covariance: Optional[VarianceCovariance] = None
        self.pedigree_scale: Optional[Any] = None
        self.mean_pedigree_variance: Optional[Any] = None
        self.mean_pedigree_variance_sq: Optional[Any] = None
        self.random_effect_terms: List[RandomEffectTerm] = []
        self.R_variance: VarianceCovariance = residual_variance_info
        self.missing_pattern: Optional[Any] = None
        self.residual_variance_handler: Optional[ResidualVariance] = None
        self.R_old_value: Optional[Union[float, np.ndarray]] = None
        self.mean_residual_variance: Optional[Union[float, np.ndarray]] = None
        self.mean_residual_variance_sq: Optional[Union[float, np.ndarray]] = None
        self.inverse_weights: Optional[np.ndarray] = None
        self.genotypes_data_list: List[GenotypesData] = []
        self.mcmc_info: Optional[MCMCInfo] = None
        self.mme_pos_counter: int = 0 # Start counter at 0 for 0-indexed column positions
        self.output_samples_terms: List[ModelTerm] = []
        self.output_ids: Optional[List[str]] = None
        self.output_genotypes_data: Optional[Dict[str, Any]] = None
        self.output_X_matrix: Optional[Union[np.ndarray, spmatrix]] = None
        self.output_results: Optional[Dict[str, Any]] = None
        self.solutions: Optional[np.ndarray] = None
        self.mean_solutions: Optional[np.ndarray] = None
        self.mean_solutions_sq: Optional[np.ndarray] = None
        self.causal_structure_matrix: Optional[np.ndarray] = None
        self.nonlinear_function: Optional[Any] = None
        self.nn_weights: Optional[np.ndarray] = None
        self.nn_sigma2_yobs: Optional[float] = None
        self.nn_is_fully_connected: bool = True
        self.nn_is_activation_function_str: bool = False
        self.nn_latent_traits: Optional[List[str]] = None
        self.nn_yobs_vector: Optional[np.ndarray] = None
        self.nn_yobs_name: Optional[str] = None
        self.nn_sigma2_weights: Optional[float] = None
        self.nn_fixed_sigma2: bool = False
        self.nn_incomplete_omics: bool = False
        self.trait_types: List[str] = ['continuous'] * n_models
        self.thresholds: Optional[Dict[int, np.ndarray]] = None
        self.current_Ri_matrix: Optional[spmatrix] = None
        self.mme_LHS_base_X_Rinv_X: Optional[spmatrix] = None

    def __repr__(self) -> str:
        return (f"MixedModelEquations(n_models={self.n_models}, "
                f"lhs_variables={self.lhs_variables}, "
                f"n_model_terms={len(self.model_terms)}, "
                f"n_random_effect_terms={len(self.random_effect_terms)}, "
                f"n_genotypes_data={len(self.genotypes_data_list)})")

    def add_structured_random_effect(self,
                                     base_term_name: str,
                                     V_inv_matrix: Union[np.ndarray, csc_matrix],
                                     level_ids: List[str],
                                     prior_vc_info: VarianceCovariance,
                                     random_type_code: str = "V"): # "A" for pedigree, "V" for general user V_inv
        """
        Adds a structured random effect (e.g., polygenic animal effect) to the model.

        This method configures a RandomEffectTerm with its (co)variance structure (V_inv)
        and prior information for its variance component (G0). It also updates the
        corresponding ModelTerm entries in this MME object to align their levels and types.

        Args:
            base_term_name: The base name of the random effect (e.g., "animal").
                            This will be combined with trait names (e.g., "y1:animal").
            V_inv_matrix: The inverse of the relationship matrix (e.g., A-inverse) or
                          covariance structure for this random effect.
            level_ids: A list of string IDs corresponding to the rows/columns of V_inv_matrix,
                       in the same order.
            prior_vc_info: A VarianceCovariance object specifying the prior for the
                           (co)variance matrix G0 of this random effect.
            random_type_code: A code for the random effect type, e.g., "A" for additive
                              genetic (pedigree), "V" for user-supplied V-inverse.
        """
        if not level_ids:
            raise ValueError("level_ids must be provided for a structured random effect.")
        if V_inv_matrix.shape[0] != len(level_ids) or V_inv_matrix.shape[1] != len(level_ids):
            raise ValueError("V_inv_matrix dimensions must match the number of level_ids.")

        # Create term strings for each trait, e.g., ["y1:animal", "y2:animal"]
        effect_term_strs_for_traits: List[str] = []
        for trait_idx, trait_name in enumerate(self.lhs_variables):
            # Check if this base_term_name is actually in the model equation for this trait
            # This requires parsing model_equations_str again or checking model_terms more carefully.
            # For now, assume if user calls this, the term is intended for all traits or specific ones.
            # Let's assume it applies to all traits for now if not specific trait list is given.
            # A more robust way: user provides list of full term_strs e.g. ["y1:animal", "y2:animal"]

            # Find the ModelTerm object that corresponds to this base_term_name for each trait
            full_term_str = f"{trait_name}:{base_term_name}"
            if full_term_str not in self.model_term_dict:
                # If the term like "animal" wasn't in the original equation string for build_model,
                # it wouldn't be in model_term_dict. This method implies it *should* be there.
                # Or, this method could *create* the ModelTerm if not present.
                # For now, assume it must exist from build_model.
                print(f"Warning: Term '{full_term_str}' not found in model_term_dict. "+
                      f"Ensure '{base_term_name}' was in the model equation for trait '{trait_name}'.")
                # If we want to add it dynamically:
                # model_term = ModelTerm(term_str=base_term_name, model_index=trait_idx + 1, trait_name=trait_name)
                # self.model_terms.append(model_term)
                # self.model_term_dict[full_term_str] = model_term
                # This would require re-thinking MME component building order.
                continue # Skip if not found from initial parsing for safety

            model_term = self.model_term_dict[full_term_str]
            model_term.random_type = random_type_code
            model_term.names = list(level_ids) # Ensure it's a copy
            model_term.n_levels = len(level_ids)
            effect_term_strs_for_traits.append(full_term_str)

            if random_type_code == "A": # Specific to pedigree polygenic effect
                 if full_term_str not in self.pedigree_effect_terms:
                    self.pedigree_effect_terms.append(full_term_str)


        if not effect_term_strs_for_traits:
            print(f"Warning: Base term '{base_term_name}' was not associated with any trait. Random effect not added.")
            return

        # Create and add the RandomEffectTerm
        # The prior_vc_info is for G0 (the (co)variance of the random effect itself)
        # Its .value will be G0_inv during MCMC.
        # Its .scale is Psi_0 (prior scale matrix for G0). Its .df is nu_0.
        re_term = RandomEffectTerm(
            term_array=effect_term_strs_for_traits,
            random_type=random_type_code,
            names=list(level_ids) # Ensure it's a copy
        )
        re_term.V_inv = V_inv_matrix

        # Set up Gi, Gi_new, Gi_old with the prior information for G0.
        # Gi will store current G0_inv. Gi.scale is prior scale for G0.
        re_term.Gi = VarianceCovariance(
            value=np.copy(prior_vc_info.value) if prior_vc_info.value is not None else None, # Initial G0_inv (or None)
            df=prior_vc_info.df,
            scale=np.copy(prior_vc_info.scale) if prior_vc_info.scale is not None else None, # Prior scale matrix Psi_0 for G0
            estimate_variance=prior_vc_info.estimate_variance,
            estimate_scale=prior_vc_info.estimate_scale, # Usually false for G0's prior scale
            constraint=prior_vc_info.constraint
        )
        # Gi_new and Gi_old are used for MCMC state, especially in single-trait lambda updates
        re_term.Gi_new = re_term.Gi.deepcopy() if hasattr(re_term.Gi, 'deepcopy') else \
                         VarianceCovariance(value=np.copy(re_term.Gi.value) if re_term.Gi.value is not None else None,
                                            df=re_term.Gi.df, scale=np.copy(re_term.Gi.scale) if re_term.Gi.scale is not None else None,
                                            estimate_variance=re_term.Gi.estimate_variance)
        re_term.Gi_old = re_term.Gi.deepcopy() if hasattr(re_term.Gi, 'deepcopy') else \
                         VarianceCovariance(value=np.copy(re_term.Gi.value) if re_term.Gi.value is not None else None,
                                            df=re_term.Gi.df, scale=np.copy(re_term.Gi.scale) if re_term.Gi.scale is not None else None,
                                            estimate_variance=re_term.Gi.estimate_variance)

        self.random_effect_terms.append(re_term)

        if random_type_code == "A": # Main polygenic animal effect
            # Store a reference to its primary VC object if this is the designated one
            self.pedigree_inv_covariance = re_term.Gi
            # self.pedigree_data could also be stored here if passed in

        print(f"Added structured random effect for base term '{base_term_name}' affecting traits: {effect_term_strs_for_traits}")


if __name__ == '__main__':
    # ... (rest of __main__ block) ...
    # Example: Basic setup for a single trait model "y1 = intercept + age"
    R_info = VarianceCovariance(value=1.0, df=4.0, scale=0.5)
    intercept_term = ModelTerm(term_str="intercept", model_index=1, trait_name="y1")
    intercept_term.n_levels = 1; intercept_term.names = ["intercept"]
    age_term = ModelTerm(term_str="age", model_index=1, trait_name="y1")
    age_term.n_levels = 1; age_term.names = ["age"]; age_term.random_type = "covariate"
    model_terms_list = [intercept_term, age_term]
    model_terms_dict = {mt.trm_str: mt for mt in model_terms_list}
    mme_instance = MixedModelEquations(
        n_models=1, model_equations_str=["y1 = intercept + age"], model_terms=model_terms_list,
        model_term_dict=model_terms_dict, lhs_variables=["y1"], residual_variance_info=R_info)
    mme_instance.covariate_variables = ["age"]
    print(mme_instance)

    # Example of adding a pedigree effect (conceptual A_inv)
    n_animals_ped = 10
    dummy_A_inv = csc_matrix(np.eye(n_animals_ped))
    ped_level_ids = [f"animal_{i}" for i in range(n_animals_ped)]
    # Need to ensure "animal" term is in model_terms_list/dict if add_structured_random_effect expects it
    # For this test, let's assume "age" was meant to be "animal" for a moment
    # Or, add it:
    animal_term_for_eq = ModelTerm(term_str="animal", model_index=1, trait_name="y1")
    mme_instance.model_terms.append(animal_term_for_eq)
    mme_instance.model_term_dict["y1:animal"] = animal_term_for_eq

    G0_prior = VarianceCovariance(value=0.5, df=4.0, scale=0.1, estimate_variance=True) # Prior for sigma_a^2
    mme_instance.add_structured_random_effect("animal", dummy_A_inv, ped_level_ids, G0_prior, "A")
    print(mme_instance)
    if mme_instance.random_effect_terms:
        print(f"  Animal RE V_inv shape: {mme_instance.random_effect_terms[0].V_inv.shape}")
        print(f"  Animal RE Gi prior scale: {mme_instance.random_effect_terms[0].Gi.scale}")

    def setup_single_step_animal_model(self,
                                     pedigree_data: Any, # PedigreeData type
                                     geno_data_for_grm: GenotypesData,
                                     polygenic_variance_prior: VarianceCovariance,
                                     base_animal_effect_name: str = "animal",
                                     weight_G_for_Hinv: float = 0.95):
        """
        Configures the MME for a single-step GBLUP (ssGBLUP) model.

        This involves:
        1. Calculating H-inverse using the provided pedigree and GRM (from geno_data_for_grm).
        2. Adding the polygenic animal effect as a structured random effect, using H-inverse.

        Args:
            pedigree_data: A PedigreeData object containing the full pedigree.
            geno_data_for_grm: A GenotypesData object where .genotypes is the GRM
                               and .obs_ids are the IDs for this GRM. Must have .is_grm=True.
            polygenic_variance_prior: A VarianceCovariance object for the prior on the
                                      polygenic variance component (sigma_a^2 or G0 matrix).
            base_animal_effect_name: The base name for the animal effect in model equations
                                     (default: "animal").
            weight_G_for_Hinv: Weight for G when blending G with A22 for H-inverse construction.
        """
        from pyjwas.single_step.relationships import calculate_H_inverse # Delayed import

        if not geno_data_for_grm.is_grm:
            raise ValueError("geno_data_for_grm must be a GRM for single-step setup.")

        print("Setting up single-step animal model...")
        # Calculate H-inverse
        # Note: calculate_H_inverse modifies pedigree_data by reordering animals
        H_inv_matrix, ordered_ids_for_H = calculate_H_inverse(
            pedigree_data,
            geno_data_for_grm,
            weight_G=weight_G_for_Hinv
        )

        # Add the polygenic effect using H_inv
        self.add_structured_random_effect(
            base_term_name=base_animal_effect_name,
            V_inv_matrix=H_inv_matrix,
            level_ids=ordered_ids_for_H, # These are all animals in H, in H-order
            prior_vc_info=polygenic_variance_prior,
            random_type_code="A_Hinv" # Special type to denote H_inv based effect
        )

        # Store the pedigree data if it's to be used by other parts (e.g. info)
        self.pedigree_data = pedigree_data

        # The RandomEffectTerm for this animal effect will now have H_inv as its V_inv
        # and its .names will be ordered_ids_for_H.
        # ModelTerms like "y1:animal" will also have their .names updated to ordered_ids_for_H.
        print(f"Single-step animal model configured using H-inverse ({H_inv_matrix.shape[0]}x{H_inv_matrix.shape[0]}).")


```
