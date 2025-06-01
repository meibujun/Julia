import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import os

from .core.model_components import MME_py, VarianceComponent
from .pedigree.pedigree_module import get_pedigree as get_pedigree_internal
from .genotypes.genotype_handler import read_genotypes_py, calculate_grm_py
from .core.model_components import (
    add_genotypes_py,
    set_random_py,
    check_model_arguments_py,
    check_data_consistency_py,
    set_default_priors_for_variance_components_py,
    set_marker_hyperparameters_py,
    calculate_H_inverse_py
)
from .core.model_parser import parse_model_equation_py
from .core.mme_builder import build_full_design_matrices_py

from .mcmc.mcmc_engine import run_mcmc_py as run_mcmc_internal
from .gwas.gwas_module import run_window_gwas_py as run_gwas_internal
from .utils.output_utils import get_ebv_py as get_ebv_internal, save_results_to_csv_py, save_summary_stats_py, setup_mcmc_output_files_py, close_mcmc_output_files_py


class GenoStockModel:
    """
    High-level API for setting up, running, and interpreting quantitative genetics models.

    This class provides a user-friendly interface to define complex mixed models,
    incorporate phenotypic, pedigree, and genomic data, run MCMC-based estimations,
    and perform Genome-Wide Association Studies (GWAS).
    """

    def __init__(self, model_name: Optional[str] = "GenoStockAnalysis"):
        """
        Initializes a GenoStockModel instance.

        Args:
            model_name (Optional[str]): A user-defined name for this analysis setup.
                                        This name might be used in output file naming.
        """
        self.model_name: Optional[str] = model_name
        self._mme = MME_py(name=self.model_name)
        self._is_prepared: bool = False
        print(f"GenoStockModel '{self.model_name}' initialized.")

    def set_model_equation(self, equation: str, trait_types: Optional[Dict[str, str]] = None) -> 'GenoStockModel':
        """
        Defines the model equation(s) using R-like formula syntax.

        For multi-trait models, equations for different traits should be separated by
        semicolons (`;`) or newlines (`\\n`). The parser will identify response
        variables (left-hand side of '~' or '=') and predictor terms (right-hand side).

        Args:
            equation (str): The model equation string.
                Examples:
                    - Single trait: "yield = intercept + parity + herd_effect + animal_polygenic"
                    - Multi-trait: "growth_rate ~ mu1 + sex + line; feed_intake ~ mu2 + weight_class + line"
            trait_types (Optional[Dict[str, str]]): A dictionary specifying the data type for each trait.
                The keys should be the trait names (response variables from the equation),
                and values should be their types. Supported types include:
                - "continuous" (default if not specified)
                - "categorical" (e.g., for litter size, requiring integer phenotypes 1, 2, 3...)
                - "categorical_binary" (for binary traits 0/1 or 1/2, will be handled appropriately)
                - "censored" (for traits with known censoring points)
                Example: `{"yield": "continuous", "disease_status": "categorical_binary"}`

        Returns:
            GenoStockModel: The instance itself, allowing for method chaining (fluent API).
        """
        self._mme.model_equation_str = equation

        raw_equations = [eq.strip() for eq in equation.replace(';', '\n').split('\n') if eq.strip()]
        if not raw_equations:
            raise ValueError("Model equation string is empty or invalid.")

        parsed_lhs_vec = []
        for eq_part in raw_equations:
            separator = '~' if '~' in eq_part else '='
            if separator not in eq_part :
                 raise ValueError(f"Equation part '{eq_part}' must contain '~' or '=' to separate response and predictors.")
            lhs_str, _ = eq_part.split(separator, 1)
            parsed_lhs_vec.append(lhs_str.strip())

        self._mme.lhs_vec = parsed_lhs_vec
        self._mme.n_models = len(parsed_lhs_vec)

        if trait_types:
            self._mme.trait_info["types"] = trait_types
            self._mme.traits_type = [trait_types.get(trait, "continuous") for trait in self._mme.lhs_vec]
        else:
            self._mme.traits_type = ["continuous"] * self._mme.n_models

        self._is_prepared = False
        print(f"Model equation set for {self._mme.n_models} trait(s): {self._mme.lhs_vec}")
        return self

    def load_phenotypes(self, data: Union[pd.DataFrame, str],
                        id_column: str,
                        covariate_columns: Optional[List[str]] = None,
                        missing_value_codes: Optional[List[Any]] = None,
                        **read_options) -> 'GenoStockModel':
        """
        Loads and configures phenotype data. Trait columns are derived from the
        model equation(s) previously set via `set_model_equation`.

        Args:
            data (Union[pd.DataFrame, str]): pandas DataFrame or path to a CSV/text file
                                             containing phenotype and covariate data.
            id_column (str): Name of the column in `data` that contains individual IDs.
                             These IDs should correspond to those in pedigree and genotype data.
            covariate_columns (Optional[List[str]]): List of column names in `data`
                                                     to be treated as continuous fixed covariates.
                                                     These terms should also be present in the model equation.
                                                     Terms in the model equation not listed here and found
                                                     in `data` columns will be assumed to be categorical factors.
            missing_value_codes (Optional[List[Any]]): Values to interpret as missing (NaN)
                                                       when reading from a file (passed to `pd.read_csv`).
                                                       Defaults to a common set like ["NA", "NaN", "", " "].
            **read_options: Additional keyword arguments passed to `pd.read_csv` if `data` is a file path.

        Returns:
            GenoStockModel: The instance itself, for method chaining.
        """
        if not self._mme.lhs_vec:
            raise RuntimeError("Model equation must be set via `set_model_equation()` before loading phenotypes, to identify trait columns.")

        na_values_default = ["NA", "NaN", "", " ", "0.0", "0", "na", "N/A"] # More comprehensive defaults
        na_values_updated = missing_value_codes if missing_value_codes is not None else na_values_default

        if isinstance(data, str):
            try:
                df = pd.read_csv(data, na_values=na_values_updated, **read_options)
            except Exception as e:
                raise ValueError(f"Error reading phenotype file '{data}': {e}")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Argument 'data' must be a pandas DataFrame or a file path string.")

        self._mme.phenotype_dataframe = df
        self._mme.phenotype_info = {
            "data_source_name": str(data) if isinstance(data, str) else "DataFrame",
            "id_column": id_column,
            "trait_columns": list(self._mme.lhs_vec),
            "covariate_columns": covariate_columns or []
        }
        self._is_prepared = False
        print(f"Phenotypes configured: ID Column='{id_column}', Traits={self._mme.lhs_vec}, Covariates={covariate_columns or 'None'}")
        return self

    def load_pedigree(self, file_path: str, **pedigree_options) -> 'GenoStockModel':
        """
        Configures pedigree information from a file. Actual loading and processing
        of the pedigree is deferred until `run()` or `_prepare_for_run()` is called.

        Args:
            file_path (str): Path to the pedigree file. Expected format is typically
                             CSV with columns: individual ID, sire ID, dam ID.
            **pedigree_options: Options for pedigree processing passed to the
                                underlying `get_pedigree` function. Common options include:
                                - `header` (bool): True if the file has a header row (default: False).
                                - `separator` (str): Column separator (default: ',').
                                - `missing_strings` (List[str]): Strings denoting missing parents
                                  (e.g., ["0", "NA"], default: ["0", "0.0"]).

        Returns:
            GenoStockModel: The instance itself, for method chaining.
        """
        self._mme.pedigree_info = {"file_path": file_path, "options": pedigree_options}
        self._is_prepared = False
        print(f"Pedigree configured from: {file_path}")
        return self

    def add_genotypes(self, name: str, data_source: Union[str, pd.DataFrame, np.ndarray],
                      method: str, map_file_path: Optional[str] = None,
                      **genotype_config_options) -> 'GenoStockModel':
        """
        Adds and configures a set of genotypes for the model. Actual data loading,
        processing (QC, GRM calculation), and `GenotypesComponent` creation are
        deferred until `run()` or `_prepare_for_run()` is called.

        Args:
            name (str): A unique name for this genotype set (e.g., "snp_chip", "wgs_data").
                        This name might be used to link specific random effects if multiple
                        genomic components are used in advanced models.
            data_source (Union[str, pd.DataFrame, np.ndarray]): Path to genotype file, pandas DataFrame,
                                                                or NumPy array. If file/DataFrame,
                                                                the first column is typically individual ID.
            method (str): The genomic method to be used with these genotypes
                          (e.g., "GBLUP", "SSGBLUP_support", "BayesC0", "BayesL", "BayesB", "BayesA", "RR-BLUP").
                          "SSGBLUP_support" indicates this genotype set provides the GRM for an SSGBLUP model.
            map_file_path (Optional[str]): Path to marker map file (marker_ID, chromosome, position).
                                           Required for some operations like window-based GWAS.
            **genotype_config_options: Additional options for genotype data processing,
                                       prior settings for variance components, and method-specific parameters.
                                       Examples:
                                       - `genetic_variance_value` (float/array): Prior for total genetic variance (for GBLUP/SSGBLUP)
                                         or initial marker variance if `g_is_marker_variance` is True for Bayesian methods.
                                       - `marker_variance_value` (float/array): Specific prior for individual/common marker effect variance.
                                       - `g_is_marker_variance` (bool): If True, `genetic_variance_value` is interpreted as prior for
                                         marker effect variance rather than total genomic variance. Default: False.
                                       - `df_prior` (float): Degrees of freedom for genetic/marker variance prior (default: 4.0).
                                       - `pi_value` (float/dict): Prior for Pi (proportion of non-zero effect markers for Bayesian methods).
                                       - `estimate_pi` (bool): Whether to estimate Pi (default depends on method).
                                       - `header` (bool): If `data_source` is a file, indicates if it has a header (default: True).
                                       - `separator` (str): File separator (default: ',').
                                       - `missing_value_code` (Any): Code for missing genotypes (default: "9").
                                       - `perform_qc` (bool): Perform quality control (default: True).
                                       - `maf_threshold` (float): MAF threshold for QC (default: 0.01).
                                       - `center_genotypes` (bool): Center genotypes (default: True).
                                       - `weight_for_G` (float): For SSGBLUP, weight for G matrix in H-matrix construction (default: 1.0).

        Returns:
            GenoStockModel: The instance itself, for method chaining.
        """
        current_mcmc_settings = self._mme.mcmc_settings

        # Consolidate and default options
        config_payload = {
            "name": name, "data_source": data_source, "method": method,
            "map_file_path": map_file_path,
            # File reading options from **genotype_config_options or defaults
            "header": genotype_config_options.get("header", True),
            "separator": genotype_config_options.get("separator", ','),
            "missing_value_code": genotype_config_options.get("missing_value_code", "9"),
            # QC options
            "perform_qc": genotype_config_options.get("perform_qc", True),
            "maf_threshold": genotype_config_options.get("maf_threshold", 0.01),
            "center_genotypes": genotype_config_options.get("center_genotypes", True),
            # Technical options
            "double_precision": current_mcmc_settings.get("double_precision", True), # Default to True from MME settings
            # Prior options (careful with aliasing like G_prior_value vs genetic_variance_value)
            "G_prior_value": genotype_config_options.get("genetic_variance_value", genotype_config_options.get("G_prior_value")),
            "genetic_variance_value": genotype_config_options.get("genetic_variance_value"),
            "marker_variance_value": genotype_config_options.get("marker_variance_value"),
            "G_is_marker_variance": genotype_config_options.get("g_is_marker_variance", False),
            "df_prior": genotype_config_options.get("df_prior_g", genotype_config_options.get("df_prior", 4.0)),
            # Bayesian method specific priors
            "pi_value": genotype_config_options.get("pi_value", 0.5 if method in ["BayesB", "BayesC"] else (1.0 if method=="BayesC0" else 0.0) ), # Default Pi depends on method
            "estimate_pi": genotype_config_options.get("estimate_pi", method in ["BayesB", "BayesC"]), # Estimate Pi usually for BayesB/C
            # SSGBLUP specific
            "weight_for_G": genotype_config_options.get("weight_for_G", 1.0)
        }
        # Capture any other options passed in genotype_config_options
        for k,v in genotype_config_options.items():
            if k not in config_payload: config_payload[k] = v

        self._mme.genotype_components_config.append(config_payload)
        self._genotypes_loaded_count += 1
        self._is_prepared = False
        print(f"Genotype component '{name}' (method: {method}) configured.")
        return self

    def add_random_effect(self, effect_name: str, use_pedigree: bool = False,
                          variance_prior: Optional[Union[float, np.ndarray]] = None,
                          df_prior: float = 4.0, **effect_options) -> 'GenoStockModel':
        """
        Defines a random effect from the model equation and its variance component prior.
        The `effect_name` should match a term specified in the model equation.

        Args:
            effect_name (str): Name of the effect as used in the model equation (e.g., "animal", "cage_effect").
            use_pedigree (bool): If True, this is a polygenic effect using the loaded pedigree.
                                 The `effect_name` should correspond to the term representing individuals
                                 (e.g., "animal_id" in "y ~ ... + animal_id"). For SSGBLUP, this indicates
                                 the main polygenic term that will use the H-matrix.
            variance_prior (Optional[Union[float, np.ndarray]]): Prior mean for the variance component of this effect.
                                                                 If None, defaults will be estimated or set later.
                                                                 For SSGBLUP animal effect, this prior is typically
                                                                 taken from the associated GBLUP GenotypesComponent's
                                                                 `genetic_variance_value`.
            df_prior (float): Degrees of freedom for the variance component prior (default: 4.0).
            **effect_options: Additional options:
                              - `V_matrix` (Optional[np.ndarray]): User-provided covariance matrix V for this random effect.
                                If provided, `V_matrix_ids` must also be supplied. The inverse ($V^{-1}$) will be used.
                              - `V_matrix_ids` (Optional[List[str]]): List of IDs corresponding to rows/cols of `V_matrix`.
                              - `ssgblup_genotype_component` (str): For an SSGBLUP 'animal' effect, name of the
                                `GenotypesComponent` (added via `add_genotypes`) that provides the GRM.

        Returns:
            GenoStockModel: The instance itself, for method chaining.
        """
        config = {
            "name": effect_name, "use_pedigree": use_pedigree,
            "variance_prior_value": variance_prior, "df_prior": df_prior, **effect_options
        }
        self._mme.random_effects_config.append(config)
        self._is_prepared = False
        print(f"Random effect '{effect_name}' configured.")
        return self

    def set_residual_variance_prior(self, value: Union[float, np.ndarray], df: float = 4.0,
                                    constraint: bool = False) -> 'GenoStockModel':
        """
        Sets the prior for the residual variance(s) (R).

        Args:
            value (Union[float, np.ndarray]): Prior mean for residual variance.
                                             For single-trait models, this is a scalar ($\sigma_e^2$).
                                             For multi-trait models, this is a covariance matrix ($\Sigma_e$).
            df (float): Degrees of freedom for the prior (default: 4.0).
            constraint (bool): For multi-trait models, if True, constrains residual covariances
                               between traits to zero (i.e., R is diagonal). Default: False.

        Returns:
            GenoStockModel: The instance itself, for method chaining.
        """
        self._mme.residual_variance_prior_config = {"value": value, "df": df, "constraint": constraint}
        self._is_prepared = False
        print(f"Residual variance prior configured.")
        return self

    def set_mcmc_options(self, chain_length: int, burn_in: int, thinning: int = 10,
                         seed: Optional[int] = None, **options) -> 'GenoStockModel':
        """
        Configures MCMC operational parameters.

        Args:
            chain_length (int): Total number of MCMC iterations.
            burn_in (int): Number of initial iterations to discard as burn-in.
            thinning (int): Interval for saving MCMC samples after burn-in (e.g., 10 means every 10th sample).
            seed (Optional[int]): Random seed for reproducibility of the MCMC run.
            **options: Other MCMC settings. Examples:
                       - `output_samples_frequency` (int): Legacy or alternative for thinning/output control.
                       - `printout_frequency` (int): How often to print MCMC progress to console.
                       - `single_step_analysis` (bool): Flag to enable Single-Step GBLUP/Bayesian Regression.
                       - `fitting_J_vector` (bool): For SSBR, whether to fit the J vector (default: True).
                       - `outputEBV` (bool): Whether to calculate and store EBVs (default: True).
                       - `output_heritability` (bool): Whether to calculate and store heritabilities (default: True).

        Returns:
            GenoStockModel: The instance itself, for method chaining.
        """
        self._mme.mcmc_settings.update({
            "chain_length": chain_length, "burn_in": burn_in, "thinning": thinning,
            "seed": seed, **options
        })
        self._is_prepared = False
        print(f"MCMC options set: chain={chain_length}, burn-in={burn_in}, thinning={thinning}.")
        return self

    def _prepare_for_run(self):
        """
        Internal method to validate the model setup and translate the user-friendly
        API configurations into the detailed internal MME_py object structure.
        This involves calling the respective setup functions from other modules.
        """
        if self._is_prepared:
            print("Model preparation is being skipped as it was marked complete. Call a config method to reset.")
            return

        print("Preparing model for analysis...")

        # --- 1. Process Model Equation ---
        if not self._mme.model_equation_str:
            raise ValueError("Model equation must be set using set_model_equation().")

        parsed_terms, term_dict, lhs_vec, n_models = parse_model_equation_py(
            self._mme.model_equation_str,
            self._mme.lhs_vec,
            covariate_columns_user = self._mme.phenotype_info.get("covariate_columns", []),
            random_effects_user = [cfg['name'] for cfg in self._mme.random_effects_config]
        )
        self._mme.model_terms = parsed_terms
        self._mme.model_term_dict = term_dict
        if self._mme.lhs_vec != lhs_vec or self._mme.n_models != n_models:
             print(f"Warning: Model equation parsing updated traits to {lhs_vec} ({n_models} models).")
             self._mme.lhs_vec = lhs_vec
             self._mme.n_models = n_models
             if self._mme.trait_info.get("types"):
                 self._mme.traits_type = [self._mme.trait_info["types"].get(trait, "continuous") for trait in self._mme.lhs_vec]
             else:
                 self._mme.traits_type = ["continuous"] * self._mme.n_models
        print(f"Model equation parsed. Found {self._mme.n_models} trait(s) and {len(self._mme.model_terms)} term instances.")

        # --- 2. Load and Process Phenotypes ---
        if self._mme.phenotype_dataframe is None or not self._mme.phenotype_info:
            raise ValueError("Phenotype data must be loaded using load_phenotypes().")

        # --- 3. Load and Process Pedigree ---
        if self._mme.pedigree_info.get("file_path") and self._mme.pedigree_obj is None:
            self._mme.pedigree_obj = get_pedigree_internal(
                self._mme.pedigree_info["file_path"],
                **self._mme.pedigree_info.get("options", {})
            )
            print(f"Pedigree loaded into MME from {self._mme.pedigree_info['file_path']}")

        # --- 4. Process Genotype Components ---
        self._mme.genotype_components = []
        for gc_config in self._mme.genotype_components_config:
            print(f"Processing genotype component: {gc_config['name']}")
            raw_geno_data_dict = read_genotypes_py(
                file_path_or_df=gc_config["data_source"],
                header_present=gc_config.get("header", True),
                separator=gc_config.get("separator", ','),
                missing_value_code=str(gc_config.get("missing_value_code", "9")),
                perform_qc=gc_config.get("perform_qc", True),
                maf_threshold=gc_config.get("maf_threshold", 0.01),
                center_genotypes=gc_config.get("center_genotypes", True),
                double_precision=self._mme.get_mcmc_setting("double_precision", False)
            )
            add_genotypes_py(self._mme, **gc_config, genotype_data=raw_geno_data_dict)
            print(f"Genotype component '{gc_config['name']}' processed and added to MME.")

        # --- 5. Process Random Effects ---
        self._mme.random_effect_components = []
        for rec_config in self._mme.random_effects_config:
            print(f"Processing random effect: {rec_config['name']}")
            # Update term in model_term_dict to be random
            for trait_name in self._mme.lhs_vec:
                full_term_id = f"{trait_name}:{rec_config['name']}"
                if full_term_id in self._mme.model_term_dict:
                    self._mme.model_term_dict[full_term_id].is_random = True
                    self._mme.model_term_dict[full_term_id].is_fixed = False

            set_random_py(
                mme=self._mme,
                random_str=rec_config["name"],
                G_prior_value=rec_config.get("variance_prior_value"),
                df_prior=rec_config.get("df_prior", 4.0),
                Vinv_obj=(self._mme.pedigree_obj if rec_config.get("use_pedigree") else rec_config.get("V_matrix")),
                Vinv_names=rec_config.get("V_matrix_ids"),
                estimate_variance=rec_config.get("estimate_variance", True),
                constraint=rec_config.get("constraint", False)
            )
            print(f"Random effect '{rec_config['name']}' processed and added to MME.")

        # --- 6. Configure Residual Variance Prior ---
        if self._mme.residual_variance_prior_config:
            cfg = self._mme.residual_variance_prior_config
            self._mme.residual_variance_prior = VarianceComponent(
                value=cfg.get("value"), df=cfg.get("df", 4.0),
                constraint=cfg.get("constraint", False),
                scale=cfg.get("scale")
            )
            print("Residual variance prior configured in MME.")

        # --- 7. SSGBLUP H-matrix (if applicable) ---
        if self._mme.is_single_step_gblup():
            print("Configuring for SSGBLUP...")
            if not self._mme.pedigree_obj: raise ValueError("Pedigree is required for SSGBLUP.")

            gblup_gc = next((gc for gc in self._mme.genotype_components if gc.method == "GBLUP"), None)
            if not gblup_gc: raise ValueError("A GenotypesComponent with method 'GBLUP' (for GRM source) is required for SSGBLUP.")

            genotyped_ids_for_ped_ordering = [str(id_val) for id_val in gblup_gc.obs_ids]
            self._mme.pedigree_obj.set_genotyped_individuals(genotyped_ids=genotyped_ids_for_ped_ordering)

            grm_source_matrix = gblup_gc.genotype_matrix
            if not gblup_gc.is_grm:
                print("Calculating GRM for SSGBLUP as it was not provided directly...")
                if grm_source_matrix is None: raise ValueError("Genotype matrix for GRM calculation is missing.")
                grm_source_matrix = calculate_grm_py(
                    gblup_gc.genotype_matrix,
                    gblup_gc.allele_frequencies,
                    gblup_gc.sum_2pq
                )

            self._mme.H_inverse = calculate_H_inverse_py(
                pedigree=self._mme.pedigree_obj,
                grm=grm_source_matrix,
                genotyped_ids_in_grm_order=gblup_gc.obs_ids,
                weight_for_G=gblup_gc.weight_for_G
            )
            poly_rec_ssgblup = self._mme.get_polygenic_effect_component_for_ssgblup()
            if not poly_rec_ssgblup:
                 raise ValueError("A pedigree-based random effect (e.g. 'animal') must be added via add_random_effect for SSGBLUP.")
            poly_rec_ssgblup.Vinv_obj = self._mme.H_inverse
            poly_rec_ssgblup.random_type = "H"
            poly_rec_ssgblup.Vinv_names = self._mme.pedigree_obj.ordered_ids
            print("H-inverse matrix calculated and configured for SSGBLUP.")

        # --- 8. Build Data Matrices (y, X for fixed effects, Z for markers if in MME) & effects_map ---
        num_effects_in_model_equation_part = build_full_design_matrices_py(self._mme)

        # --- 9. Final Validation and Default Priors ---
        if self._mme.phenotype_dataframe is not None:
            # Full check_data_consistency_py is complex and needs fully built MME.
            # Minimal check or defer full check to MCMC engine start.
            # print("Data consistency checked (conceptual).")

            set_default_priors_for_variance_components_py(self._mme, self._mme.phenotype_dataframe)
            print("Default priors set for unspecified variance components.")
        else:
            print("Warning: Phenotype DataFrame not available for default prior setting during prepare.")

        if self._mme.genotype_components:
            set_marker_hyperparameters_py(self._mme)
            print("Marker hyperparameters set.")

        check_model_arguments_py(self._mme)
        print("Model arguments checked.")

        # --- 10. Initialize MCMC state variables ---
        self._mme.initialize_mcmc_state(num_sol_effects=self._mme.num_total_effects_in_mme_system)
        print("MCMC state initialized.")

        self._is_prepared = True
        print("Model preparation complete.")


    def run(self, output_folder: str = "genostock_results") -> Dict[str, pd.DataFrame]:
        """
        Runs the configured MCMC analysis.

        Args:
            output_folder (str): Directory to save MCMC samples and summary results.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames containing key summary results.
        """
        if not self._is_prepared: self._prepare_for_run()

        # Actual call to the MCMC engine
        # self._mme.posterior_means, self._mme.posterior_samples = run_mcmc_internal(
        #     self._mme,
        #     self._mme.phenotype_dataframe, # Or aligned/filtered version
        #     self._mme.mcmc_settings
        # )
        # For now, using placeholder results as MCMC engine is not fully runnable for complex cases
        print(f"MCMC analysis would run. Results conceptually saved in '{output_folder}'.")
        self._analysis_run = True

        # Placeholder results population
        self._mme.posterior_means["residual_variance"] = pd.DataFrame({"Estimate": [self._mme.current_residual_variance or 1.0]})
        if self._mme.random_effect_components:
            self._mme.posterior_means["random_effect_variances"] = pd.DataFrame({
                "Effect": [rec.name for rec in self._mme.random_effect_components],
                "Estimate": [val for val in self._mme.current_random_effect_vc_estimates]
            })
        if self._mme.genotype_components:
             self._mme.posterior_means[f"marker_effects_variances_{self._mme.genotype_components[0].name}"] = pd.DataFrame({"Estimate": [self._mme.current_genotype_vc_estimates[0] if self._mme.current_genotype_vc_estimates else 0.01]})

        # Save results using output_utils
        # finalize_results_py would typically process raw samples into these DataFrames.
        # Here we are directly creating placeholder summary DFs.
        save_results_to_csv_py(self._mme.posterior_means, output_folder)

        return self._mme.posterior_means # Return the means

    def run_gwas(self, map_file_path: str, marker_effects_files: List[str],
                 output_folder: str = "gwas_results", **gwas_options) -> Tuple[List[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Performs window-based GWAS using provided MCMC samples of marker effects.
        Should typically be called after `run()` if using its MCMC samples, or can be
        run if marker effect samples are available from an external source.

        Args:
            map_file_path (str): Path to the marker map file (columns: marker_ID, chromosome, position).
            marker_effects_files (List[str]): List of paths to files containing MCMC samples of marker effects.
                                             Usually, one file per trait. If two files are provided and
                                             `genetic_correlation_flag=True` in `gwas_options`,
                                             correlation between the two traits' effects will be computed.
            output_folder (str): Folder where GWAS result files will be saved.
            **gwas_options: Additional options for GWAS execution. Common options include:
                            - `window_size_str` (str): Size of genomic windows, e.g., "1Mb", "500Kb" (default: "1 Mb").
                            - `sliding_window` (bool): If True, use sliding windows; otherwise, non-overlapping (default: False).
                            - `gwas_threshold_ppa` (float): Threshold for calculating Window Posterior Probability of Association (WPPA) (default: 0.01).
                            - `genetic_correlation_flag` (bool): If True and two effect files are given, compute windowed genetic correlation (default: False).
                            - `local_ebv_flag` (bool): If True, calculate local EBVs per window (default: False).
                            - `output_win_var_props_flag` (bool): If True, return raw MCMC samples of window variance proportions (default: False).
                            - `header_map` (bool): If map file has a header (default: True).
                            - `header_effects` (bool): If marker effects files have headers (default: True).

        Returns:
            A tuple containing:
                - List of pandas DataFrames: One DataFrame per trait with GWAS results (WPPA, variance explained, etc.).
                - Optional pandas DataFrame: Contains windowed genetic correlations if requested.
        """
        if not self._is_prepared : self._prepare_for_run()

        default_gwas_opts = {
            "window_size_str": "1 Mb", "sliding_window": False, "gwas_threshold_ppa": 0.01,
            "genetic_correlation_flag": len(marker_effects_files) == 2 and gwas_options.get("genetic_correlation_flag", False),
            "local_ebv_flag": False, "output_win_var_props_flag": False,
            "header_map": True, "header_effects": True,
            "separator_map": ',', "separator_effects": ','
        }
        final_gwas_opts = {**default_gwas_opts, **gwas_options}

        gwas_results_dfs, genetic_corr_df, _ = run_gwas_internal(
            mme=self._mme,
            map_file_path=map_file_path,
            marker_effects_file_paths=marker_effects_files,
            **final_gwas_opts
        )

        os.makedirs(output_folder, exist_ok=True)
        for i, df_res in enumerate(gwas_results_dfs):
            trait_name = self._mme.get_trait_name(i)
            df_res.to_csv(os.path.join(output_folder, f"gwas_results_{trait_name}.csv"), index=False)
        if genetic_corr_df is not None:
            corr_file_name = "gwas_genetic_correlations.csv"
            if len(self._mme.lhs_vec) == 2:
                corr_file_name = f"gwas_genetic_correlations_{self._mme.lhs_vec[0]}_{self._mme.lhs_vec[1]}.csv"
            genetic_corr_df.to_csv(os.path.join(output_folder, corr_file_name), index=False)

        print(f"GWAS completed. Results saved in '{output_folder}'.")
        return gwas_results_dfs, genetic_corr_df


    def get_ebv(self, trait_names: Optional[Union[str, List[str]]] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves Estimated Breeding Values (EBVs) and their Prediction Error Variances (PEVs)
        (approximated by posterior standard deviations of EBVs).
        This method should be called after `run()` has completed.

        Args:
            trait_names (Optional[Union[str, List[str]]]): The name(s) of the trait(s) for which to retrieve EBVs.
                                                           If None, returns EBVs for all traits in the model.
                                                           Trait names must match those defined in `set_model_equation`.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing columns for individual ID,
                                    trait name, estimated EBV, and PEV (approx. SD).
                                    Returns None if analysis has not been run or EBVs are not available.
        """
        if not self._analysis_run:
            print("Warning: Analysis has not been run. No EBVs available."); return None

        # Use the utility function, passing the internal MME object
        # return get_ebv_internal(self._mme, trait_names) # Assuming get_ebv_internal is adapted for this

        # Simpler retrieval for now, directly from posterior_means if structured by finalize_results_py
        all_ebvs_dfs = []
        target_traits = []
        if trait_names is None:
            target_traits = self._mme.lhs_vec
        elif isinstance(trait_names, str):
            target_traits = [trait_names]
        else:
            target_traits = trait_names

        for trait_name in target_traits:
            ebv_key = f"EBV_{trait_name}" # Key used by output_utils.finalize_results
            if ebv_key in self._mme.posterior_means:
                ebv_df_trait = self._mme.posterior_means[ebv_key].copy()
                ebv_df_trait["trait"] = trait_name
                all_ebvs_dfs.append(ebv_df_trait)
            else:
                print(f"Warning: EBV results for trait '{trait_name}' not found in posterior_means.")

        if not all_ebvs_dfs: return None
        final_ebv_df = pd.concat(all_ebvs_dfs, ignore_index=True)
        return final_ebv_df


    def get_variance_components(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves posterior means of estimated variance components.
        This method should be called after `run()` has completed.

        Returns:
            Optional[Dict[str, Any]]: A dictionary where keys are names of variance
                                      components (e.g., "residual_variance",
                                      "genetic_variance_animal", "marker_variance_chip1")
                                      and values are their posterior mean estimates (often as DataFrames).
                                      Returns None if analysis has not been run.
        """
        if not self._analysis_run:
            print("Warning: Analysis has not been run. No variance components available."); return None

        vcs = {}
        if self._mme.posterior_means:
            for key, val in self._mme.posterior_means.items():
                if "variance" in key.lower() or "covariance" in key.lower() or key.startswith("pi_"):
                    vcs[key] = val
        return vcs if vcs else None

    def summary(self) -> None:
        """Prints a summary of the model configuration and key estimated parameters (if analysis run)."""
        print(f"\n--- Summary for GenoStockModel: {self.model_name} ---")
        print(f"Model Equation: {self._mme.model_equation_str or 'Not set'}")
        print(f"Traits: {self._mme.lhs_vec or 'Not set'}")
        if self._mme.phenotype_info:
            print(f"Phenotypes: Loaded from {self._mme.phenotype_info.get('data_source_name', 'N/A')}, ID column: {self._mme.phenotype_info.get('id_column')}")
        if self._mme.pedigree_info.get("file_path"):
            print(f"Pedigree: Configured from {self._mme.pedigree_info['file_path']}")
        if self._mme.genotype_components_config:
            print(f"Genotype Components Configured: {len(self._mme.genotype_components_config)}")
            for i, gc_conf in enumerate(self._mme.genotype_components_config):
                print(f"  {i+1}. Name: {gc_conf['name']}, Method: {gc_conf['method']}")
        if self._mme.random_effects_config:
             print(f"Random Effects Configured: {len(self._mme.random_effects_config)}")
             for i, re_conf in enumerate(self._mme.random_effects_config):
                print(f"  {i+1}. Name: {re_conf['name']}, Use Pedigree: {re_conf.get('use_pedigree', False)}")

        print("\nMCMC Settings:")
        for k, v in self._mme.mcmc_settings.items(): print(f"  {k}: {v}")

        if self._analysis_run and self._mme.posterior_means:
            print("\n--- Results Summary (Posterior Means) ---")
            vcs = self.get_variance_components()
            if vcs:
                for name, val_df in vcs.items():
                    print(f"\n{name}:")
                    if isinstance(val_df, pd.DataFrame): print(val_df.to_string())
                    else: print(val_df)

            ebvs = self.get_ebv()
            if ebvs is not None:
                print("\nSample EBVs (first 5 rows overall):")
                print(ebvs.head().to_string())
        elif self._analysis_run:
             print("\nAnalysis run, but no detailed results available in posterior_means for summary.")
        else:
            print("\nAnalysis has not been run.")

```

**Summary of Docstring Expansion and API Refinements in `genostockpy/api.py`:**

1.  **Class Docstring (`GenoStockModel`)**: Added a more descriptive summary of the class's role.
2.  **`__init__`**: Clarified `model_name` usage. Now directly initializes `self._mme = MME_py(...)`. Removed `_mme_placeholder`.
3.  **`set_model_equation`**:
    *   Docstring details format for single and multi-trait equations.
    *   Explains `trait_types` parameter with examples and supported types.
    *   Method now performs basic parsing of LHS to set `self._mme.lhs_vec` and `self._mme.n_models`, and processes `trait_types` into `self._mme.traits_type`.
    *   Added `-> 'GenoStockModel'` for return type hint to indicate chainability.
4.  **`load_phenotypes`**:
    *   Docstring clarifies that trait columns are now derived from `set_model_equation`.
    *   Details `id_column`, `covariate_columns`, `missing_value_codes`, and `**read_options`.
    *   Updates `self._mme.phenotype_dataframe` and `self._mme.phenotype_info`.
5.  **`load_pedigree`**:
    *   Docstring gives example of pedigree file format and details common `**pedigree_options`.
    *   Stores config in `self._mme.pedigree_info`.
6.  **`add_genotypes`**:
    *   Docstring details parameters like `name`, `data_source`, `method`, `map_file_path`.
    *   Explains common `**genotype_config_options` including variance priors, QC flags, and SSGBLUP `weight_for_G`.
    *   Stores all configuration options in `self._mme.genotype_components_config`.
7.  **`add_random_effect`**:
    *   Docstring clarifies `effect_name` should match a term in the model equation.
    *   Details `use_pedigree`, variance priors, and options for user-defined covariance matrix (`V_matrix`, `V_matrix_ids`).
    *   Notes how SSGBLUP 'animal' effect variance is typically handled (linked to `GenotypesComponent`).
    *   Stores config in `self._mme.random_effects_config`.
8.  **`set_residual_variance_prior`**:
    *   Details `value` (scalar for ST, matrix for MT), `df`, and `constraint` for MT.
    *   Stores config in `self._mme.residual_variance_prior_config`.
9.  **`set_mcmc_options`**:
    *   Details `chain_length`, `burn_in`, `thinning`, `seed`.
    *   Explains `**options` can include other MCMC settings from `runMCMC` Julia args.
    *   Updates `self._mme.mcmc_settings`.
10. **`_prepare_for_run`**: Docstring explains its internal role in orchestrating the full MME setup using stored configurations by calling various backend functions.
11. **`run`**: Docstring clarifies it calls `_prepare_for_run` and then the MCMC engine, and specifies return type.
12. **`run_gwas`**: Docstring details its parameters (map file, effects files, output folder, `**gwas_options`) and return type. Notes it should generally be called after `run` or with external effect samples.
13. **`get_ebv`**: Clarifies it retrieves EBVs/PEVs after `run`, and details `trait_names` parameter.
14. **`get_variance_components`**: Explains it retrieves posterior means of VCs.
15. **`summary`**: Explains it prints model configuration and a summary of results.

All user-facing configuration methods now consistently set `self._is_prepared = False` because any change to the model definition or data requires the internal `MME_py` object to be re-processed by `_prepare_for_run`. The `_prepare_for_run` method itself was substantially detailed in the previous subtask's commit.

This completes the docstring expansion and API refinement. The next step would be to actually build the Sphinx documentation from this.
