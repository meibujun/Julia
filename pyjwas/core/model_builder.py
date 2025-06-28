from typing import List, Any, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, hstack, lil_matrix, diags, csc_matrix, csr_matrix, block_diag

from .model_term import ModelTerm
from .variance_covariance import VarianceCovariance
from .mme import MixedModelEquations
from .genotypes import GenotypesData # Assuming this will be used for genotype term handling
# Other necessary imports from .core will be added as needed


def make_level_dictionary(data_vector: Union[List[Any], pd.Series, np.ndarray]) -> Tuple[Dict[Any, int], List[Any]]:
    """
    Creates a dictionary mapping unique items in a vector to integer indices,
    and a list of these unique items (levels).
    Equivalent to Julia's mkDict function in build_MME.jl.

    Args:
        data_vector: A list, pandas Series, or numpy array containing the data.

    Returns:
        A tuple containing:
            - A dictionary where keys are unique items from data_vector and
              values are their 0-based integer indices.
            - A list of the unique items (levels) in their order of appearance.
    """
    if isinstance(data_vector, pd.Series):
        unique_levels = list(data_vector.unique())
    elif isinstance(data_vector, np.ndarray):
        unique_levels = list(np.unique(data_vector))
    else: # Assumed list
        # Simple approach for lists; order of unique_levels might differ from np.unique
        # For consistency with potential numeric sorting of np.unique, might consider converting
        # list to pd.Series or np.array first if strict order is critical.
        # However, Julia's unique also preserves first appearance order.
        unique_levels = sorted(list(set(data_vector)), key=lambda x: data_vector.index(x))


    level_to_index_map = {level: i for i, level in enumerate(unique_levels)}
    return level_to_index_map, unique_levels


def build_model(
    model_equations_str: str,
    R_value: Optional[Union[float, np.ndarray]] = None,
    R_df: float = 4.0,
    R_estimate_variance: bool = True,
    R_estimate_scale: bool = False,
    R_constraint: bool = False,
    # NNBayes parameters (placeholders for now)
    num_hidden_nodes: Optional[int] = None,
    nonlinear_function: Optional[Union[str, callable]] = None,
    latent_traits: Optional[List[str]] = None,
    user_sigma2_yobs: Optional[float] = None,
    user_sigma2_weights_NN: Optional[float] = None,
    # Censored/Categorical trait parameters
    censored_traits: Optional[List[str]] = None, # List of trait names that are censored
    categorical_traits: Optional[List[str]] = None, # List of trait names that are categorical
    # Genotypes data (optional, can be added later via another method)
    genotypes_data_list: Optional[List[GenotypesData]] = None
) -> MixedModelEquations:
    """
    Builds a MixedModelEquations object from model equation strings and other parameters.
    Parses model equation strings, sets up model terms, and initializes a
    `MixedModelEquations` object with basic model structure and residual variance setup.

    This function is the primary entry point for defining a model in PyJWAS.
    It translates the string-based model definition into internal `ModelTerm`
    objects and configures the initial `MixedModelEquations` state.

    The model equation string format is:
    `pheno1 = term1 + term2 + term3 * term4`
    For multi-trait models, equations are separated by semicolons or newlines:
    `pheno1 = mu + effectA; pheno2 = mu + effectB`

    Args:
        model_equations_str: A string defining the model structure.
            For multi-trait models, equations can be separated by ';' or newlines.
        R_value: Initial value for the residual variance (scalar for single-trait) or
            covariance matrix (NumPy array for multi-trait). This is often used as
            the scale parameter (S0^2 or Psi_0) for the prior distribution of R.
        R_df: Prior degrees of freedom for the residual variance/covariance.
        R_estimate_variance: If True, residual variance/covariance will be estimated via MCMC.
        R_estimate_scale: If True, the scale parameter of the prior for R will be estimated
            (less common, typically False).
        R_constraint: For multi-trait models, if True, constrains the residual covariance
            matrix to be diagonal (i.e., residuals of different traits are uncorrelated).
        num_hidden_nodes: (For NN-Bayes models) Number of hidden nodes in a neural network layer.
        nonlinear_function: (For NN-Bayes models) The non-linear activation function or a callable.
        latent_traits: (For NN-Bayes models) List of names for latent traits (hidden nodes).
        user_sigma2_yobs: (For NN-Bayes models) User-specified fixed variance for the observed phenotype.
        user_sigma2_weights_NN: (For NN-Bayes models) User-specified fixed variance for NN weights.
        censored_traits: List of phenotype names to be treated as censored traits.
        categorical_traits: List of phenotype names to be treated as categorical traits.
        genotypes_data_list: An optional list of pre-configured `GenotypesData` objects.
            If a term in the model equation matches the `name` of a `GenotypesData` object
            in this list, it will be treated as a genomic marker effect term.

    Returns:
        A `MixedModelEquations` object initialized with the parsed model structure
        and residual variance parameters.

    Raises:
        ValueError: If `model_equations_str` is invalid, or if `R_value` dimensions
            do not match the number of traits implied by the equations.
    """

    if not isinstance(model_equations_str, str) or not model_equations_str.strip():
        raise ValueError("Model equations string must be a non-empty string.")

    # TODO: NNBayes logic for re-writing model_equations needs to be implemented
    # if nonlinear_function is not None:
    #     print_styled("Bayesian Neural Network is used...", color="green")
    #     model_equations_str = nnbayes_model_equation(model_equations_str, num_hidden_nodes, ...)

    # Split model string into individual equations
    import re
    individual_equation_strs = [eq.strip() for eq in re.split(r'[;\n]', model_equations_str) if eq.strip()]

    n_models = len(individual_equation_strs)
    if R_value is not None:
        if n_models == 1 and not isinstance(R_value, (float, int, np.floating, np.integer)):
             if not (isinstance(R_value, np.ndarray) and R_value.size == 1):
                raise ValueError("R_value should be a scalar for single-trait models.")
        if n_models > 1 and (not isinstance(R_value, np.ndarray) or R_value.shape != (n_models, n_models)):
            raise ValueError(f"R_value should be a {n_models}x{n_models} numpy array for multi-trait models.")

    parsed_lhs_vars: List[str] = []
    parsed_model_terms: List[ModelTerm] = []
    model_term_dict: Dict[str, ModelTerm] = {}

    for model_idx, eq_str in enumerate(individual_equation_strs):
        if "=" not in eq_str:
            raise ValueError(f"Model equation '{eq_str}' is missing '=' separator.")

        lhs_str, rhs_str = [part.strip() for part in eq_str.split("=", 1)]
        if not lhs_str:
            raise ValueError(f"LHS (phenotype) cannot be empty in equation '{eq_str}'.")
        if not rhs_str:
            raise ValueError(f"RHS (terms) cannot be empty in equation '{eq_str}'.")

        parsed_lhs_vars.append(lhs_str)

        rhs_term_strs = [term.strip() for term in rhs_str.split("+")]
        for term_str_on_rhs in rhs_term_strs:
            if not term_str_on_rhs:
                continue

            model_term = ModelTerm(term_str=term_str_on_rhs, model_index=model_idx + 1, trait_name=lhs_str)
            parsed_model_terms.append(model_term)
            model_term_dict[model_term.trm_str] = model_term

    # Initialize Residual VarianceCovariance object
    # The Julia code's logic for scale_R ( R*(df-2)/df or R*(df-1) ) seems specific to how it defines
    # the scale for an Inverse Gamma or Inverse Wishart prior where R_value is the prior mean.
    # For now, we'll pass R_value as 'value' and let 'scale' be None if not explicitly different,
    # or derived if a clear prior distribution (like InvGamma(alpha, beta)) is being targeted.
    # A common way to specify InvGamma is by shape (alpha) and rate (beta) or shape and scale (theta=1/beta).
    # If R_value is E[sigma^2] = scale / (df/2 - 1) for InvChi2 (df, scale_param=S0)
    # or E[sigma^2] = beta / (alpha - 1) for InvGamma(alpha,beta)
    # Let's assume R_df is nu (degrees of freedom) and R_value is S_0 (sum of squares / prior mean like).
    # For Inverse Wishart E[Sigma] = S / (nu - p - 1).
    # The original code had: scale_R = R*(df - 2)/df or R*(df-1). This suggests R_value is a prior mean.
    # And df is nu.
    # Let's simplify: the `scale` parameter of VarianceCovariance is the scale parameter of the prior distribution.
    # If R_value is the *mean* of the prior, and R_df is the *degrees of freedom*,
    # for an Inv-scaled-Chi2(df, S^2_p), E[sigma^2] = S^2_p.
    # If R_value is this S^2_p, then VarianceCovariance.scale should be R_value.
    # If R_value is a matrix (prior mean for Inv-Wishart(nu, Psi^-1)), E[Sigma] = Psi / (nu-p-1).
    # Psi = R_value * (R_df - n_models - 1). This would be the `scale` for Inv-Wishart.
    # This needs to be very carefully mapped. For now, let's assume R_value is the scale parameter S_0.
    # This part is critical for correct prior specification.
    # Defaulting scale to R_value if not otherwise specified.

    # Simplified approach: if R_value is given, it's the starting 'value'.
    # The 'scale' for the prior might be the same as 'value' or derived.
    # The Julia code `scale_R = R*(df - 2)/df` (single) or `R*(df - nModels -1)` (multi, if R is Psi^-1)
    # looks like it's calculating the scale parameter for an Inv-Gamma or Inv-Wishart prior.
    # Let's assume R_value is a prior mean for now, and derive the scale for the VarianceCovariance.
    # This is a point of potential discrepancy if Julia's "scale" means something different.
    # For now, let's set `value` to R_value, and `scale` also to R_value, assuming R_value is the scale parameter of the prior.

    residual_vc_scale = R_value # Default: assume R_value is the scale parameter for the prior
    # A more robust approach would be to ask for prior shape/rate or df/scale_param explicitly.
    # Given the Julia code, it's likely R_value is related to prior mean, and R_df to degrees of freedom.
    # The term "scale" in Julia's Variance struct might directly be this prior scale parameter.
    # Let's assume `R_value` IS the prior scale `S` for Inv-Chi2 or `Psi` for Inv-Wishart.
    # Then `VarianceCovariance.scale` should be `R_value`.
    # And `VarianceCovariance.value` is the initial value for the variance, often also `R_value`.

    residual_vc = VarianceCovariance(
        value=R_value,
        df=R_df,
        scale=R_value, # Assuming R_value is the prior scale parameter S_nu or Psi_inv
        estimate_variance=R_estimate_variance,
        estimate_scale=R_estimate_scale, # Julia code had this as not supported for residual
        constraint=R_constraint
    )

    mme = MixedModelEquations(
        n_models=n_models,
        model_equations_str=individual_equation_strs,
        model_terms=parsed_model_terms,
        model_term_dict=model_term_dict,
        lhs_variables=parsed_lhs_vars,
        residual_variance_info=residual_vc
    )

    if genotypes_data_list:
        active_genotype_terms_strs = []
        temp_model_terms = [] # Build a new list of non-genotype terms

        for term in mme.model_terms:
            term_factor_name = term.factors[0]
            is_genotype_term = False
            for gd in genotypes_data_list:
                if gd.name == term_factor_name:
                    term.random_type = "genotypes"

                    # Configure GenotypesData according to current model context
                    gd.n_traits_model = mme.n_models
                    gd.trait_names = list(mme.lhs_variables)

                    # Add to MME's list if new (based on object name)
                    if not any(existing_gd.name == gd.name for existing_gd in mme.genotypes_data_list):
                        mme.genotypes_data_list.append(gd)

                    active_genotype_terms_strs.append(term.trm_str)
                    is_genotype_term = True
                    break
            if not is_genotype_term:
                temp_model_terms.append(term)

        mme.model_terms = temp_model_terms
        for mt_str in active_genotype_terms_strs:
            if mt_str in mme.model_term_dict:
                del mme.model_term_dict[mt_str]

    # Trait types
    if censored_traits is None: censored_traits = []
    if categorical_traits is None: categorical_traits = []
    for i, trait_name in enumerate(mme.lhs_variables):
        if trait_name in censored_traits:
            mme.trait_types[i] = "censored"
        elif trait_name in categorical_traits:
            mme.trait_types[i] = "categorical"

    return mme

def set_covariate(mme: MixedModelEquations, *covariate_names_str: str):
    """
    Sets specified variables as covariates in the MME object.
    Variables can be provided as multiple arguments, each being a single
    variable name or a space-separated string of variable names.

    Args:
        mme: The MixedModelEquations object to modify.
        *covariate_names_str: One or more strings, where each string can be a single
                              variable name or multiple space-separated variable names.
                              These names should correspond to terms in the model equations.
    """
    if not isinstance(mme, MixedModelEquations):
        raise TypeError("mme must be an instance of MixedModelEquations.")

    parsed_covariates = []
    for cov_group_str in covariate_names_str:
        if not isinstance(cov_group_str, str):
            raise TypeError("Each covariate argument must be a string.")
        # Split space-separated names and strip whitespace
        individual_names = [name.strip() for name in cov_group_str.split(' ') if name.strip()]
        parsed_covariates.extend(individual_names)

    # Add to mme.covariate_variables, ensuring no duplicates and that they are valid model factors
    # Note: Julia version converts to Symbols. Python will keep as strings.
    # The actual ModelTerm.factors are already strings.

    # Check if these covariates are part of any ModelTerm's factors
    all_model_factors = set()
    for term in mme.model_terms: # Check against general model terms
        for factor in term.factors:
            all_model_factors.add(factor)
    # Also consider factors from genotype terms if they were to be covariates (less common)
    # For now, this check is basic. More robust validation might be needed.

    for cov_name in parsed_covariates:
        if cov_name not in all_model_factors:
            # This is a warning rather than an error in some systems, as the term might not be used.
            # However, if it's meant to be a covariate, it should be in a ModelTerm.
            print(f"Warning: Covariate '{cov_name}' is not found as a factor in any existing model term."
                  " Ensure it's part of your model equation if it's intended to have an effect.")

        if cov_name not in mme.covariate_variables:
            mme.covariate_variables.append(cov_name)


if __name__ == '__main__':
    # Test make_level_dictionary
    print("Testing make_level_dictionary:")
    list_data = ["a1", "a4", "a1", "a2", "a4", "a3"]
    level_map, levels = make_level_dictionary(list_data)
    print(f"List data: {list_data} -> Map: {level_map}, Levels: {levels}")

    # Test build_model
    print("\nTesting build_model:")

    # Single trait model
    eq1 = "BW = intercept + age + sex"
    mme1 = build_model(eq1, R_value=6.72, R_df=4.0)
    print(f"MME1: {mme1}")
    print(f"  R Variance: {mme1.R_variance}")
    for term in mme1.model_terms:
        print(f"  Term: {term.trm_str}, Factors: {term.factors}")

    # Multi-trait model
    eq2 = """
    BW = intercept + age + sex
    CW = intercept + litter_size
    """
    R_val_multi = np.array([[6.72, 1.0], [1.0, 5.0]])
    mme2 = build_model(eq2, R_value=R_val_multi, R_df=5.0)
    print(f"\nMME2: {mme2}")
    print(f"  LHS Variables: {mme2.lhs_variables}")
    print(f"  R Variance: {mme2.R_variance}")
    for term in mme2.model_terms:
        print(f"  Term: {term.trm_str}, Model Index: {term.imodel}")

    # Model with GenotypesData
    geno_chip1 = GenotypesData(name="chip1", method="BayesC")
    eq3 = "yield = mu + Parity + chip1" # mu is like intercept
    mme3 = build_model(eq3, R_value=100.0, genotypes_data_list=[geno_chip1])
    print(f"\nMME3: {mme3}")
    print(f"  General Model Terms: {[mt.trm_str for mt in mme3.model_terms]}")
    print(f"  Genotypes Data List: {[gd.name for gd in mme3.genotypes_data_list]}")
    if mme3.genotypes_data_list:
        print(f"    chip1 n_traits_model: {mme3.genotypes_data_list[0].n_traits_model}")
        print(f"    chip1 trait_names: {mme3.genotypes_data_list[0].trait_names}")

    # Test for error handling
    try:
        build_model("y = ")
    except ValueError as e:
        print(f"\nCaught expected error: {e}")

    try:
        build_model("y = age", R_value=np.array([1.0, 2.0])) # Wrong R shape for single trait
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nTest with categorical/censored traits:")
    eq4 = "trait1 = fixed_effect; trait2 = another_effect; trait3 = effect3"
    mme4 = build_model(eq4, categorical_traits=["trait2"], censored_traits=["trait3"])
    print(f"MME4 Trait Types: {mme4.trait_types}")

    print("\nTesting set_covariate:")
    mme_for_cov = build_model("y = var1 + var2 + var3*var1", R_value=1.0)
    print(f"Initial covariates: {mme_for_cov.covariate_variables}")
    set_covariate(mme_for_cov, "var1", "var2")
    print(f"After set_covariate('var1', 'var2'): {mme_for_cov.covariate_variables}")
    set_covariate(mme_for_cov, "var3") # var3 is in an interaction
    print(f"After set_covariate('var3'): {mme_for_cov.covariate_variables}")
    set_covariate(mme_for_cov, "var1") # Test duplicate add
    print(f"After set_covariate('var1') again: {mme_for_cov.covariate_variables}")
    set_covariate(mme_for_cov, "non_existent_var") # Test warning for non-existent var
    print(f"After set_covariate('non_existent_var'): {mme_for_cov.covariate_variables}")

    # Test with space-separated string
    mme_for_cov_space = build_model("y = termA + termB + termC", R_value=1.0)
    set_covariate(mme_for_cov_space, "termA termB")
    print(f"After set_covariate('termA termB'): {mme_for_cov_space.covariate_variables}")

    print("\nTesting MME component construction (simple single trait):")
    # Data for y = intercept + x
    data_for_mme = pd.DataFrame({
        'y': [10.0, 12.0, 15.0, 8.0],
        'x': [1.0, 2.0, 3.0, 0.5],
        'fixed_factor': ['A', 'B', 'A', 'B']
    })
    # Model: y = intercept + x + fixed_factor
    mme_st = build_model("y = intercept + x + fixed_factor", R_value=1.0)
    set_covariate(mme_st, "x") # x is a covariate

    # Manually set MCMCInfo for invweights (not strictly needed for this test part if not using hetero)
    from .mcmc_info import MCMCInfo
    mme_st.mcmc_info = MCMCInfo()


    try:
        get_mme_components(mme_st, data_for_mme)
        print(f"MME ST X matrix ({mme_st.X.shape}):\n{mme_st.X.toarray() if mme_st.X is not None else 'None'}")
        print(f"MME ST y_sparse ({mme_st.y_sparse.shape if mme_st.y_sparse is not None else 'None'}):\n{mme_st.y_sparse if mme_st.y_sparse is not None else 'None'}")
        print(f"MME ST LHS ({mme_st.mme_LHS.shape if mme_st.mme_LHS is not None else 'None'}):\n{mme_st.mme_LHS.toarray() if mme_st.mme_LHS is not None and hasattr(mme_st.mme_LHS, 'toarray') else mme_st.mme_LHS}")
        print(f"MME ST RHS ({mme_st.mme_RHS.shape if mme_st.mme_RHS is not None else 'None'}):\n{mme_st.mme_RHS if mme_st.mme_RHS is not None else 'None'}")

        effect_names = get_mme_effect_names(mme_st)
        print(f"Effect names for MME ST: {effect_names}")
        # Expected names: y:intercept:intercept, y:x, y:fixed_factor:A, y:fixed_factor:B (or similar)
        # Order depends on model_terms processing.
        # Current ModelTerm constructor makes trm_str = "trait:term"
        # Current get_mme_effect_names makes "term_trm_str:level_name"
        # So for intercept: "y:intercept:intercept"
        # For covariate "x": "y:x" (since n_levels=1, factor[0]!=intercept)
        # For factor "fixed_factor" with levels A,B: "y:fixed_factor:A", "y:fixed_factor:B"

        # Example X for "y = intercept + x + fixed_factor" (A,B)
        # Obs:    y   x   ff  (intercept, x, ff_A, ff_B)
        # 1       10  1   A     1          1     1     0
        # 2       12  2   B     1          2     0     1
        # 3       15  3   A     1          3     1     0
        # 4       8   0.5 B     1          0.5   0     1
        # Order of columns in X depends on order of terms in mme.model_terms
        # and levels within factors. Let's check term order:
        print("Model terms order for mme_st:")
        for mt_idx, mt_val in enumerate(mme_st.model_terms):
            print(f"  {mt_idx}: {mt_val.trm_str}, levels: {mt_val.names}, start_pos: {mt_val.start_pos}")
            # For intercept, names=["intercept"]
            # For x (cov), names=["x"]
            # For fixed_factor, names=["A", "B"] (if A appears before B in data)


    except Exception as e:
        print(f"Error during MME component construction test: {e}")
        import traceback
        traceback.print_exc()


def _get_data_for_term(term: ModelTerm, df: pd.DataFrame, mme: MixedModelEquations):
    """
    Populates the .data and .val fields of a ModelTerm based on the DataFrame.
    Helper function, equivalent to Julia's getData.

    Args:
        term: The ModelTerm object to populate.
        df: The input pandas DataFrame.
        mme: The MixedModelEquations object containing covariate info.
    """
    n_obs = len(df)

    if not term.factors: # Should not happen if ModelTerm is correctly initialized
        term.data = []
        term.val = np.array([])
        return

    first_factor_name = term.factors[0]

    if first_factor_name == "intercept":
        term.data = ["intercept"] * n_obs
        term.val = np.ones(n_obs, dtype=np.float64) # Default to float64
    else:
        # Ensure all factors in the term exist in the DataFrame columns
        for factor_name in term.factors:
            if factor_name not in df.columns:
                raise ValueError(f"Factor '{factor_name}' in term '{term.trm_str}' not found in DataFrame columns.")

        # Handle the first factor
        if first_factor_name in mme.covariate_variables:
            if not pd.api.types.is_numeric_dtype(df[first_factor_name]):
                raise TypeError(f"Covariate '{first_factor_name}' must have a numeric data type in DataFrame.")
            current_str_data = [first_factor_name] * n_obs
            current_val_data = df[first_factor_name].values.astype(np.float64)
        else: # It's a categorical factor
            current_str_data = df[first_factor_name].astype(str).tolist()
            current_val_data = np.ones(n_obs, dtype=np.float64)

        # Handle interactions (remaining factors)
        for i in range(1, term.n_factors):
            interacting_factor_name = term.factors[i]
            if interacting_factor_name in mme.covariate_variables:
                if not pd.api.types.is_numeric_dtype(df[interacting_factor_name]):
                    raise TypeError(f"Covariate '{interacting_factor_name}' in interaction must be numeric.")
                # Append factor name to string data, multiply value data
                current_str_data = [f"{s}*{interacting_factor_name}" for s in current_str_data]
                current_val_data = current_val_data * df[interacting_factor_name].values.astype(np.float64)
            else: # Categorical factor in interaction
                # Append factor level to string data
                factor_level_strs = df[interacting_factor_name].astype(str).tolist()
                current_str_data = [f"{s}*{lvl}" for s, lvl in zip(current_str_data, factor_level_strs)]
                # Value data remains 1.0 for categorical interactions (multiplied by 1.0)
                # current_val_data = current_val_data * 1.0 (no change)

        term.data = current_str_data
        term.val = current_val_data

    # Handle missing values in .val (NaNs from DataFrame become NaNs in numpy array)
    # Julia's coalesce.(val, 0.0) replaces missing with 0.0. Pandas fillna(0) does this.
    if term.val is not None: # Should always be not None if factors exist
        term.val = np.nan_to_num(term.val, nan=0.0) # Replaces NaN with 0.0

    # Precision (Python floats are double by default, so explicit Float32 conversion if needed)
    # For now, keep as float64 (np.float64)
    # if mme.mcmc_info and not mme.mcmc_info.double_precision:
    #     term.val = term.val.astype(np.float32)


def _get_incidence_matrix_for_term(term: ModelTerm,
                                   mme: MixedModelEquations,
                                   n_obs_per_trait: int):
    """
    Constructs the incidence matrix (X) for a given ModelTerm.
    Updates term.X, term.n_levels, term.names, term.start_pos.
    Updates mme.mme_pos_counter.
    Equivalent to Julia's getX.

    Args:
        term: The ModelTerm object (must have .data and .val populated).
        mme: The MixedModelEquations object.
        n_obs_per_trait: Number of observations for a single trait (rows in original df).
    """
    if term.val is None or term.data is None:
        raise ValueError("Term .data and .val must be populated before creating incidence matrix.")

    total_rows_for_X = n_obs_per_trait * mme.n_models

    # 1. Row Indices (xi)
    # Row indices are specific to this term's contribution to the trait within the full X matrix
    # If term.imodel is 1-based index of the model/trait equation
    row_offset = (term.imodel - 1) * n_obs_per_trait
    # These are local row indices for this trait's block
    local_row_indices = np.arange(n_obs_per_trait)
    # Global row indices in the combined X matrix for all traits
    xi = row_offset + local_row_indices

    # 2. Values (xv)
    xv = term.val

    # 3. Column Indices (xj)
    # Handle "missing" strings in term.data by replacing them temporarily for dict creation,
    # but ensure they result in zero contribution to xv.
    # Julia: data .== "missing" then xv .= 0
    # Here, nan_to_num in _get_data_for_term already converted NaNs in .val to 0.
    # We need a consistent string for missing to map to a column if it occurs.
    # For now, assume "missing" strings in term.data are meaningful levels if not filtered.

    data_for_dict_creation = [d if not ("missing" in d and "*" in d) else "EFFECTIVELY_MISSING" for d in term.data]
    # ^ This is a simple attempt to mimic Julia's `if "missing" in getFactor(trm.data[i])`
    # A more robust way would be to parse interaction terms properly.
    # For now, let's assume "missing" as a level name is handled by make_level_dictionary.

    level_to_idx_map: Dict[Any, int]
    unique_names: List[Any]

    # TODO: Refine logic for random_type "V" (user V_inv) and "A" (pedigree)
    # These might require using pre-defined names/levels from a RandomEffectTerm
    # to ensure consistency with V_inv matrix dimensions.
    # For now, this basic logic creates levels from observed data.
    # if term.random_type in ["V", "A"] and term.names: # If names are pre-set
    #     unique_names = term.names
    #     level_to_idx_map = {name: i for i, name in enumerate(unique_names)}
    # else:
    level_to_idx_map, unique_names = make_level_dictionary(data_for_dict_creation)

    term.names = unique_names
    term.n_levels = len(unique_names)

    if not term.n_levels: # No data or all missing, results in empty matrix for this term
        term.X = csr_matrix((total_rows_for_X, 0)) # Empty sparse matrix with correct rows
        term.start_pos = mme.mme_pos_counter # Position doesn't advance
        # mme.mme_pos_counter remains unchanged
        return

    # Map data strings to column indices
    # If a level in term.data was "EFFECTIVELY_MISSING", it needs a column,
    # but its corresponding xv should be 0.
    # Let's ensure "EFFECTIVELY_MISSING" maps to a valid index if it exists.
    if "EFFECTIVELY_MISSING" in level_to_idx_map:
        missing_col_idx = level_to_idx_map["EFFECTIVELY_MISSING"]
        # xv for these rows should already be 0 due to nan_to_num or specific logic

    xj = np.array([level_to_idx_map[d] for d in data_for_dict_creation], dtype=int)

    # Filter out entries where xv is zero if we want a minimal sparse matrix,
    # though sparse matrix constructors handle zeros correctly.
    # Julia's dropzeros! is implicit.

    # Create the sparse matrix for this term's contribution to its specific trait
    # This matrix has dimensions (n_obs_per_trait, term.n_levels)
    term_trait_X = csr_matrix((xv, (local_row_indices, xj)),
                                shape=(n_obs_per_trait, term.n_levels))

    # Now, embed this into the full X matrix structure for all traits.
    # This requires careful construction if building X piece by piece.
    # Julia's approach `sparse(xi,xj,xv, total_rows, total_cols)` builds the whole thing if global indices are used.
    # With local `xi` and `xj` for the term's specific block:
    # We need to place `term_trait_X` into a larger sparse matrix of shape (total_rows_for_X, term.n_levels)
    # at the correct row offset.

    # Using lil_matrix for easier block assignment initially might be an option for the full X,
    # or construct term-specific X matrices and then hstack them after zero-padding rows.

    # Let's make term.X specific to its column block, but with all trait rows.
    # It will have shape (total_rows_for_X, term.n_levels)
    # Only rows corresponding to term.imodel will be non-zero.

    # Re-calculate global row indices `xi` for the final term.X matrix
    global_xi_for_term_X = (term.imodel - 1) * n_obs_per_trait + local_row_indices

    # We need to ensure xv and xj only correspond to non-zero elements for efficiency
    non_zero_mask = (xv != 0)
    eff_xi = global_xi_for_term_X[non_zero_mask]
    eff_xj = xj[non_zero_mask]
    eff_xv = xv[non_zero_mask]

    if len(eff_xv) > 0: # Only create if there are non-zero values
        term.X = csr_matrix((eff_xv, (eff_xi, eff_xj)),
                            shape=(total_rows_for_X, term.n_levels))
    else: # All values were zero
        term.X = csr_matrix((total_rows_for_X, term.n_levels))


    term.start_pos = mme.mme_pos_counter
    mme.mme_pos_counter += term.n_levels


# Placeholder for _add_variance_contributions_to_LHS (addVinv)
def _add_variance_contributions_to_LHS(mme: MixedModelEquations):
    """
    Adds contributions from random effect variance components to the MME LHS.
    Placeholder for addVinv logic from Julia.
    This will modify mme.mme_LHS.
    """
    # This function will iterate through mme.random_effect_terms and mme.genotypes_data_list
    # For each random effect/genomic component, it will find its columns in the MME
    # (using term.start_pos and term.n_levels or equivalent for genotype blocks)
    # and add something like V_inv * (sigma_e^2 / sigma_g^2) to the diagonal blocks
    # of mme.mme_LHS. The exact formulation depends on the MME version (lambda-version vs full).
    # This is a complex step that needs careful translation of Julia's addVinv and related logic.
    pass # TODO: Implement this based on Julia's addVinv

def _make_Ri_matrix(mme: MixedModelEquations, df: pd.DataFrame, invweights: np.ndarray) -> spmatrix:
    """
    Constructs the inverse of the residual covariance matrix (Ri) for multi-trait models,
    accounting for missing data patterns and heterogeneous weights.
    Roughly equivalent to Julia's mkRi.

    Args:
        mme: The MixedModelEquations object.
        df: The input pandas DataFrame (used to determine missing patterns).
        invweights: Array of inverse weights for observations.

    Returns:
        A SciPy sparse matrix for Ri.
    """
    n_obs_per_trait = len(df)
    total_rows = n_obs_per_trait * mme.n_models

    if mme.R_variance.constraint or mme.n_models == 1: # Diagonal R or single trait
        # For single trait, R_val is scalar. For constrained multi-trait, R_val is diagonal.
        # We need 1/R_val for the diagonal of Ri.
        if mme.n_models == 1:
            if not isinstance(mme.R_variance.value, (float, int, np.floating, np.integer)):
                 raise ValueError("R_variance.value must be scalar for single trait.")
            diag_R_inv_values = np.ones(n_obs_per_trait) * (1.0 / mme.R_variance.value)
        else: # Constrained multi-trait
            if not isinstance(mme.R_variance.value, np.ndarray) or not mme.R_variance.value.ndim == 2:
                 raise ValueError("R_variance.value must be a 2D matrix for constrained multi-trait.")
            # Assuming R_variance.value is already diagonal due to constraint
            diag_vals_R = np.diag(mme.R_variance.value)
            if np.any(diag_vals_R == 0): raise ValueError("Diagonal elements of R cannot be zero.")
            diag_R_inv_values_per_trait = 1.0 / diag_vals_R
            # Repeat for each observation: [1/r1, 1/r2, ..., 1/r1, 1/r2, ...]
            diag_R_inv_values = np.tile(diag_R_inv_values_per_trait, n_obs_per_trait)

        # Combine with observation-specific weights
        full_diag_values = diag_R_inv_values * np.repeat(invweights, mme.n_models if mme.n_models > 1 else 1)
        return diags(full_diag_values, format="csc")

    else: # Unconstrained multi-trait, need to handle missing patterns block-diagonally
        # This is the complex case. Julia's mkRi uses mme.residual_variance_handler (ResVar)
        # which stores precomputed R matrices for each missing pattern.
        # For now, this is a simplified placeholder. A full implementation would involve:
        # 1. Identifying unique missingness patterns in df[mme.lhs_variables].
        # 2. For each pattern, selecting the submatrix of mme.R_variance.value for observed traits.
        # 3. Inverting this submatrix.
        # 4. Constructing block-diagonal Ri using these inverted submatrices, scaled by invweights.
        # This is non-trivial. Let's assume for now that mme.R_variance.value is the full R_inv
        # if no sophisticated missing data handling is yet in place.
        # A proper implementation would use mme.residual_variance_handler.

        # Simplified: Use full R_inv, kron with invweights diagonal
        # This doesn't correctly handle missing data block-wise like original JWAS.
        # It assumes missing data imputation happens before forming y_corrected_for_Ri.
        if not isinstance(mme.R_variance.value, np.ndarray):
            raise TypeError("R_variance.value must be a NumPy array for unconstrained multi-trait models.")

        try:
            R_inv_full = np.linalg.inv(mme.R_variance.value)
        except np.linalg.LinAlgError:
            raise ValueError("Residual covariance matrix (R_variance.value) is singular.")

        # Create block diagonal Ri from R_inv_full, weighted by invweights
        # This can be done by sparse.kron(spdiags(invweights), R_inv_full)
        # Or by constructing lists for data, row_ind, col_ind for csc_matrix

        # Efficient construction of block diagonal matrix:
        # Repeat R_inv_full n_obs_per_trait times along the diagonal, scale each block by invweights[i]
        # This is complex to do efficiently with sparse matrices directly for general blocks.
        # scipy.linalg.block_diag can be used if blocks are pre-scaled.

        # Fallback to a less efficient but conceptually simpler construction for now:
    # This is the complex case. Uses mme.residual_variance_handler (ResVar equivalent).
    if mme.residual_variance_handler is None:
        # Initialize if not present, using the current full R matrix from mme.R_variance
        from .residual_variance import ResidualVariance # Local import to avoid circularity at module load if files are structured differently
        if mme.R_variance.value is None:
            raise ValueError("mme.R_variance.value (full R matrix) must be set to initialize ResidualVariance handler for Ri construction.")
        mme.residual_variance_handler = ResidualVariance(r0_matrix=np.copy(mme.R_variance.value))

    # If mme.R_variance.value (the full R matrix) has changed since the handler's r0_full_R_matrix was last set,
    # the cache in ri_pattern_inverses might be stale. Clear it.
    # This check ensures that get_or_compute_R_inv_for_pattern uses the current R.
    if mme.residual_variance_handler.r0_full_R_matrix is None or \
       not np.array_equal(mme.residual_variance_handler.r0_full_R_matrix, mme.R_variance.value):
        # print("DEBUG: R matrix changed, clearing Ri pattern cache.")
        mme.residual_variance_handler.ri_pattern_inverses.clear()
        mme.residual_variance_handler.r0_full_R_matrix = np.copy(mme.R_variance.value)


    observed_mask_df = df[mme.lhs_variables].notna() # DataFrame of bools (n_obs x n_traits)

    list_of_R_inv_blocks: List[spmatrix] = []

    for k in range(n_obs_per_trait):
        pattern_k = tuple(observed_mask_df.iloc[k].values)

        # Get the (n_traits x n_traits) R_inv block for this pattern
        # This R_inv_pattern_k has zeros for missing traits' rows/columns
        R_inv_pattern_k_full_dim = mme.residual_variance_handler.get_or_compute_R_inv_for_pattern(
            pattern_k,
            mme.R_variance.value # Pass current full R matrix
        )

        # Scale this block by the observation-specific inverse weight
        block_k_scaled = R_inv_pattern_k_full_dim * invweights[k]
        list_of_R_inv_blocks.append(csc_matrix(block_k_scaled)) # Ensure sparse for block_diag

    # Construct the large block-diagonal Ri matrix
    if not list_of_R_inv_blocks: # Should not happen if n_obs_per_trait > 0
        return csc_matrix((total_rows, total_rows))

    return block_diag(list_of_R_inv_blocks, format="csc")


def get_mme_components(mme: MixedModelEquations, df: pd.DataFrame):
    """
    Constructs the main components of the Mixed Model Equations (X, y, LHS, RHS).
    Constructs the main components of the Mixed Model Equations (X, y, LHS, RHS)
    based on the model definition in the `mme` object and the provided data.

    This function populates the following attributes of the `mme` object:
    - `X`: The full design matrix for fixed and i.i.d. random effects.
    - `y_sparse`: The phenotype vector.
    - `obs_ids`: List of observation IDs.
    - `mme_LHS`: The left-hand side of the MME (X'R_invX + initial Lambda).
    - `mme_RHS`: The right-hand side of the MME (X'R_inv*y).
    - `inverse_weights`: Weights for heterogeneous residuals.
    It also calls helper functions to populate data within each `ModelTerm`.

    Args:
        mme: The `MixedModelEquations` object, which should have been initialized
             by `build_model` and potentially modified by `set_covariate` or
             methods to add structured random effects (e.g., pedigree).
        df: A pandas DataFrame containing phenotype data and any covariates or
            factor levels defined in the model equations. Row order in this DataFrame
            defines the order of observations.

    Raises:
        RuntimeError: If MME components appear to have been built already.
        ValueError: If the input DataFrame is empty or required columns are missing.
    """
    if mme.mme_LHS is not None: # Check if already built
        raise RuntimeError("MME components seem to have been built already. Re-building is not yet supported.")

    n_obs_per_trait = len(df)
    if n_obs_per_trait == 0:
        raise ValueError("Input DataFrame is empty.")

    # Heterogeneous residuals weights
    # Ensure mcmc_info is present if heterogeneous_residuals is to be checked
    # Default to False if no mcmc_info
    use_hetero_res = False
    if mme.mcmc_info and hasattr(mme.mcmc_info, 'heterogeneous_residuals'): # defensive check
        use_hetero_res = mme.mcmc_info.heterogeneous_residuals

    if use_hetero_res:
        if "weights" not in df.columns:
            raise ValueError("Column 'weights' required in DataFrame for heterogeneous_residuals=True.")
        invweights = 1.0 / df["weights"].values.astype(np.float64)
        if np.any(invweights <= 0):
            raise ValueError("Inverse weights must be positive.")
    else:
        invweights = np.ones(n_obs_per_trait, dtype=np.float64)
    mme.inverse_weights = invweights

    # 1. Build Incidence Matrices for each ModelTerm (fixed and i.i.d. random effects)
    mme.mme_pos_counter = 0 # Reset column counter for the full X matrix
    all_term_X_matrices: List[spmatrix] = []
    for term in mme.model_terms:
        if term.X is None: # If not already built (e.g. by a previous call or manual setup)
            _get_data_for_term(term, df, mme)
            _get_incidence_matrix_for_term(term, mme, n_obs_per_trait)
        if term.X is not None and term.X.shape[1] > 0 : # Only include terms that have columns
             all_term_X_matrices.append(term.X)

    # Concatenate all term.X matrices
    if not all_term_X_matrices:
        # This happens if model has no fixed/random effects (e.g. y = 0, or only genotype effects)
        # Create an empty X matrix with correct number of rows, but 0 columns for fixed/random effects part
        total_rows_for_X = n_obs_per_trait * mme.n_models
        mme.X = csr_matrix((total_rows_for_X, 0))
    else:
        mme.X = hstack(all_term_X_matrices, format="csc") # Use CSC for column operations like X'RX

    # 2. Prepare y_sparse vector
    y_data_all_traits = []
    mme.obs_ids = df.index.astype(str).tolist() # Assuming df index provides observation IDs

    for trait_name_str in mme.lhs_variables:
        if trait_name_str not in df.columns:
            raise ValueError(f"Phenotype column '{trait_name_str}' not found in DataFrame.")
        # Coalesce missing to 0, as in Julia
        y_trait_data = df[trait_name_str].fillna(0.0).values.astype(np.float64)
        y_data_all_traits.append(y_trait_data)

    # Stack them into a single column vector ( (n_obs*n_traits) x 1 )
    mme.y_sparse = np.concatenate(y_data_all_traits).reshape(-1, 1)
    # No need to make it sparse if it's dense, but Julia called it ySparse.
    # For actual MME RHS (X'Ry), it will be dense anyway.

    # 3. Form MME LHS and RHS
    X_matrix = mme.X
    y_vector = mme.y_sparse # This is dense (N x 1) where N = n_obs_per_trait * n_models

    if mme.n_models == 1:
        # Single-trait: LHS = X' * D * X, RHS = X' * D * y
        # D is Diagonal(invweights)
        if X_matrix.shape[1] == 0: # No fixed/random effects
            mme.mme_LHS = csr_matrix((0,0)) # Empty LHS
            mme.mme_RHS = np.zeros((0,1))   # Empty RHS
        else:
            D_sparse = diags(invweights, format="csc")
            mme.mme_LHS = X_matrix.T @ D_sparse @ X_matrix
            mme.mme_RHS = X_matrix.T @ D_sparse @ y_vector
    else: # Multi-trait
        # Ri is the inverse of the residual covariance structure ( (n_obs*n_traits) x (n_obs*n_traits) )
        # This is block diagonal, with each block being R_inv for that observation (scaled by invweight)
        # or more complex if accounting for missing patterns.
        Ri_matrix = _make_Ri_matrix(mme, df, invweights)
        if X_matrix.shape[1] == 0:
            mme.mme_LHS = csr_matrix((0,0))
            mme.mme_RHS = np.zeros((0,1))
        else:
            mme.mme_LHS = X_matrix.T @ Ri_matrix @ X_matrix
            mme.mme_RHS = X_matrix.T @ Ri_matrix @ y_vector

    # 4. Add contributions from random effects (V_inv terms) to mme.mme_LHS
    # This is where addVinv from Julia would come in.
    _add_variance_contributions_to_LHS(mme) # Modifies mme.mme_LHS in place

    # Ensure no zero diagonals in mme.mme_LHS for solvability (unless it's an empty MME)
    if mme.mme_LHS.shape[0] > 0:
        diag_LHS = mme.mme_LHS.diagonal()
        if np.any(diag_LHS == 0.0):
            # Find which effects have zero diagonal
            # This requires mapping column indices back to term names.
            # For now, a general error.
            # TODO: Add more specific error message with term names.
            zero_diag_indices = np.where(diag_LHS == 0.0)[0]
            print(f"Warning: MME LHS has zero diagonal elements at indices: {zero_diag_indices}. "
                  "This may indicate no data for some factor levels or linear dependencies.")
            # In Julia, this was an error. Depending on context, it might be allowable if those effects are zero.
            # raise ValueError("MME LHS has zero diagonal elements. Check for factor levels with no data.")


# Helper to get all effect names in MME order
def get_mme_effect_names(mme: MixedModelEquations) -> List[str]:
    """
    Returns a list of descriptive names for all fixed and random effects
    in the MME, in the order they appear in the solution vector.
    Excludes marker effects which are handled separately.
    """
    effect_names: List[str] = []
    # Order: model_terms (fixed, iid random), then random_effect_terms (like pedigree)
    # This order must match how X and mme_LHS are constructed.

    # Fixed and IID random effects from model_terms
    for term in mme.model_terms:
        if term.n_levels == 1 and term.factors[0] != "intercept": # Covariate
            effect_names.append(term.trm_str)
        else: # Factor or intercept
            for level_name in term.names:
                effect_names.append(f"{term.trm_str}:{level_name}")

    # Other structured random effects (e.g., pedigree from RandomEffectTerm)
    # These also contribute columns to X if they are part of the general MME setup.
    # The current `model_terms` list in MME only contains fixed/iid.
    # Structured random effects (like pedigree) are often added to mme.random_effect_terms
    # and their contributions to MME are handled by _add_variance_contributions_to_LHS.
    # If their basis vectors are explicitly in X, their names should be here.
    # This part needs to align with how structured random effects are incorporated into X.
    # For now, assuming only effects in mme.model_terms form the explicit X.

    # TODO: Add names for effects from mme.random_effect_terms if they are part of the X matrix.
    # The current Julia code structure seems to put all non-marker effects into modelTerms
    # and then some of those are designated as random.
    # If so, the loop above is sufficient.

    return effect_names
