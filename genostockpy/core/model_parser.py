import re
from typing import List, Dict, Tuple, Any, Optional # Added Optional

class ParsedTerm:
    def __init__(self, term_str: str, term_type: str, trait_name: Optional[str] = None, factors: Optional[List[str]] = None):
        self.term_str = term_str          # Original term string, e.g., "age", "animal", "herd*parity"
        self.base_name = term_str # Usually same as term_str, unless it's an interaction, then it's the full "A*B"
        self.full_id = f"{trait_name}:{term_str}" if trait_name else term_str # e.g., "y1:age"
        self.term_type = term_type        # "intercept", "covariate", "factor", "random_factor", "interaction", "unknown_potential_random"
        self.trait_name = trait_name      # Trait this term instance belongs to
        self.factors = factors if factors else [term_str] # List of main effects involved, e.g., ["herd", "parity"]
        self.is_fixed = True             # Default, can be changed
        self.is_random = False
        self.is_covariate = (term_type == "covariate")

        # For MME construction (populated by mme_builder)
        self.n_levels: int = 0
        self.start_col_in_X: Optional[int] = None # Start column in X_effects_matrix (if applicable)
        self.columns_in_X: Optional[List[str]] = None # Actual column names in X (e.g. for dummy vars)

        # For global MME system (populated by mme_builder's effects_map logic)
        self.start_col_in_mme_system: Optional[int] = None
        self.num_cols_in_mme_system: Optional[int] = None


    def __repr__(self):
        return f"ParsedTerm({self.full_id}, type={self.term_type}, random={self.is_random})"


def parse_model_equation_py(equation_str: str,
                            data_columns: List[str],
                            fixed_effects_user: Optional[List[str]] = None,
                            random_effects_user: Optional[List[str]] = None,
                            covariate_columns_user: Optional[List[str]] = None
                           ) -> Tuple[List[str], List[ParsedTerm], Dict[str, ParsedTerm]]:
    """
    Parses R-like model equation string(s).

    Args:
        equation_str: e.g., "y ~ mu + age + sex + animal" or "t1~X1+X2; t2~X1+X3".
        data_columns: All available column names from the phenotype data (lowercase).
        fixed_effects_user: Optional list of terms explicitly declared fixed by user.
        random_effects_user: Optional list of terms explicitly declared random by user.
        covariate_columns_user: Optional list of terms explicitly declared covariates by user.

    Returns:
        Tuple containing:
            - List of response variable names (ordered as they appear).
            - List of ParsedTerm objects for all term instances (one per trait per term).
            - Dictionary mapping unique full_term_id ("trait:term_base") to ParsedTerm objects.
    """
    response_vars: List[str] = []
    all_terms_list: List[ParsedTerm] = []
    all_terms_map: Dict[str, ParsedTerm] = {}

    # Normalize data_columns for case-insensitive matching
    data_columns_lower = [col.lower() for col in data_columns]
    user_covariates_lower = [cov.lower() for cov in (covariate_columns_user or [])]
    user_randoms_lower = [rand.lower() for rand in (random_effects_user or [])]
    # user_fixed_lower = [fix.lower() for fix in (fixed_effects_user or [])] # Not used yet, fixed is default

    equations = [eq.strip() for eq in equation_str.replace(';', '\n').split('\n') if eq.strip()]
    if not equations:
        raise ValueError("Model equation string is empty or invalid.")

    for eq_part in equations:
        if '~' not in eq_part and '=' not in eq_part: # Allow both separators
            raise ValueError(f"Equation part '{eq_part}' must contain '~' or '=' to separate response and predictors.")

        separator = '~' if '~' in eq_part else '='
        lhs_str, rhs_str = eq_part.split(separator, 1)
        trait_name = lhs_str.strip()
        if not trait_name:
            raise ValueError(f"Response variable name is empty in equation part: '{eq_part}'")
        if trait_name not in response_vars: # Keep order of first appearance
            response_vars.append(trait_name)

        rhs_terms_str = [term.strip() for term in re.split(r'\s*\+\s*', rhs_str) if term.strip()]

        for term_s_orig in rhs_terms_str:
            term_s_lower = term_s_orig.lower()
            term_type = "unknown"
            is_interaction = '*' in term_s_orig

            # Factors in term are original case, base_name is also original case for map key
            factors_in_term = [t.strip() for t in term_s_orig.split('*')] if is_interaction else [term_s_orig]
            base_name_for_id = term_s_orig # Use original case for IDs and lookups initially

            if base_name_for_id == "1" or term_s_lower == "mu" or term_s_lower == "intercept":
                term_type = "intercept"
                base_name_for_id = "intercept" # Standardize base name
                factors_in_term = ["intercept"]
            elif is_interaction:
                term_type = "interaction"
                # Check if all parts of interaction are known covariates or factors
                all_factors_known = True
                for factor_part in factors_in_term:
                    factor_part_lower = factor_part.lower()
                    if not (factor_part_lower in user_covariates_lower or factor_part_lower in data_columns_lower):
                        # If a part of interaction is not in data/covariates, it could be problematic
                        # For now, we allow it, assuming it might be a term that gets special handling (e.g. random interaction)
                        # print(f"Warning: Factor '{factor_part}' in interaction '{term_s_orig}' not in data columns or covariate list.")
                        pass # all_factors_known = False; break
                # if not all_factors_known: term_type = "unknown_interaction"
            elif term_s_lower in user_covariates_lower:
                term_type = "covariate"
            elif term_s_lower in data_columns_lower: # In data, not specified as covariate -> factor
                term_type = "factor"
            elif term_s_lower in user_randoms_lower: # Explicitly random, not necessarily in data cols (e.g. 'animal')
                term_type = "random_factor" # Generic type, could be further specified (polygenic, IID) later
            else: # Not in data, not intercept, not explicitly random/covariate
                term_type = "unknown_potential_random"
                # print(f"Info: Term '{term_s_orig}' for trait '{trait_name}' not in data/covariates. Assumed potential random or needs definition.")

            parsed_term = ParsedTerm(term_str=base_name_for_id, term_type=term_type, trait_name=trait_name, factors=factors_in_term)

            if term_s_lower in user_randoms_lower:
                parsed_term.is_random = True
                parsed_term.is_fixed = False

            # If term_s_lower is in user_covariates_lower, is_covariate is already true via term_type
            if term_type == "covariate":
                parsed_term.is_covariate = True


            all_terms_list.append(parsed_term)
            # Store unique term definitions (trait-specific) in map
            if parsed_term.full_id not in all_terms_map:
                all_terms_map[parsed_term.full_id] = parsed_term
            else: # If it exists, update if current one is more specific (e.g. now random)
                if parsed_term.is_random and not all_terms_map[parsed_term.full_id].is_random:
                    all_terms_map[parsed_term.full_id].is_random = True
                    all_terms_map[parsed_term.full_id].is_fixed = False
                if parsed_term.is_covariate and not all_terms_map[parsed_term.full_id].is_covariate:
                     all_terms_map[parsed_term.full_id].is_covariate = True
                     all_terms_map[parsed_term.full_id].term_type = "covariate"


    # Ensure all_terms_list reflects the (potentially updated) unique definitions from all_terms_map
    # This is if a term "age" was first fixed, then later declared random for same trait (not typical)
    # For now, all_terms_list contains instances, all_terms_map contains unique definitions by full_id
    # The builder will primarily use all_terms_map for term properties.

    return response_vars, all_terms_list, all_terms_map

