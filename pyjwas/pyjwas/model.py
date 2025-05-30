"""
Defines the structure for specifying statistical models, including parsing
model equations and managing fixed and random effects.
"""
from .parser import parse_model_equation

class ModelDefinition:
    """
    Represents a statistical model definition, derived from a model equation string.

    Attributes:
        model_equation_string (str): The original string defining the model.
        dependent_vars (list[str]): List of dependent variable names.
        terms (list[str]): List of all terms on the right-hand side of the equation.
        fixed_effects (list[str]): List of terms designated as fixed effects.
                                   Populated by `build_model` (for "intercept") or `set_covariate`.
        random_effects (list[dict]): List of dictionaries, each defining a random effect,
                                     populated by `set_random`. Each dict contains keys like
                                     'term', 'type', 'relationship_matrix', 'data_column'.
        covariates (list[dict]): List of dictionaries, each defining a covariate,
                                 populated by `set_covariate`. Each dict contains keys
                                 like 'term', 'type', 'data_column'.
        variance_components (any): Placeholder for variance component information.
                                   (Currently stores initial value passed to __init__).
    """
    def __init__(self, model_equation_string: str, fixed_effects=None, random_effects=None, variance_components=None):
        """
        Initializes a ModelDefinition from a model equation string.

        The model equation string is parsed to identify dependent variables and terms.
        Initial lists for fixed_effects, random_effects, and covariates are set up.

        Args:
            model_equation_string (str): A string representing the model
                (e.g., "y = intercept + x1 + x2*x3"). Must not be empty.
            fixed_effects: Initial value for fixed effects (mostly for internal/future use,
                           as `set_covariate` and `build_model` manage the `fixed_effects` list).
            random_effects: Initial value for random effects (similar to `fixed_effects`).
            variance_components: Initial value for variance components.
        
        Raises:
            ValueError: If `model_equation_string` is empty or invalid for parsing.
        """
        if not isinstance(model_equation_string, str) or not model_equation_string.strip():
            raise ValueError("model_equation_string must be a non-empty string.")
            
        self.model_equation_string = model_equation_string
        parsed_equation = parse_model_equation(model_equation_string)
        self.dependent_vars = parsed_equation["dependent_vars"]
        self.terms = parsed_equation["terms"]
        
        self.fixed_effects = [] 
        self.random_effects = [] 
        self.covariates = [] 
        
        self._original_fixed_effects = fixed_effects 
        self._original_random_effects = random_effects 
        self.variance_components = variance_components

    def set_covariate(self, term_name: str, data_column: str = None):
        """
        Designates a term from the model equation as a covariate (fixed effect).

        This method adds the term to `self.covariates` (as a dictionary with its
        details) and also to `self.fixed_effects` (as a string, ensuring no duplicates).

        Args:
            term_name (str): The name of the term (must be present in `self.terms`
                             parsed from the model equation).
            data_column (str, optional): The name of the column in the input data
                                         that corresponds to this term. If None,
                                         `term_name` is used as the `data_column`.
        
        Raises:
            ValueError: If `term_name` is not found in `self.terms`.
        """
        if term_name not in self.terms:
            raise ValueError(f"Term '{term_name}' not found in model equation terms: {self.terms}")

        self.covariates.append({
            'term': term_name,
            'type': 'covariate',
            'data_column': data_column if data_column else term_name
        })
        if term_name not in self.fixed_effects:
            self.fixed_effects.append(term_name)

    def set_random(self, term_name_str, relationship_matrix=None, data_column=None):
        """
        Designates one or more terms from the model equation as random effects.

        Each specified term is added to `self.random_effects` as a dictionary
        containing its details, including an optional relationship matrix and data column.
        Random effects are typically not added to `self.fixed_effects` unless
        explicitly set as covariates as well.

        Args:
            term_name_str (str): A string containing one or more term names from
                                 `self.terms`, separated by spaces.
            relationship_matrix (str, optional): A path or identifier for a
                                                 relationship matrix associated with
                                                 these random effect(s). Defaults to None.
            data_column (str, optional): The name of the column in the input data
                                         that corresponds to these term(s). If None,
                                         the respective `individual_term_name` is used.
                                         If multiple terms are specified, this `data_column`
                                         applies to all of them.
        
        Raises:
            ValueError: If `term_name_str` is empty or if any `individual_term_name`
                        derived from it is not found in `self.terms`.
        """
        individual_term_names = term_name_str.split()
        if not individual_term_names: # Handles empty string or string with only spaces
            raise ValueError("term_name_str cannot be empty or just whitespace.")

        for individual_term_name in individual_term_names:
            if individual_term_name not in self.terms:
                raise ValueError(f"Term '{individual_term_name}' from '{term_name_str}' not found in model equation terms: {self.terms}")
            
            self.random_effects.append({
                'term': individual_term_name,
                'type': 'random',
                'relationship_matrix': relationship_matrix,
                'data_column': data_column if data_column else individual_term_name
            })
            # Typically, random effects are not also fixed effects unless specified,
            # so we don't add to self.fixed_effects here. The user can call set_covariate if needed.

def build_model(model_equation_string: str) -> ModelDefinition:
    """
    Factory function to create and partially configure a ModelDefinition instance.

    This function initializes a `ModelDefinition` with the given equation string.
    It then automatically identifies if "intercept" is a term in the equation
    and, if so, adds it to the `fixed_effects` list of the model instance.

    Args:
        model_equation_string (str): The model equation string (e.g., "y = intercept + x1").
                                     Must not be empty.

    Returns:
        ModelDefinition: A new instance of `ModelDefinition` configured with the
                         parsed equation and intercept handling.
    
    Raises:
        ValueError: If `model_equation_string` is empty or invalid for parsing by ModelDefinition.
    """
    # ModelDefinition's __init__ will raise ValueError if equation string is empty,
    # which is implicitly handled here.
    model_instance = ModelDefinition(model_equation_string)

    # Automatically add "intercept" to fixed_effects if it's in the parsed terms.
    if "intercept" in model_instance.terms:
        if "intercept" not in model_instance.fixed_effects:
            model_instance.fixed_effects.append("intercept")
        # Note: "intercept" is not added to model_instance.covariates by default here.
        # User can call model_instance.set_covariate("intercept") if specific covariate entry is needed.
    
    return model_instance
