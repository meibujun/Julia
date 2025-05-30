"""
Provides functions for parsing model equation strings used in statistical modeling.
"""
import re

def parse_model_equation(equation_string: str) -> dict:
    """
    Parses a model equation string into its dependent variables and terms.

    The function splits the equation string at the '=' sign. The left part is
    treated as dependent variable(s), and the right part as terms. Dependent
    variables and terms are split by '+' and stripped of whitespace.
    Interaction terms are expected to be specified with a '*' (e.g., "A*B").

    Args:
        equation_string (str): The model equation string to parse.
            Example: "y1 + y2 = intercept + genotype + environment + genotype*environment"

    Returns:
        dict: A dictionary with two keys:
            - "dependent_vars" (list[str]): A list of dependent variable names.
            - "terms" (list[str]): A list of terms (e.g., "intercept", "genotype", "genotype*environment").
    
    Raises:
        TypeError: If `equation_string` is not a string.
        ValueError: If the equation string is malformed (e.g., no '=', missing
                    dependent variables or terms, invalid term format).
    """
    if not isinstance(equation_string, str):
        raise TypeError("Input equation_string must be a string.")

    if '=' not in equation_string:
        raise ValueError("Equation string must contain '=' to separate dependent and independent variables.")

    dependent_part, independent_part = equation_string.split('=', 1)

    dependent_vars = [var.strip() for var in dependent_part.split('+') if var.strip()]

    # Split terms by '+', then handle interaction terms (those with '*')
    # Terms are stripped of whitespace.
    raw_terms = [term.strip() for term in independent_part.split('+') if term.strip()]
    
    # Validate terms to ensure they are valid identifiers or interaction terms
    # A valid identifier starts with a letter or underscore, followed by letters, digits, or underscores.
    # An interaction term is two valid identifiers separated by a '*'.
    term_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\*[a-zA-Z_][a-zA-Z0-9_]*)?$")
    
    terms = []
    for term in raw_terms:
        if not term_pattern.match(term):
            raise ValueError(f"Invalid term format: '{term}'. Terms must be valid identifiers or two identifiers separated by '*'.")
        terms.append(term)
        
    if not dependent_vars:
        raise ValueError("No dependent variables found in the equation.")
    if not terms:
        raise ValueError("No terms found in the equation.")

    return {
        "dependent_vars": dependent_vars,
        "terms": terms
    }
