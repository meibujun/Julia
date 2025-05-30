import pytest

# Assuming pyjwas is installed or PYTHONPATH is set up correctly
from pyjwas.parser import parse_model_equation
from pyjwas.model import build_model, ModelDefinition

def test_parse_simple_equation():
    """Tests parsing a simple model equation."""
    equation = "y1 = intercept + x1"
    parsed = parse_model_equation(equation)
    
    assert parsed["dependent_vars"] == ["y1"], "Dependent variables parsing incorrect."
    assert parsed["terms"] == ["intercept", "x1"], "Terms parsing incorrect."

def test_parse_multi_dependent_equation():
    """Tests parsing an equation with multiple dependent variables."""
    equation = "y1 + y2 = intercept + x1 + x2"
    parsed = parse_model_equation(equation)
    
    assert parsed["dependent_vars"] == ["y1", "y2"], "Multiple dependent variables parsing incorrect."
    assert parsed["terms"] == ["intercept", "x1", "x2"], "Terms parsing incorrect for multi-dependent."

def test_parse_interaction_equation():
    """Tests parsing an equation with interaction terms."""
    equation = "yield = intercept + variety + location + variety*location"
    parsed = parse_model_equation(equation)
    
    assert parsed["dependent_vars"] == ["yield"], "Dependent variable parsing incorrect for interaction."
    expected_terms = ["intercept", "variety", "location", "variety*location"]
    assert parsed["terms"] == expected_terms, f"Interaction terms parsing incorrect. Expected {expected_terms}, got {parsed['terms']}"

def test_parse_equation_with_extra_spaces():
    """Tests parsing with varied spacing."""
    equation = "  y  =  intercept  +  x1 * x2  +  x3 "
    parsed = parse_model_equation(equation)
    
    assert parsed["dependent_vars"] == ["y"], "Dependent variable parsing incorrect with extra spaces."
    expected_terms = ["intercept", "x1*x2", "x3"]
    assert parsed["terms"] == expected_terms, f"Terms parsing incorrect with extra spaces. Expected {expected_terms}, got {parsed['terms']}"

def test_parse_invalid_equations():
    """Tests that invalid equations raise ValueErrors."""
    with pytest.raises(ValueError, match="Equation string must contain '='"):
        parse_model_equation("y1 intercept + x1")
        
    with pytest.raises(ValueError, match="No dependent variables found"):
        parse_model_equation("= intercept + x1")
        
    with pytest.raises(ValueError, match="No terms found"):
        parse_model_equation("y1 = ")

    with pytest.raises(TypeError, match="Input equation_string must be a string."):
        parse_model_equation(None)

    with pytest.raises(ValueError, match="Invalid term format: 'x1**x2'"):
        parse_model_equation("y1 = x1**x2") # Only '*' for interaction

    with pytest.raises(ValueError, match="Invalid term format: 'x1*x2*x3'"):
        parse_model_equation("y1 = x1*x2*x3") # Only two-way

def test_build_model_intercept():
    """Tests that build_model correctly identifies and adds 'intercept'."""
    model_instance = build_model("y = intercept + x1 + x2")
    
    assert "intercept" in model_instance.terms, "'intercept' should be in parsed terms."
    assert "intercept" in model_instance.fixed_effects, "'intercept' should be automatically added to fixed_effects by build_model."
    
    model_instance_no_intercept = build_model("y = x1 + x2")
    assert "intercept" not in model_instance_no_intercept.terms
    assert "intercept" not in model_instance_no_intercept.fixed_effects

def test_set_covariate_random_in_model_definition():
    """Tests setting covariates and random effects in ModelDefinition."""
    # Equation defines available terms
    md = ModelDefinition("yield = Env + Variety + Variety*Env + error_term")
    
    # Set 'Env' as a covariate
    md.set_covariate("Env", data_column="EnvironmentColumn")
    assert any(c['term'] == "Env" and c['data_column'] == "EnvironmentColumn" for c in md.covariates), "Covariate 'Env' not set correctly."
    assert "Env" in md.fixed_effects, "'Env' should be added to fixed_effects."
    
    # Set 'Variety' as a random effect
    md.set_random("Variety", relationship_matrix="A_matrix.txt")
    assert any(r['term'] == "Variety" and r['relationship_matrix'] == "A_matrix.txt" for r in md.random_effects), "Random effect 'Variety' not set correctly."
    
    # Test setting multiple random effects at once
    md.set_random("Variety*Env error_term") # Assuming "error_term" was meant to be random
    assert any(r['term'] == "Variety*Env" for r in md.random_effects), "Random effect 'Variety*Env' not set."
    assert any(r['term'] == "error_term" for r in md.random_effects), "Random effect 'error_term' not set."


    # Test error handling
    with pytest.raises(ValueError, match="'NonExistentTerm' not found in model equation terms"):
        md.set_covariate("NonExistentTerm")
        
    with pytest.raises(ValueError, match="'AnotherBadTerm' from 'AnotherBadTerm' not found"):
        md.set_random("AnotherBadTerm")

    # Test that intercept can be set as a covariate if desired (build_model handles it automatically for fixed_effects)
    md_with_intercept = ModelDefinition("y = intercept + x")
    md_with_intercept.set_covariate("intercept") # Manually setting it as covariate
    assert any(c['term'] == "intercept" for c in md_with_intercept.covariates)
    assert "intercept" in md_with_intercept.fixed_effects

# To run these tests, navigate to the directory containing `pyjwas` 
# and run `pytest pyjwas/tests/test_model_parser.py`
# Ensure pyjwas is in PYTHONPATH or installed.
# Example: export PYTHONPATH=$PYTHONPATH:$(pwd) (from the directory containing the pyjwas folder)
