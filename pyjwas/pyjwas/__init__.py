# pyjwas package initialization

# Import key classes and functions to be available at the top level
from .data import PhenotypeData, PedigreeData, GenotypeData
from .model import build_model, ModelDefinition
from .parser import parse_model_equation # Also useful standalone
from .mcmc import run_mcmc, BaseMCMCSampler # GBLUPSampler can be accessed via run_mcmc
from .results import save_summary_df, save_mcmc_samples

# Define __all__ for explicit public API if desired
__all__ = [
    "PhenotypeData", 
    "PedigreeData", 
    "GenotypeData",
    "build_model", 
    "ModelDefinition",
    "parse_model_equation",
    "run_mcmc",
    "BaseMCMCSampler", # Exposing BaseMCMCSampler for extensibility
    "save_summary_df",
    "save_mcmc_samples"
]

# You can also include a version string
__version__ = "0.1.0-alpha"

print("pyjwas package loaded.")
