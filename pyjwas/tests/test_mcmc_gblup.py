import pytest
import numpy as np
import pandas as pd
import os

# Assuming pyjwas is installed or PYTHONPATH is set up correctly
from pyjwas.model import build_model, ModelDefinition
from pyjwas.mcmc import run_mcmc, GBLUPSampler # Import GBLUPSampler for direct instantiation
from pyjwas.data import PhenotypeData, GenotypeData

# Determine the directory of this test file to build paths to data files
current_dir = os.path.dirname(os.path.abspath(__file__))
TEST_PHENOTYPES_CSV = os.path.join(current_dir, "data", "test_phenotypes.csv")
TEST_GENOTYPES_CSV = os.path.join(current_dir, "data", "test_genotypes.csv")


@pytest.fixture
def basic_data():
    """Pytest fixture to load data once for multiple tests."""
    assert os.path.exists(TEST_PHENOTYPES_CSV), f"Test phenotype file not found: {TEST_PHENOTYPES_CSV}"
    assert os.path.exists(TEST_GENOTYPES_CSV), f"Test genotype file not found: {TEST_GENOTYPES_CSV}"
    
    pheno_data = PhenotypeData(TEST_PHENOTYPES_CSV)
    geno_data = GenotypeData(TEST_GENOTYPES_CSV)
    # Model: y1 depends on an intercept and 'sex' as a fixed effect.
    # 'g' (genomic random effect) will be implicitly handled by GBLUPSampler.
    model_def = build_model("y1 = intercept + sex") 
    # 'sex' is a column in test_phenotypes.csv, 'intercept' is standard.
    # build_model already adds 'intercept' to fixed_effects.
    # We need to explicitly declare 'sex' as a covariate if it's to be used as such.
    # The model_def.terms are ['intercept', 'sex'].
    # build_model adds 'intercept' to model_def.fixed_effects.
    # We need to add 'sex' to fixed_effects via set_covariate.
    model_def.set_covariate("sex") # This adds 'sex' to fixed_effects and covariates list.
    
    return pheno_data, geno_data, model_def

def test_gblup_run_smoke(basic_data):
    """Smoke test for a GBLUP run to ensure it completes and returns expected structure."""
    pheno_data, geno_data, model_def = basic_data
    
    n_iterations = 10
    burn_in = 2
    
    samples = run_mcmc(model_definition=model_def, 
                       phenotype_data=pheno_data, 
                       genotype_data=geno_data, 
                       n_iterations=n_iterations, 
                       burn_in=burn_in, 
                       method="GBLUP")
    
    assert isinstance(samples, dict), "Result of run_mcmc should be a dictionary."
    assert samples, "Samples dictionary should not be empty."
    
    expected_keys = ['beta', 'g', 'variance_components']
    for key in expected_keys:
        assert key in samples, f"'{key}' should be a key in the samples dictionary."
        assert len(samples[key]) == (n_iterations - burn_in), f"Number of stored samples for '{key}' is incorrect."

    # Check shapes of the stored samples (first sample after burn-in)
    # Beta: intercept + sex (2 fixed effects)
    # Note: GBLUPSampler._initialize_parameters constructs X based on model_def.fixed_effects
    # model_def.fixed_effects should be ['intercept', 'sex']
    num_fixed_effects = len(model_def.fixed_effects)
    assert num_fixed_effects == 2, f"Expected 2 fixed effects (intercept, sex), got {num_fixed_effects}"
    assert np.array(samples['beta'][0]).shape == (num_fixed_effects,), \
        f"Shape of beta samples is incorrect. Expected ({num_fixed_effects},), got {np.array(samples['beta'][0]).shape}"
    
    # g: number of genotyped individuals (5 in test_genotypes.csv)
    # GRM is 5x5, so g vector should be 5.
    # geno_data.genotype_matrix.shape[0] is not directly available here easily,
    # but geno_data.GRM.shape[0] is used by sampler.
    # We can access it via geno_data.grm.shape[0] if GRM has been calculated.
    if geno_data.grm is None:
        geno_data.calculate_grm() # Ensure GRM is calculated if not already by GBLUPSampler init
    
    num_genotyped_individuals = geno_data.grm.shape[0]
    assert num_genotyped_individuals == 5, "Expected 5 genotyped individuals."
    assert np.array(samples['g'][0]).shape == (num_genotyped_individuals,), \
        f"Shape of g samples is incorrect. Expected ({num_genotyped_individuals},), got {np.array(samples['g'][0]).shape}"

    # Variance components: dict with 'genetic' and 'residual'
    assert isinstance(samples['variance_components'][0], dict), "Variance components should be stored as dicts."
    assert 'genetic' in samples['variance_components'][0], "'genetic' variance missing."
    assert 'residual' in samples['variance_components'][0], "'residual' variance missing."


def test_gblup_ebv_summary_and_posterior_summary(basic_data):
    """Tests EBV summary and posterior summary generation."""
    pheno_data, geno_data, model_def = basic_data
    
    n_iterations = 12 # Using different numbers to ensure no hardcoding
    burn_in = 2
    
    # Instantiate GBLUPSampler directly to call its methods
    sampler = GBLUPSampler(model_definition=model_def,
                           phenotype_data=pheno_data,
                           genotype_data=geno_data,
                           n_iterations=n_iterations,
                           burn_in=burn_in)
    
    sampler.run() # Run the MCMC
    
    # Test get_ebv_summary()
    ebv_summary_df = sampler.get_ebv_summary()
    
    assert ebv_summary_df is not None, "EBV summary DataFrame should not be None."
    assert isinstance(ebv_summary_df, pd.DataFrame), "EBV summary should be a pandas DataFrame."
    
    expected_cols = ['individual_id', 'ebv_mean', 'ebv_std', 'ebv_ci_low', 'ebv_ci_high']
    for col in expected_cols:
        assert col in ebv_summary_df.columns, f"Column '{col}' missing in EBV summary DataFrame."
    
    # There are 5 individuals in test_genotypes.csv
    num_genotyped_individuals = len(geno_data.individual_ids) 
    assert len(ebv_summary_df) == num_genotyped_individuals, \
        f"EBV summary should have rows for all genotyped individuals. Expected {num_genotyped_individuals}, got {len(ebv_summary_df)}"
    assert ebv_summary_df['individual_id'].tolist() == geno_data.individual_ids, "Individual IDs in EBV summary do not match genotype data."

    # Test get_posterior_summary for 'beta'
    beta_summary = sampler.get_posterior_summary('beta')
    assert beta_summary is not None
    assert isinstance(beta_summary, dict)
    assert 'mean' in beta_summary and len(beta_summary['mean']) == len(model_def.fixed_effects)

    # Test get_posterior_summary for 'variance_components'
    vc_summary = sampler.get_posterior_summary('variance_components')
    assert vc_summary is not None
    assert isinstance(vc_summary, dict)
    assert 'genetic' in vc_summary and 'mean' in vc_summary['genetic']
    assert 'residual' in vc_summary and 'mean' in vc_summary['residual']


# To run: pytest pyjwas/tests/test_mcmc_gblup.py
# Ensure PYTHONPATH includes the parent directory of pyjwas.
# Example: export PYTHONPATH=$PYTHONPATH:$(pwd) (from the directory containing the pyjwas folder)
