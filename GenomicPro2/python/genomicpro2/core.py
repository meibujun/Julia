"""
Core GenomicPro2 Python Interface

Provides Python bindings to GenomicPro2 Julia package via PyJulia.
"""

import os
import warnings
from typing import Optional, Union, List, Dict, Any
import numpy as np

try:
    from julia import Julia, Main
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False
    warnings.warn(
        "PyJulia not found. Please install: pip install julia\n"
        "Then run: python -c 'import julia; julia.install()'"
    )


class GenomicPro2:
    """
    Main GenomicPro2 interface for Python.

    This class provides Python bindings to the GenomicPro2 Julia package,
    allowing seamless access to all genomic analysis functionality.

    Attributes:
        julia: Julia runtime instance
        gp2: GenomicPro2 Julia module

    Example:
        >>> gp = GenomicPro2()
        >>> genotypes = gp.read_plink("data/genotypes")
        >>> phenotypes = gp.read_phenotypes("data/phenotypes.csv")
        >>> results = gp.gwas(genotypes, phenotypes)
    """

    def __init__(self, julia_project: Optional[str] = None,
                 compiled_modules: bool = False):
        """
        Initialize GenomicPro2 Python interface.

        Args:
            julia_project: Path to Julia project (optional)
            compiled_modules: Use compiled modules for faster loading
        """
        if not JULIA_AVAILABLE:
            raise ImportError(
                "PyJulia is required. Install with:\n"
                "  pip install julia\n"
                "  python -c 'import julia; julia.install()'"
            )

        # Initialize Julia
        self.julia = Julia(compiled_modules=compiled_modules)

        # Activate project if specified
        if julia_project:
            Main.eval(f'using Pkg; Pkg.activate("{julia_project}")')

        # Load GenomicPro2
        Main.eval("using GenomicPro2")
        self.gp2 = Main.GenomicPro2

        print("GenomicPro2 initialized successfully!")

    # ========================================================================
    # Data I/O Methods
    # ========================================================================

    def read_plink(self, prefix: str, verbose: bool = True) -> Any:
        """
        Read PLINK binary format files.

        Args:
            prefix: File prefix (e.g., "data/genotypes")
            verbose: Show progress

        Returns:
            CompactGenotypes object
        """
        return Main.eval(f'read_plink("{prefix}", verbose={verbose})')

    def read_vcf(self, filename: str, **kwargs) -> Any:
        """
        Read VCF format file.

        Args:
            filename: VCF file path
            **kwargs: Additional arguments

        Returns:
            CompactGenotypes object
        """
        return Main.eval(f'read_vcf("{filename}")')

    def read_phenotypes(self, filename: str,
                       id_col: int = 1,
                       pheno_col: int = 2) -> Any:
        """
        Read phenotype data from CSV.

        Args:
            filename: CSV file path
            id_col: Column index for IDs
            pheno_col: Column index for phenotype values

        Returns:
            PhenotypeData object
        """
        return Main.eval(
            f'read_phenotypes("{filename}", '
            f'id_col={id_col}, pheno_col={pheno_col})'
        )

    # ========================================================================
    # GWAS Methods
    # ========================================================================

    def gwas(self, genotypes, phenotypes,
             model: str = 'mixed',
             adjust_pcs: bool = True,
             n_pcs: int = 10,
             parallel: bool = True) -> Dict[str, Any]:
        """
        Perform genome-wide association study.

        Args:
            genotypes: Genotype data
            phenotypes: Phenotype data
            model: 'linear' or 'mixed'
            adjust_pcs: Adjust for population structure
            n_pcs: Number of principal components
            parallel: Use parallel computation

        Returns:
            Dictionary with GWAS results
        """
        Main.genotypes = genotypes
        Main.phenotypes = phenotypes

        if model == 'linear':
            Main.eval(
                f'model = LinearModelGWAS('
                f'adjust_population_structure={adjust_pcs}, '
                f'n_pcs={n_pcs})'
            )
        else:
            Main.eval('model = MixedModelGWAS()')

        Main.eval(f'results = perform_gwas(genotypes, phenotypes, model, '
                 f'parallel={parallel})')

        # Convert to Python dict
        results = {
            'snp_ids': list(Main.results.snp_ids),
            'chromosomes': list(Main.results.chromosomes),
            'positions': list(Main.results.positions),
            'pvalues': np.array(Main.results.pvalues),
            'effect_sizes': np.array(Main.results.effect_sizes),
            'std_errors': np.array(Main.results.std_errors),
        }

        return results

    # ========================================================================
    # Model Methods
    # ========================================================================

    def gblup(self, **kwargs):
        """Create GBLUP model."""
        from .models import GBLUP
        return GBLUP(self, **kwargs)

    def bayesr(self, **kwargs):
        """Create BayesR model."""
        from .models import BayesR
        return BayesR(self, **kwargs)

    def bayescpi(self, **kwargs):
        """Create BayesCπ model."""
        from .models import BayesCpi
        return BayesCpi(self, **kwargs)

    def rkhs(self, **kwargs):
        """Create RKHS model."""
        from .models import RKHS
        return RKHS(self, **kwargs)

    def deepgblup(self, **kwargs):
        """Create Deep GBLUP model."""
        from .models import DeepGBLUP
        return DeepGBLUP(self, **kwargs)

    # ========================================================================
    # Quality Control
    # ========================================================================

    def quality_control(self, genotypes, phenotypes,
                       maf_threshold: float = 0.05,
                       missing_rate_threshold: float = 0.1,
                       hwe_pvalue: float = 1e-6):
        """
        Apply quality control filters.

        Args:
            genotypes: Genotype data
            phenotypes: Phenotype data
            maf_threshold: Minor allele frequency threshold
            missing_rate_threshold: Maximum missing rate
            hwe_pvalue: Hardy-Weinberg p-value threshold

        Returns:
            Filtered genotypes, phenotypes, and QC report
        """
        Main.genotypes = genotypes
        Main.phenotypes = phenotypes

        Main.eval(
            f'filters = QCFilters('
            f'maf_threshold={maf_threshold}, '
            f'missing_rate_threshold={missing_rate_threshold}, '
            f'hwe_pvalue={hwe_pvalue})'
        )

        Main.eval('geno_qc, pheno_qc, report = '
                 'quality_control(genotypes, phenotypes, filters)')

        return Main.geno_qc, Main.pheno_qc, Main.report

    # ========================================================================
    # Population Structure
    # ========================================================================

    def pca(self, genotypes, n_components: int = 20,
            method: str = 'svd') -> Dict[str, np.ndarray]:
        """
        Perform PCA analysis.

        Args:
            genotypes: Genotype data
            n_components: Number of principal components
            method: 'svd' or 'eig'

        Returns:
            Dictionary with PCA results
        """
        Main.genotypes = genotypes
        Main.eval(f'pca_results = perform_pca(genotypes, '
                 f'n_components={n_components}, method=:{method})')

        return {
            'scores': np.array(Main.pca_results.scores),
            'loadings': np.array(Main.pca_results.loadings),
            'explained_variance': np.array(Main.pca_results.explained_variance),
        }

    def admixture(self, genotypes, K: int = 3,
                  niter: int = 1000) -> Dict[str, Any]:
        """
        Perform ADMIXTURE analysis.

        Args:
            genotypes: Genotype data
            K: Number of ancestral populations
            niter: Maximum iterations

        Returns:
            Dictionary with admixture results
        """
        Main.genotypes = genotypes
        Main.eval(f'adm_results = perform_admixture(genotypes, '
                 f'K={K}, niter={niter})')

        return {
            'Q': np.array(Main.adm_results.Q),
            'F': np.array(Main.adm_results.F),
            'log_likelihood': float(Main.adm_results.log_likelihood),
            'converged': bool(Main.adm_results.converged),
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def compute_grm(self, genotypes, method: str = 'vanraden',
                    parallel: bool = True, use_gpu: bool = False):
        """
        Compute genomic relationship matrix.

        Args:
            genotypes: Genotype data
            method: 'vanraden' or 'additive'
            parallel: Use parallel computation
            use_gpu: Use GPU acceleration

        Returns:
            GRM matrix as numpy array
        """
        Main.genotypes = genotypes

        if use_gpu:
            Main.eval('grm = compute_grm_gpu(genotypes)')
        elif parallel:
            Main.eval(f'grm = compute_grm_parallel(genotypes, method=:{method})')
        else:
            Main.eval(f'grm = compute_grm(genotypes, method=:{method})')

        return np.array(Main.grm)

    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return bool(Main.eval('has_cuda()'))

    def gpu_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        if not self.has_gpu():
            return {'available': False}

        info = Main.eval('gpu_info()')
        return dict(info)


# Convenience function
def load_genomicpro2(julia_project: Optional[str] = None) -> GenomicPro2:
    """
    Load GenomicPro2 package.

    Args:
        julia_project: Path to Julia project

    Returns:
        GenomicPro2 instance
    """
    return GenomicPro2(julia_project=julia_project)
