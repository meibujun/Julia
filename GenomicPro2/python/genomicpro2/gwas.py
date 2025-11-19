"""
GWAS Analysis for Python

Python wrappers for GenomicPro2 GWAS functionality.
"""

import numpy as np
from typing import Dict, Any, Optional
from julia import Main


class GWAS:
    """
    Genome-Wide Association Study interface.

    Example:
        >>> gwas = GWAS(gp)
        >>> results = gwas.run(genotypes, phenotypes, model='mixed')
        >>> significant = gwas.get_significant_snps(results, threshold=5e-8)
    """

    def __init__(self, gp2_instance):
        """
        Initialize GWAS.

        Args:
            gp2_instance: GenomicPro2 instance
        """
        self.gp2 = gp2_instance

    def run(self, genotypes, phenotypes,
            model: str = 'mixed',
            adjust_pcs: bool = True,
            n_pcs: int = 10,
            parallel: bool = True) -> Dict[str, Any]:
        """
        Run GWAS analysis.

        Args:
            genotypes: Genotype data
            phenotypes: Phenotype data
            model: 'linear' or 'mixed'
            adjust_pcs: Adjust for population structure
            n_pcs: Number of PCs
            parallel: Use parallel computation

        Returns:
            Dictionary with GWAS results
        """
        return self.gp2.gwas(genotypes, phenotypes, model=model,
                            adjust_pcs=adjust_pcs, n_pcs=n_pcs,
                            parallel=parallel)

    def adjust_pvalues(self, pvalues: np.ndarray,
                       method: str = 'bonferroni') -> np.ndarray:
        """
        Adjust p-values for multiple testing.

        Args:
            pvalues: P-values array
            method: 'bonferroni', 'fdr', or 'sidak'

        Returns:
            Adjusted p-values
        """
        Main.pvalues = pvalues
        adjusted = Main.eval(f'adjust_pvalues(pvalues, method=:{method})')
        return np.array(adjusted)

    def get_significant_snps(self, results: Dict[str, Any],
                            threshold: float = 5e-8) -> Dict[str, Any]:
        """
        Get significant SNPs.

        Args:
            results: GWAS results dictionary
            threshold: Significance threshold

        Returns:
            Dictionary with significant SNPs
        """
        pvalues = results['pvalues']
        significant_idx = np.where(pvalues < threshold)[0]

        return {
            'snp_ids': [results['snp_ids'][i] for i in significant_idx],
            'chromosomes': [results['chromosomes'][i] for i in significant_idx],
            'positions': [results['positions'][i] for i in significant_idx],
            'pvalues': pvalues[significant_idx],
            'effect_sizes': results['effect_sizes'][significant_idx],
        }

    def calculate_lambda(self, pvalues: np.ndarray) -> float:
        """
        Calculate genomic inflation factor (lambda).

        Args:
            pvalues: P-values array

        Returns:
            Lambda value
        """
        Main.pvalues = pvalues
        lambda_val = Main.eval('calculate_genomic_control_lambda(pvalues)')
        return float(lambda_val)


class LinearModelGWAS:
    """Linear model GWAS with optional PCA correction."""

    def __init__(self, adjust_population_structure: bool = True,
                 n_pcs: int = 10):
        self.adjust_population_structure = adjust_population_structure
        self.n_pcs = n_pcs


class MixedModelGWAS:
    """Mixed linear model GWAS with random effects."""

    def __init__(self, grm: Optional[np.ndarray] = None,
                 reml: bool = True):
        self.grm = grm
        self.reml = reml
