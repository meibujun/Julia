"""
Quality Control for Python

Python wrappers for GenomicPro2 quality control functionality.
"""

from typing import Dict, Any, Optional, Tuple
from julia import Main


class QualityControl:
    """
    Quality control filters for genomic data.

    Example:
        >>> qc = QualityControl(gp)
        >>> filtered_geno, filtered_pheno, report = qc.filter(
        ...     genotypes, phenotypes,
        ...     maf=0.05, missing=0.1, hwe=1e-6
        ... )
        >>> print(report)
    """

    def __init__(self, gp2_instance):
        """
        Initialize QC.

        Args:
            gp2_instance: GenomicPro2 instance
        """
        self.gp2 = gp2_instance

    def filter(self, genotypes, phenotypes,
               maf_threshold: float = 0.05,
               missing_rate_threshold: float = 0.1,
               hwe_pvalue: float = 1e-6) -> Tuple[Any, Any, Dict]:
        """
        Apply quality control filters.

        Args:
            genotypes: Genotype data
            phenotypes: Phenotype data
            maf_threshold: Minor allele frequency threshold
            missing_rate_threshold: Maximum missing rate
            hwe_pvalue: Hardy-Weinberg p-value threshold

        Returns:
            Tuple of (filtered_genotypes, filtered_phenotypes, report)
        """
        return self.gp2.quality_control(
            genotypes, phenotypes,
            maf_threshold=maf_threshold,
            missing_rate_threshold=missing_rate_threshold,
            hwe_pvalue=hwe_pvalue
        )

    def ld_pruning(self, genotypes,
                   window_size: int = 50,
                   threshold: float = 0.2):
        """
        Perform LD pruning.

        Args:
            genotypes: Genotype data
            window_size: Window size for LD calculation
            threshold: LD r² threshold

        Returns:
            Indices of pruned SNPs
        """
        Main.genotypes = genotypes
        pruned = Main.eval(f'ld_pruning(genotypes, '
                          f'window_size={window_size}, '
                          f'threshold={threshold})')
        return list(pruned)
