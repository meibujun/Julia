"""
GenomicPro2 Python Package

Python bindings for GenomicPro2 - High-Performance Genomic Analysis Toolkit

This package provides Python interface to the Julia-based GenomicPro2 library
using PyJulia for seamless integration.

Installation:
    pip install genomicpro2

    # Install Julia packages
    python -c "import julia; julia.install()"

Requirements:
    - Python 3.8+
    - Julia 1.10+
    - PyJulia

Example:
    >>> from genomicpro2 import GenomicPro2
    >>> gp = GenomicPro2()
    >>>
    >>> # Load data
    >>> genotypes = gp.read_plink("data/genotypes")
    >>> phenotypes = gp.read_phenotypes("data/phenotypes.csv")
    >>>
    >>> # Run GWAS
    >>> results = gp.gwas(genotypes, phenotypes, model='mixed')
    >>>
    >>> # GBLUP prediction
    >>> model = gp.gblup()
    >>> model.fit(genotypes, phenotypes)
    >>> predictions = model.predict(genotypes)
"""

__version__ = "2.0.0"
__author__ = "GenomicPro2 Development Team"
__license__ = "MIT"

from .core import GenomicPro2
from .models import GBLUP, BayesR, BayesCpi, RKHS, DeepGBLUP
from .gwas import GWAS, LinearModelGWAS, MixedModelGWAS
from .qc import QualityControl
from .visualization import ManhattanPlot, QQPlot, PCAPlot

__all__ = [
    'GenomicPro2',
    'GBLUP',
    'BayesR',
    'BayesCpi',
    'RKHS',
    'DeepGBLUP',
    'GWAS',
    'LinearModelGWAS',
    'MixedModelGWAS',
    'QualityControl',
    'ManhattanPlot',
    'QQPlot',
    'PCAPlot',
]
