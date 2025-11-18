"""
Setup script for GenomicPro2 Python package
"""

from setuptools import setup, find_packages
import os

# Read long description from README
readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "GenomicPro2: High-Performance Genomic Analysis Toolkit"

setup(
    name='genomicpro2',
    version='2.0.0',
    author='GenomicPro2 Development Team',
    author_email='genomicpro2@example.com',
    description='Python bindings for GenomicPro2 genomic analysis toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/meibujun/Julia',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'julia>=0.6.0',
        'numpy>=1.20.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'sphinx>=4.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.900',
        ],
        'viz': [
            'plotly>=5.0',
            'pandas>=1.3.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'genomicpro2-python=genomicpro2.cli:main',
        ],
    },
    keywords=[
        'genomics',
        'gwas',
        'genomic-prediction',
        'bioinformatics',
        'gblup',
        'bayesian-methods',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/meibujun/Julia/issues',
        'Documentation': 'https://github.com/meibujun/Julia/tree/main/docs',
        'Source': 'https://github.com/meibujun/Julia',
    },
)
