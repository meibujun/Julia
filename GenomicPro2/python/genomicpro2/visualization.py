"""
Visualization tools for Python

Python wrappers for GenomicPro2 visualization functionality.
"""

import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from julia import Main


class ManhattanPlot:
    """
    Manhattan plot for GWAS results.

    Example:
        >>> plot = ManhattanPlot()
        >>> plot.plot(gwas_results)
        >>> plot.save("manhattan.png")
    """

    def __init__(self):
        self.fig = None
        self.ax = None

    def plot(self, results: Dict[str, Any],
             significant_threshold: float = 5e-8,
             suggestive_threshold: float = 1e-5,
             title: str = "Manhattan Plot",
             figsize=(14, 6)):
        """
        Create Manhattan plot.

        Args:
            results: GWAS results dictionary
            significant_threshold: Genome-wide significance threshold
            suggestive_threshold: Suggestive threshold
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure and axes
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)

        chromosomes = np.array(results['chromosomes'])
        positions = np.array(results['positions'])
        pvalues = np.array(results['pvalues'])

        # Calculate -log10(p)
        log_pvalues = -np.log10(pvalues)

        # Calculate cumulative positions
        cum_pos = []
        last_pos = 0
        chr_centers = []

        for chrom in sorted(set(chromosomes)):
            chr_mask = chromosomes == chrom
            chr_positions = positions[chr_mask]

            if len(chr_positions) > 0:
                max_pos = chr_positions.max()
                cum_pos.extend(positions[chr_mask] + last_pos)
                chr_centers.append(last_pos + max_pos / 2)
                last_pos += max_pos

        # Plot points
        colors = ['#1f77b4', '#ff7f0e']
        for i, chrom in enumerate(sorted(set(chromosomes))):
            chr_mask = chromosomes == chrom
            self.ax.scatter(
                np.array(cum_pos)[chr_mask],
                log_pvalues[chr_mask],
                c=colors[i % 2],
                s=5,
                alpha=0.6
            )

        # Add significance lines
        self.ax.axhline(y=-np.log10(significant_threshold),
                       color='red', linestyle='--',
                       label=f'Significant ({significant_threshold})')
        self.ax.axhline(y=-np.log10(suggestive_threshold),
                       color='blue', linestyle='--',
                       label=f'Suggestive ({suggestive_threshold})')

        # Labels
        self.ax.set_xlabel('Chromosome')
        self.ax.set_ylabel('-log10(P-value)')
        self.ax.set_title(title)
        self.ax.set_xticks(chr_centers)
        self.ax.set_xticklabels(sorted(set(chromosomes)))
        self.ax.legend()

        plt.tight_layout()
        return self.fig, self.ax

    def save(self, filename: str, dpi: int = 300):
        """Save plot to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')


class QQPlot:
    """
    QQ plot for GWAS p-values.

    Example:
        >>> plot = QQPlot()
        >>> plot.plot(pvalues)
        >>> plot.save("qq_plot.png")
    """

    def __init__(self):
        self.fig = None
        self.ax = None

    def plot(self, pvalues: np.ndarray,
             confidence_interval: float = 0.95,
             title: str = "QQ Plot",
             figsize=(8, 8)):
        """
        Create QQ plot.

        Args:
            pvalues: P-values array
            confidence_interval: CI level
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure and axes
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Sort p-values
        sorted_pvalues = np.sort(pvalues)
        n = len(sorted_pvalues)

        # Calculate expected p-values
        expected = (np.arange(1, n + 1) - 0.5) / n

        # Transform to -log10
        observed = -np.log10(sorted_pvalues)
        expected_log = -np.log10(expected)

        # Calculate lambda
        lambda_gc = np.median(observed ** 2) / 0.455

        # Plot
        self.ax.scatter(expected_log, observed, s=10, alpha=0.6)

        # Add diagonal line
        max_val = max(expected_log.max(), observed.max())
        self.ax.plot([0, max_val], [0, max_val],
                    'r--', label='Expected')

        # Labels
        self.ax.set_xlabel('Expected -log10(P-value)')
        self.ax.set_ylabel('Observed -log10(P-value)')
        self.ax.set_title(f'{title} (λ = {lambda_gc:.3f})')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self.fig, self.ax

    def save(self, filename: str, dpi: int = 300):
        """Save plot to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')


class PCAPlot:
    """
    PCA plot for population structure.

    Example:
        >>> plot = PCAPlot()
        >>> plot.plot(pca_results, pc_x=1, pc_y=2)
        >>> plot.save("pca_plot.png")
    """

    def __init__(self):
        self.fig = None
        self.ax = None

    def plot(self, pca_results: Dict[str, np.ndarray],
             pc_x: int = 1, pc_y: int = 2,
             labels: Optional[np.ndarray] = None,
             title: str = "PCA Plot",
             figsize=(10, 8)):
        """
        Create PCA plot.

        Args:
            pca_results: PCA results dictionary
            pc_x: X-axis PC
            pc_y: Y-axis PC
            labels: Sample labels for coloring
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure and axes
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)

        scores = pca_results['scores']
        explained_var = pca_results['explained_variance']

        pc_x_idx = pc_x - 1
        pc_y_idx = pc_y - 1

        if labels is not None:
            scatter = self.ax.scatter(
                scores[:, pc_x_idx],
                scores[:, pc_y_idx],
                c=labels,
                cmap='tab10',
                s=50,
                alpha=0.6
            )
            self.fig.colorbar(scatter, ax=self.ax, label='Population')
        else:
            self.ax.scatter(
                scores[:, pc_x_idx],
                scores[:, pc_y_idx],
                s=50,
                alpha=0.6
            )

        # Labels
        self.ax.set_xlabel(
            f'PC{pc_x} ({explained_var[pc_x_idx]*100:.2f}% variance)'
        )
        self.ax.set_ylabel(
            f'PC{pc_y} ({explained_var[pc_y_idx]*100:.2f}% variance)'
        )
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self.fig, self.ax

    def save(self, filename: str, dpi: int = 300):
        """Save plot to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
