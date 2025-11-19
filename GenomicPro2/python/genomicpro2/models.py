"""
Genomic Prediction Models for Python

Python wrappers for GenomicPro2 models with scikit-learn style API.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from julia import Main


class BaseModel:
    """
    Base class for all genomic prediction models.

    Provides scikit-learn style interface with fit() and predict() methods.
    """

    def __init__(self, gp2_instance):
        """
        Initialize model.

        Args:
            gp2_instance: GenomicPro2 instance
        """
        self.gp2 = gp2_instance
        self.model = None
        self.fitted_ = False

    def fit(self, X, y, **kwargs):
        """
        Fit model to training data.

        Args:
            X: Genotype data
            y: Phenotype data
            **kwargs: Model-specific parameters

        Returns:
            self
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predict phenotypes.

        Args:
            X: Genotype data

        Returns:
            Predicted values as numpy array
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        Main.model = self.model
        Main.X = X
        predictions = Main.eval('predict(model, X)')
        return np.array(predictions)

    def score(self, X, y, metric='correlation'):
        """
        Evaluate model performance.

        Args:
            X: Genotype data
            y: True phenotype values
            metric: 'correlation', 'mse', 'mae', or 'r2'

        Returns:
            Score value
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before scoring")

        Main.model = self.model
        Main.X = X
        Main.y = y
        score = Main.eval(f'score(model, X, y, metric=:{metric})')
        return float(score)


class GBLUP(BaseModel):
    """
    Genomic Best Linear Unbiased Prediction model.

    Example:
        >>> model = GBLUP(gp)
        >>> model.fit(genotypes, phenotypes)
        >>> predictions = model.predict(genotypes)
        >>> print(f"Heritability: {model.heritability_}")
    """

    def __init__(self, gp2_instance):
        super().__init__(gp2_instance)
        self.heritability_ = None
        self.genetic_variance_ = None
        self.environmental_variance_ = None

    def fit(self, X, y, grm=None, method='reml', verbose=True):
        """
        Fit GBLUP model.

        Args:
            X: Genotype data
            y: Phenotype data
            grm: Pre-computed GRM (optional)
            method: 'reml' or 'ml'
            verbose: Print progress

        Returns:
            self
        """
        Main.X = X
        Main.y = y

        if grm is not None:
            Main.grm = grm
            Main.eval(f'model = GBLUPModel(); '
                     f'fit!(model, X, y, grm=grm, method=:{method}, '
                     f'verbose={verbose})')
        else:
            Main.eval(f'model = GBLUPModel(); '
                     f'fit!(model, X, y, method=:{method}, verbose={verbose})')

        self.model = Main.model
        self.heritability_ = float(Main.model.heritability)
        self.genetic_variance_ = float(Main.model.genetic_variance)
        self.environmental_variance_ = float(Main.model.environmental_variance)
        self.fitted_ = True

        return self


class BayesR(BaseModel):
    """
    BayesR Bayesian mixture model.

    Example:
        >>> model = BayesR(gp, n_components=4, niter=50000)
        >>> model.fit(genotypes, phenotypes)
        >>> predictions = model.predict(genotypes)
    """

    def __init__(self, gp2_instance, n_components=4,
                 niter=50000, burnin=10000):
        super().__init__(gp2_instance)
        self.n_components = n_components
        self.niter = niter
        self.burnin = burnin
        self.beta_ = None
        self.heritability_ = None

    def fit(self, X, y, verbose=True):
        """
        Fit BayesR model.

        Args:
            X: Genotype data
            y: Phenotype data
            verbose: Print progress

        Returns:
            self
        """
        Main.X = X
        Main.y = y

        Main.eval(f'model = BayesRModel(n_components={self.n_components}, '
                 f'niter={self.niter}, burnin={self.burnin}); '
                 f'fit!(model, X, y)')

        self.model = Main.model
        self.fitted_ = True

        return self


class BayesCpi(BaseModel):
    """
    BayesCπ Bayesian variable selection model.

    Example:
        >>> model = BayesCpi(gp, estimate_pi=True)
        >>> model.fit(genotypes, phenotypes)
        >>> print(f"Estimated π: {model.pi_}")
    """

    def __init__(self, gp2_instance, niter=50000, burnin=10000,
                 estimate_pi=True, pi_prior=0.995):
        super().__init__(gp2_instance)
        self.niter = niter
        self.burnin = burnin
        self.estimate_pi = estimate_pi
        self.pi_prior = pi_prior
        self.beta_ = None
        self.pi_ = None
        self.inclusion_prob_ = None

    def fit(self, X, y):
        """
        Fit BayesCπ model.

        Args:
            X: Genotype data
            y: Phenotype data

        Returns:
            self
        """
        Main.X = X
        Main.y = y

        Main.eval(f'results = fit_bayescpi(X, y, '
                 f'niter={self.niter}, burnin={self.burnin}, '
                 f'estimate_pi={self.estimate_pi}, pi_prior={self.pi_prior})')

        self.model = Main.results
        self.beta_ = np.array(Main.results.beta)
        self.pi_ = float(Main.results.pi_estimated)
        self.inclusion_prob_ = np.array(Main.results.inclusion_prob)
        self.fitted_ = True

        return self

    def predict(self, X):
        """Predict using BayesCπ model."""
        Main.model = self.model
        Main.X = X
        predictions = Main.eval('predict_bayescpi(model, X)')
        return np.array(predictions)


class RKHS(BaseModel):
    """
    Reproducing Kernel Hilbert Space model.

    Example:
        >>> model = RKHS(gp, kernel='gaussian', auto_bandwidth=True)
        >>> model.fit(genotypes, phenotypes)
        >>> predictions = model.predict(genotypes)
    """

    def __init__(self, gp2_instance, kernel='gaussian',
                 auto_bandwidth=True, lambda_reg=1e-5):
        super().__init__(gp2_instance)
        self.kernel = kernel
        self.auto_bandwidth = auto_bandwidth
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        """
        Fit RKHS model.

        Args:
            X: Genotype data
            y: Phenotype data

        Returns:
            self
        """
        Main.X = X
        Main.y = y

        if self.kernel == 'gaussian':
            Main.eval('kernel = GaussianKernel()')
        elif self.kernel == 'polynomial':
            Main.eval('kernel = PolynomialKernel()')
        elif self.kernel == 'exponential':
            Main.eval('kernel = ExponentialKernel()')
        else:
            Main.eval('kernel = GaussianKernel()')

        Main.eval(f'results = fit_rkhs(X, y, kernel=kernel, '
                 f'auto_bandwidth={self.auto_bandwidth}, '
                 f'lambda={self.lambda_reg})')

        self.model = Main.results
        self.fitted_ = True

        return self

    def predict(self, X):
        """Predict using RKHS model."""
        Main.model = self.model
        Main.X = X
        predictions = Main.eval('predict_rkhs(model, X)')
        return np.array(predictions)


class DeepGBLUP(BaseModel):
    """
    Deep Learning genomic prediction model.

    Example:
        >>> model = DeepGBLUP(gp, hidden_layers=[256, 128, 64])
        >>> model.fit(genotypes, phenotypes, epochs=100)
        >>> predictions = model.predict(genotypes)
    """

    def __init__(self, gp2_instance, hidden_layers=None,
                 activation='relu', dropout_rate=0.2):
        super().__init__(gp2_instance)
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.activation = activation
        self.dropout_rate = dropout_rate

    def fit(self, X, y, epochs=100, batch_size=32,
            learning_rate=0.001, early_stopping=True, patience=10):
        """
        Fit Deep GBLUP model.

        Args:
            X: Genotype data
            y: Phenotype data
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping: Use early stopping
            patience: Early stopping patience

        Returns:
            self
        """
        Main.X = X
        Main.y = y

        n_snps = X.shape[1] if hasattr(X, 'shape') else Main.eval('size(X, 2)')

        Main.eval(f'model = DeepGBLUP({n_snps}, '
                 f'hidden_layers={self.hidden_layers}, '
                 f'activation=:{self.activation}, '
                 f'dropout_rate={self.dropout_rate})')

        Main.eval(f'results = train_deepgblup!(model, X, y, '
                 f'epochs={epochs}, batch_size={batch_size}, '
                 f'learning_rate={learning_rate}, '
                 f'early_stopping={early_stopping}, patience={patience})')

        self.model = Main.model
        self.fitted_ = True

        return self

    def predict(self, X):
        """Predict using Deep GBLUP model."""
        Main.model = self.model
        Main.X = X
        predictions = Main.eval('predict_deepgblup(model, X)')
        return np.array(predictions)


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models with weighted averaging.

    Example:
        >>> models = [GBLUP(gp), BayesR(gp), RKHS(gp)]
        >>> ensemble = EnsembleModel(gp, models)
        >>> ensemble.fit(genotypes, phenotypes)
        >>> predictions = ensemble.predict(genotypes)
    """

    def __init__(self, gp2_instance, models: List[BaseModel]):
        super().__init__(gp2_instance)
        self.models = models
        self.weights_ = None

    def fit(self, X, y, optimize_weights=True):
        """
        Fit all models in ensemble.

        Args:
            X: Genotype data
            y: Phenotype data
            optimize_weights: Optimize model weights

        Returns:
            self
        """
        # Fit all models
        for model in self.models:
            model.fit(X, y)

        # Optimize weights
        if optimize_weights:
            correlations = [model.score(X, y, metric='correlation')
                          for model in self.models]
            total = sum(correlations)
            self.weights_ = [c / total for c in correlations]
        else:
            n = len(self.models)
            self.weights_ = [1.0 / n] * n

        self.fitted_ = True
        return self

    def predict(self, X):
        """Predict using ensemble."""
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        predictions = np.zeros(X.shape[0] if hasattr(X, 'shape')
                              else Main.eval('size(X, 1)'))

        for model, weight in zip(self.models, self.weights_):
            predictions += weight * model.predict(X)

        return predictions
