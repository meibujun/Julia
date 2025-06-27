from typing import Union, Optional
import numpy as np

class VarianceCovariance:
    """
    Represents variance/covariance components and their associated parameters.
    Translated from the Julia JWAS.jl Variance struct.
    """
    def __init__(self,
                 value: Optional[Union[float, np.ndarray]] = None,
                 df: Optional[float] = None,
                 scale: Optional[Union[float, np.ndarray]] = None,
                 estimate_variance: bool = True,
                 estimate_scale: bool = False,
                 constraint: bool = False):
        """
        Initializes a VarianceCovariance instance.

        Args:
            value: The variance value (float for single-trait) or
                   covariance matrix (np.ndarray for multi-trait).
                   Can be None if to be estimated/set later.
            df: Degrees of freedom for the prior (e.g., for an inverse Wishart
                or scaled inverse chi-squared distribution). Can be None.
            scale: Scale parameter for the prior. Can be float or np.ndarray.
                   Can be None.
            estimate_variance: Boolean indicating if the variance/covariance
                               should be estimated during MCMC.
            estimate_scale: Boolean indicating if the scale parameter should be
                            estimated (less common).
            constraint: Boolean indicating if constraints apply (e.g., zero
                        covariances in a multi-trait model, making the matrix
                        diagonal).
        """
        self.value: Optional[Union[float, np.ndarray]] = value
        self.df: Optional[float] = df
        self.scale: Optional[Union[float, np.ndarray]] = scale
        self.estimate_variance: bool = estimate_variance
        self.estimate_scale: bool = estimate_scale
        self.constraint: bool = constraint

    def __repr__(self) -> str:
        value_repr = f"array({self.value.shape})" if isinstance(self.value, np.ndarray) else self.value
        scale_repr = f"array({self.scale.shape})" if isinstance(self.scale, np.ndarray) else self.scale
        return (f"VarianceCovariance(value={value_repr}, df={self.df}, scale={scale_repr}, "
                f"estimate_variance={self.estimate_variance}, estimate_scale={self.estimate_scale}, "
                f"constraint={self.constraint})")

if __name__ == '__main__':
    # Example Usage:
    # Single-trait residual variance
    R_single = VarianceCovariance(value=10.0, df=4.0, scale=8.0)
    print(R_single)

    # Multi-trait residual variance (2 traits)
    R_multi_val = np.array([[10.0, 2.0], [2.0, 5.0]])
    R_multi_scale = np.array([[8.0, 1.0], [1.0, 4.0]])
    R_multi = VarianceCovariance(value=R_multi_val, df=5.0, scale=R_multi_scale, constraint=False)
    print(R_multi)

    # Marker effect variance (to be estimated, no initial value)
    G_marker = VarianceCovariance(df=4.0, estimate_variance=True)
    print(G_marker)

    # Constrained multi-trait (diagonal)
    R_multi_constrained_val = np.array([[10.0, 0.0], [0.0, 5.0]])
    R_multi_constrained = VarianceCovariance(value=R_multi_constrained_val, df=5.0, constraint=True)
    print(R_multi_constrained)
