import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class GenotypeData:
    """Minimal container for genotype matrices."""

    obsID: Sequence
    genotypes: np.ndarray
    isGRM: bool = False
    output_genotypes: Optional[np.ndarray] = None
    nObs: int = 0


@dataclass
class MME:
    """Simplified representation of the MME structure used for alignment."""

    obsID: Sequence
    output_ID: Sequence
    M: List[GenotypeData] = field(default_factory=list)


def get_column_ref(X: np.ndarray) -> List[np.ndarray]:
    """Return views for each column of X without copying."""
    return [X[:, i] for i in range(X.shape[1])]


def center(X: np.ndarray) -> np.ndarray:
    """Center columns of ``X`` in place and return the column means."""
    col_means = X.mean(axis=0)
    X -= col_means
    return col_means


def get_XpRinvX(X: np.ndarray, Rinv: Optional[np.ndarray] = None) -> List[float]:
    """Return diagonal values of ``X' * diag(Rinv) * X`` efficiently."""
    if Rinv is None:
        return (X * X).sum(axis=0).tolist()
    return (X * Rinv[:, None] * X).sum(axis=0).tolist()


def get_column_blocks_ref(X: np.ndarray, fast_blocks: Optional[List[int]] = None) -> List[np.ndarray]:
    if not fast_blocks:
        return []
    refs = []
    for i, start in enumerate(fast_blocks):
        end = fast_blocks[i + 1] - 1 if i != len(fast_blocks) - 1 else X.shape[1]
        refs.append(X[:, start:end])
    return refs


@dataclass
class GibbsMats:
    X: np.ndarray
    nrows: int
    ncols: int
    xArray: List[np.ndarray]
    xRinvArray: List[np.ndarray]
    xpRinvx: np.ndarray
    XArray: List[np.ndarray]
    XRinvArray: List[np.ndarray]
    XpRinvX: List[np.ndarray]

    @classmethod
    def build(cls, X: np.ndarray, Rinv: np.ndarray, fast_blocks: Optional[List[int]] = None) -> "GibbsMats":
        nrows, ncols = X.shape
        xArray = get_column_ref(X)
        xpRinvx = get_XpRinvX(X, Rinv)
        if np.allclose(Rinv, np.ones_like(Rinv)):
            xRinvArray = xArray
        else:
            xRinvArray = [col * Rinv for col in xArray]
        if fast_blocks:
            XArray = get_column_blocks_ref(X, fast_blocks)
            XRinvArray = [X_block * Rinv[:, None] for X_block in XArray]
            XpRinvX = [XR.T @ X_block for XR, X_block in zip(XRinvArray, XArray)]
        else:
            XArray = XRinvArray = XpRinvX = []
        return cls(X, nrows, ncols, xArray, xRinvArray, np.array(xpRinvx), XArray, XRinvArray, XpRinvX)


def align_genotypes(mme: MME, output_heritability: bool = False, single_step_analysis: bool = False) -> None:
    """Align genotype matrices with phenotype order."""
    if single_step_analysis:
        return

    if mme.output_ID:
        for Mi in mme.M:
            if list(mme.output_ID) != list(Mi.obsID):
                Zo = mkmat_incidence_factor(mme.output_ID, Mi.obsID)
                Mi.output_genotypes = Zo @ Mi.genotypes
            else:
                Mi.output_genotypes = Mi.genotypes
            if Mi.isGRM:
                Z = mkmat_incidence_factor(mme.obsID, Mi.obsID)
                Mi.output_genotypes = Mi.output_genotypes @ Z.T

    if list(mme.obsID) != list(mme.M[0].obsID):
        for Mi in mme.M:
            Z = mkmat_incidence_factor(mme.obsID, Mi.obsID)
            genotypes = Z @ Mi.genotypes
            if Mi.isGRM:
                genotypes = genotypes @ Z.T
            Mi.genotypes = genotypes
            Mi.obsID = mme.obsID
            Mi.nObs = len(mme.obsID)


def mkmat_incidence_factor(yID: Sequence, uID: Sequence) -> np.ndarray:
    """Create an incidence matrix ``Z`` so that ``y = Z @ u`` reorders ``u`` to ``y``."""
    index = {id_: i for i, id_ in enumerate(uID)}
    try:
        cols = [index[id_] for id_ in yID]
    except KeyError as e:
        raise ValueError(f"{e.args[0]} is not found!")
    Z = np.zeros((len(yID), len(uID)), dtype=float)
    Z[np.arange(len(yID)), cols] = 1.0
    return Z
