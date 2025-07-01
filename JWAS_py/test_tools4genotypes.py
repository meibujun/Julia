import numpy as np
from JWAS_py import (
    get_column_ref,
    center,
    get_XpRinvX,
    mkmat_incidence_factor,
    GenotypeData,
    MME,
    align_genotypes,
)


def test_center():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    means = center(X)
    assert np.allclose(means, [2.0, 3.0])
    assert np.allclose(X, [[-1.0, -1.0], [1.0, 1.0]])


def test_get_XpRinvX():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    r = np.array([1.0, 1.0])
    vals = get_XpRinvX(X, r)
    assert np.allclose(vals, [10.0, 20.0])


def test_mkmat_incidence_factor():
    yID = [1, 2]
    uID = [2, 1]
    Z = mkmat_incidence_factor(yID, uID)
    assert np.array_equal(Z, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_align_genotypes():
    mme = MME(obsID=[1, 2], output_ID=[1, 2])
    geno = np.array([[1.0, 2.0], [3.0, 4.0]])
    g1 = GenotypeData(obsID=[2, 1], genotypes=geno.copy())
    mme.M.append(g1)
    align_genotypes(mme)
    assert np.array_equal(g1.output_genotypes, np.array([[3.0, 4.0], [1.0, 2.0]]))


if __name__ == "__main__":
    test_center()
    test_get_XpRinvX()
    test_mkmat_incidence_factor()
    test_align_genotypes()
    print("All tests passed.")
