# coding: utf-8
"""Compute relationship matrix and coefficient for a pedigree.

Python version of the relationship coefficient example. It builds the
relationship matrix (A matrix) recursively and computes coefficients between
individuals in a pedigree.
"""

from typing import Dict, Tuple, Optional, List

Pedigree = Dict[str, Tuple[Optional[str], Optional[str]]]


def _topological_order(pedigree: Pedigree) -> List[str]:
    """Return individuals in parent-first order for matrix construction."""
    visited = set()
    order: List[str] = []

    def dfs(ind: str):
        if ind in visited:
            return
        visited.add(ind)
        sire, dam = pedigree[ind]
        if sire is not None:
            dfs(sire)
        if dam is not None:
            dfs(dam)
        order.append(ind)

    for ind in pedigree:
        dfs(ind)

    return order


def relationship_matrix(pedigree: Pedigree):
    individuals: List[str] = _topological_order(pedigree)
    idx = {ind: i for i, ind in enumerate(individuals)}
    n = len(individuals)
    A = [[0.0] * n for _ in range(n)]

    for i, ind in enumerate(individuals):
        sire, dam = pedigree[ind]
        if sire is None and dam is None:
            A[i][i] = 1.0
        else:
            val = 1.0
            if sire is not None and dam is not None:
                val += 0.5 * A[idx[sire]][idx[dam]]
            A[i][i] = val

        for j in range(i):
            s = 0.0 if sire is None else A[idx[sire]][j]
            d = 0.0 if dam is None else A[idx[dam]][j]
            A[i][j] = 0.5 * (s + d)
            A[j][i] = A[i][j]

    return A, idx


def relationship_coefficient(pedigree: Pedigree, x: str, y: str) -> float:
    A, idx = relationship_matrix(pedigree)
    return A[idx[x]][idx[y]]


if __name__ == "__main__":
    pedigree = {
        "F": (None, None),
        "M1": (None, None),
        "M2": (None, None),
        "B": ("F", "M1"),
        "C": ("F", "M2"),
    }

    A, idx = relationship_matrix(pedigree)
    print("Relationship matrix:")
    for row in A:
        print([round(x, 2) for x in row])
    print("Relationship coefficient between B and C:", A[idx["B"]][idx["C"]])
