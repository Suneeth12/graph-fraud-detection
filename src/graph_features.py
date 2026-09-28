"""Graph feature engineering with scipy.sparse (no NetworkX dependency).

For every account we compute structural features that expose collusion:
  - degree / weighted degree (transaction count and volume)
  - local clustering coefficient (do my counterparties transact with each other?)
  - PageRank (power iteration)
  - connected-component id and size (ring detection signal)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def build_adjacency(n: int, edges) -> csr_matrix:
    src = edges["src"].values
    dst = edges["dst"].values
    w = edges["amount"].values
    # Undirected, symmetric weighted adjacency.
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([w, w])
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A


def pagerank(A: csr_matrix, damping: float = 0.85, iters: int = 50) -> np.ndarray:
    n = A.shape[0]
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    deg[deg == 0] = 1
    M = (A > 0).astype(float)
    M = M.multiply(1.0 / deg[:, None]).tocsr()
    pr = np.full(n, 1.0 / n)
    teleport = (1 - damping) / n
    for _ in range(iters):
        pr = teleport + damping * (M.T @ pr)
    return pr


def clustering_coefficient(A: csr_matrix) -> np.ndarray:
    B = (A > 0).astype(int)
    deg = np.asarray(B.sum(axis=1)).ravel()
    # triangles through each node = diag(B^3) / 2
    B3_diag = (B @ B).multiply(B).sum(axis=1)
    tri = np.asarray(B3_diag).ravel() / 2.0
    denom = deg * (deg - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = np.where(denom > 0, 2 * tri / denom, 0.0)
    return cc


def compute(n: int, edges) -> "np.ndarray | dict":
    A = build_adjacency(n, edges)
    B = (A > 0).astype(int)
    degree = np.asarray(B.sum(axis=1)).ravel()
    weighted = np.asarray(A.sum(axis=1)).ravel()
    cc = clustering_coefficient(A)
    pr = pagerank(A)
    n_comp, comp = connected_components(A, directed=False)
    comp_size = np.bincount(comp)[comp]

    feats = np.column_stack([degree, weighted, cc, pr, comp_size])
    return {
        "features": feats,
        "names": ["degree", "weighted_degree", "clustering", "pagerank",
                  "component_size"],
        "component": comp,
        "n_components": n_comp,
    }
