"""Seeded synthetic transaction graph with planted fraud rings.

Nodes are accounts; edges are money transfers. Legitimate accounts transact
sparsely and at random (a sparse Erdos-Renyi-style background). Fraud rings are
small, densely interconnected clusters of colluding accounts that cycle money
among themselves — the structural signature graph methods catch but a per-row
tabular model misses.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def make(n_legit: int = 1800, n_rings: int = 12, ring_size: int = 8,
         seed: int = 42):
    rng = np.random.default_rng(seed)
    n_fraud = n_rings * ring_size
    n = n_legit + n_fraud
    labels = np.zeros(n, dtype=int)
    labels[n_legit:] = 1

    edges = []

    # Per-transaction amounts deliberately OVERLAP between legit and fraud, so
    # a model using transaction amount alone cannot separate them — only the
    # graph structure (dense cycling) reveals the rings.
    def amount():
        return float(rng.gamma(2, 40))

    # Legitimate background: sparse random transfers.
    for _ in range(n_legit * 2):
        a, b = rng.integers(0, n_legit, 2)
        if a != b:
            edges.append((int(a), int(b), amount()))

    # Fraud rings: dense intra-ring cycling + a few legit "mule" links out.
    for r in range(n_rings):
        members = list(range(n_legit + r * ring_size,
                             n_legit + (r + 1) * ring_size))
        for a in members:
            for b in members:
                if a != b and rng.random() < 0.6:
                    edges.append((a, b, amount()))
        # a couple of cash-out links to legit accounts (same amount profile)
        for _ in range(2):
            edges.append((int(rng.choice(members)),
                          int(rng.integers(0, n_legit)), amount()))

    edge_df = pd.DataFrame(edges, columns=["src", "dst", "amount"])
    node_df = pd.DataFrame({"node": np.arange(n), "is_fraud": labels})
    return node_df, edge_df


def main() -> None:
    DATA.mkdir(exist_ok=True)
    nodes, edges = make()
    nodes.to_csv(DATA / "nodes.csv", index=False)
    edges.to_csv(DATA / "edges.csv", index=False)
    print(f"Nodes: {len(nodes)} ({nodes.is_fraud.sum()} fraud)  "
          f"Edges: {len(edges)}  -> {DATA}")


if __name__ == "__main__":
    main()
