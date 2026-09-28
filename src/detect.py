"""Fraud-ring detection + node classification.

Two complementary outputs:
  1. Unsupervised ring detection: small, dense connected components with high
     average clustering are flagged as candidate fraud rings.
  2. Supervised node classification: a classifier on graph features, compared
     against a tabular-only baseline (transaction volume alone) to quantify how
     much the *graph structure* adds.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from scipy.sparse.csgraph import connected_components

from graph_features import clustering_coefficient, build_adjacency, compute

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def detect_rings(n, edges, min_clust: float = 0.25, max_size: int = 30):
    """Flag high-clustering accounts, then group them into rings.

    Cash-out 'mule' links merge rings into the giant component, so raw
    connected components don't isolate them. Instead we keep only high-
    clustering nodes (the near-clique members), restrict the graph to the edges
    among them, and take the connected components of that subgraph.
    """
    A = build_adjacency(n, edges)
    cc = clustering_coefficient(A)
    candidates = np.where(cc > min_clust)[0]
    if len(candidates) == 0:
        return []
    cand_set = set(candidates.tolist())
    mask = edges["src"].isin(cand_set) & edges["dst"].isin(cand_set)
    sub = edges[mask]
    sub_A = build_adjacency(n, sub)
    _, comp = connected_components(sub_A, directed=False)
    rings = []
    for c in np.unique(comp[candidates]):
        members = np.intersect1d(np.where(comp == c)[0], candidates)
        if 2 < len(members) <= max_size:
            rings.append(members)
    return rings


def main() -> None:
    if not (DATA / "nodes.csv").exists():
        raise SystemExit("Run: python3 src/generate_graph.py")
    nodes = pd.read_csv(DATA / "nodes.csv")
    edges = pd.read_csv(DATA / "edges.csv")
    n = len(nodes)
    y = nodes["is_fraud"].values

    g = compute(n, edges)
    feats, names = g["features"], g["names"]

    # Per-account mean transaction amount — the feature a non-graph, tabular
    # model would have. Amounts overlap by construction, so it is a weak signal.
    A = build_adjacency(n, edges)
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    mean_amount = np.asarray(A.sum(axis=1)).ravel() / np.maximum(deg, 1)

    # --- 1. Unsupervised ring detection ---
    rings = detect_rings(n, edges)
    flagged = np.concatenate(rings) if rings else np.array([], dtype=int)
    ring_precision = y[flagged].mean() if len(flagged) else 0.0
    ring_recall = y[flagged].sum() / y.sum() if len(flagged) else 0.0

    # --- 2. Supervised classification: graph vs tabular-only ---
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.3,
                                      random_state=42, stratify=y)

    def evaluate(X):
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=42)
        clf.fit(X[idx_tr], y[idx_tr])
        p = clf.predict_proba(X[idx_te])[:, 1]
        return {"roc_auc": round(roc_auc_score(y[idx_te], p), 3),
                "auprc": round(average_precision_score(y[idx_te], p), 3)}

    amount_only = mean_amount.reshape(-1, 1)       # what a tabular model sees
    graph_full = feats                             # structural features
    results = {"tabular_only(amount)": evaluate(amount_only),
               "graph_features": evaluate(graph_full)}

    REPORTS.mkdir(exist_ok=True)
    summary = {
        "ring_detection": {
            "rings_found": len(rings),
            "accounts_flagged": int(len(flagged)),
            "precision": round(float(ring_precision), 3),
            "recall": round(float(ring_recall), 3),
        },
        "classification": results,
        "feature_names": names,
    }
    (REPORTS / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"Connected components: {g['n_components']}")
    print(f"Ring detection: {len(rings)} rings, {len(flagged)} accounts flagged "
          f"(precision={ring_precision:.2f}, recall={ring_recall:.2f})")
    print("\nNode classification (held-out):")
    for name, m in results.items():
        print(f"  {name:22s} ROC-AUC={m['roc_auc']:.3f}  AUPRC={m['auprc']:.3f}")
    lift = results["graph_features"]["auprc"] - results["tabular_only(amount)"]["auprc"]
    print(f"\nGraph features add +{lift:.3f} AUPRC over amount-only "
          f"-> reports/results.json")


if __name__ == "__main__":
    main()
