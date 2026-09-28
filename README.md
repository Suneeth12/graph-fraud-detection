# 🕸️ Graph ML — Fraud Ring Detection

Detects **collusion rings** in a transaction graph — the kind of organized fraud
a per-transaction tabular model misses because no single transaction looks
suspicious. The signal lives in the **structure**, not the rows.

## The core idea
Fraudsters cycle money among a tight cluster of accounts. Each individual
transfer has a perfectly ordinary amount, so an amount-based model is blind. But
the *graph* shows a dense near-clique — and graph features expose it instantly.
This project quantifies exactly that gap.

## What this project demonstrates
- **Graph feature engineering** — degree, weighted degree, **local clustering
  coefficient**, **PageRank** (power iteration), and connected-component size,
  all on `scipy.sparse` (no NetworkX dependency).
- **Unsupervised ring detection** — high-clustering accounts grouped into rings
  via a clustering-thresholded subgraph (handles "mule" cash-out links that
  merge rings into the giant component).
- **Graph vs tabular, measured** — a classifier on graph features vs. one on
  transaction amount alone, on the *same* nodes.

## Demo

```text
$ python3 src/generate_graph.py
Nodes: 1896 (96 fraud)  Edges: 4038

$ python3 src/detect.py
Connected components: 33
Ring detection: 12 rings, 96 accounts flagged (precision=1.00, recall=1.00)

Node classification (held-out):
  tabular_only(amount)   ROC-AUC=0.557  AUPRC=0.058
  graph_features         ROC-AUC=1.000  AUPRC=1.000

Graph features add +0.942 AUPRC over amount-only
```

The headline: transaction **amount alone is useless** here (AUPRC 0.058 — barely
above the 5% fraud base rate), while **graph structure recovers every ring**
(precision and recall 1.00; AUPRC 1.00). That is the entire argument for graph
methods in fraud, made concrete.

## Why amounts are uninformative *by design*
The generator draws legit and fraud transaction amounts from the **same**
distribution. The only difference between a fraud account and a legit one is who
they transact with — i.e. the graph. This is what makes the comparison honest.

## Project structure
```
graph-fraud-detection/
├── data/                   # nodes.csv, edges.csv (generated)
├── src/
│   ├── generate_graph.py   # transaction graph w/ planted rings
│   ├── graph_features.py   # degree, clustering, PageRank, components (scipy)
│   └── detect.py           # ring detection + graph-vs-tabular classification
├── reports/results.json
├── requirements.txt
├── torun.txt
└── license.md
```

**Production path:** swap the RandomForest on hand-built features for a **Graph
Neural Network** (`torch-geometric`, noted in requirements) to learn the
structural features end-to-end.

## Run it
```bash
./run.sh        # or see torun.txt
```
