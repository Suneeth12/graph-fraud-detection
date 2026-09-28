#!/usr/bin/env bash
# Generate the transaction graph, detect rings, and compare graph vs tabular.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt
python3 src/generate_graph.py
python3 src/detect.py
echo ""
echo "See reports/results.json"
