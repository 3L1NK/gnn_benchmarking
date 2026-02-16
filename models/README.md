# Models

Model implementations are organized by family:

- `models/graph/`
  - `static_gnn.py` for GCN/GAT
  - `tgcn_static.py` for static-labeled TGCN baseline
  - `tgat_static.py` for static-labeled TGAT baseline
- `models/sequence/`
  - `lstm.py` for per-asset sequence baseline
- `models/baselines/`
  - `mlp.py` for baseline MLP components

Compatibility wrappers are kept at the old top-level module paths
(`models/gnn_model.py`, `models/tgcn_model.py`, `models/tgat_model.py`,
`models/lstm_model.py`, `models/baseline_mlp.py`).
