# Core Runs

These configs define the canonical thesis comparison matrix.

Included runs:
- non-graph: `xgb_raw.yaml`, `lstm.yaml`
- graph feature baseline: `xgb_node2vec_corr.yaml`
- static GNN edge ablations: `gcn_*`, `gat_*`
- static temporal-labeled baselines: `tgcn_static_corr_only.yaml`, `tgat_static_corr_only.yaml`

Run all with:

```bash
python scripts/run_core_thesis_matrix.py --fresh-results --with-report
```
