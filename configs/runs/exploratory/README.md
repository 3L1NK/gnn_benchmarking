# Exploratory Runs

Non-core experiments, smoke tests, and older variants live here.

These configs are useful for diagnostics and appendix analyses, but are not part of
the canonical thesis comparison matrix.

Useful non-neural graph baselines:

- `xgb_graphlasso_linear.yaml`: Graphical Lasso adjacency + graph-smoothed features + Ridge head.
- `xgb_graphlasso.yaml`: Graphical Lasso adjacency + graph-smoothed features + XGBoost head.
- `xgb_granger_smooth.yaml`: Granger adjacency + graph-smoothed features + XGBoost head.

Run examples:

```bash
python train.py --config configs/runs/exploratory/xgb_graphlasso_linear.yaml
python train.py --config configs/runs/exploratory/xgb_graphlasso.yaml
python train.py --config configs/runs/exploratory/xgb_granger_smooth.yaml
```
