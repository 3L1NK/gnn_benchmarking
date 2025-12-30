Model config layout

- `configs/models/<model>/base.yaml`    : model-specific defaults (includes `configs/default.yaml`)
- `configs/models/<model>/experiment.yaml` : top-level experiment file (duplicate of previous top-level `configs/<model>.yaml`)
- `configs/models/<model>/ablation/*.yaml` : ablation YAMLs; each file includes `base.yaml` and the shared `default.yaml`.

This groups model configs and ablations to reduce top-level clutter. Use `scripts/run_ablation.py` with the model's ablation folder or the existing `configs/ablation/<model>/` (both supported).
