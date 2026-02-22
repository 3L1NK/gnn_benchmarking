# Configs

This folder is organized by intent, not by model implementation detail.

- `templates/`
  - shared protocol defaults and reusable model templates
- `runs/core/`
  - canonical thesis matrix runs used for main tables/plots
- `runs/ablation/`
  - targeted ablation experiments
- `runs/exploratory/`
  - additional non-core experiments and smoke configs
- `legacy/`
  - archived historical configs kept for reference

Use:

```bash
python train.py --config configs/runs/core/gcn_corr_only.yaml
```

Compatibility:

- Older paths are resolved through alias mapping in `utils/config_aliases.py`.
