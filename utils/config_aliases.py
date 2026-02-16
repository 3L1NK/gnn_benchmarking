from __future__ import annotations

from pathlib import Path
from typing import Optional


LEGACY_CONFIG_ALIASES = {
    # Core runs
    "configs/xgb_raw.yaml": "configs/runs/core/xgb_raw.yaml",
    "configs/lstm.yaml": "configs/runs/core/lstm.yaml",
    "configs/xgb_node2vec.yaml": "configs/runs/core/xgb_node2vec_corr.yaml",
    "configs/gcn/gcn_corr_only.yaml": "configs/runs/core/gcn_corr_only.yaml",
    "configs/gcn/gcn_sector_only.yaml": "configs/runs/core/gcn_sector_only.yaml",
    "configs/gcn/gcn_granger_only.yaml": "configs/runs/core/gcn_granger_only.yaml",
    "configs/gcn/gcn_corr_sector_granger.yaml": "configs/runs/core/gcn_corr_sector_granger.yaml",
    "configs/gat/gat_corr_only.yaml": "configs/runs/core/gat_corr_only.yaml",
    "configs/gat/gat_sector_only.yaml": "configs/runs/core/gat_sector_only.yaml",
    "configs/gat/gat_granger_only.yaml": "configs/runs/core/gat_granger_only.yaml",
    "configs/gat/gat_corr_sector_granger.yaml": "configs/runs/core/gat_corr_sector_granger.yaml",
    "configs/tgcn_corr_only.yaml": "configs/runs/core/tgcn_static_corr_only.yaml",
    "configs/tgat_corr_only.yaml": "configs/runs/core/tgat_static_corr_only.yaml",
    # Exploratory
    "configs/granger_xgb.yaml": "configs/runs/exploratory/xgb_granger_smooth.yaml",
    "configs/graphlasso_linear.yaml": "configs/runs/exploratory/xgb_graphlasso_linear.yaml",
    "configs/graphlasso_xgb.yaml": "configs/runs/exploratory/xgb_graphlasso.yaml",
    "configs/_smoke_gcn_corr_only.yaml": "configs/runs/exploratory/smoke_gcn_corr_only.yaml",
    "configs/tgcn.yaml": "configs/runs/exploratory/tgcn_static.yaml",
    "configs/gcn/gcn.yaml": "configs/runs/exploratory/gcn_default.yaml",
    "configs/gcn/gcn_corr_granger.yaml": "configs/runs/exploratory/gcn_corr_granger.yaml",
    "configs/gcn/gcn_corr_only_sparse.yaml": "configs/runs/exploratory/gcn_corr_only_sparse.yaml",
    "configs/gcn/gcn_corr_sector.yaml": "configs/runs/exploratory/gcn_corr_sector.yaml",
    "configs/gcn/gcn_finetune.yaml": "configs/runs/exploratory/gcn_finetune.yaml",
    "configs/gcn/gcn_sector_only_sparse.yaml": "configs/runs/exploratory/gcn_sector_only_sparse.yaml",
    "configs/gcn/gcn_sparse_scaled.yaml": "configs/runs/exploratory/gcn_sparse_scaled.yaml",
    "configs/gcn/gcn_sparse_scaled_v2.yaml": "configs/runs/exploratory/gcn_sparse_scaled_v2.yaml",
    "configs/gat/gat.yaml": "configs/runs/exploratory/gat_default.yaml",
    "configs/gat/gat_corr_granger.yaml": "configs/runs/exploratory/gat_corr_granger.yaml",
    "configs/gat/gat_corr_sector.yaml": "configs/runs/exploratory/gat_corr_sector.yaml",
    "configs/gat/gat_finetune.yaml": "configs/runs/exploratory/gat_finetune.yaml",
    "configs/gat/gat_sparse_scaled.yaml": "configs/runs/exploratory/gat_sparse_scaled.yaml",
    # Ablation
    "configs/ablation/tgcn/ablation_A_increase_wcorr_topk.yaml": "configs/runs/ablation/tgcn/wcorr_topk.yaml",
    "configs/ablation/tgcn/ablation_B_loosen_max_edge_weight.yaml": "configs/runs/ablation/tgcn/loosen_max_edge_weight.yaml",
    "configs/ablation/tgcn/ablation_C_num_layers_2.yaml": "configs/runs/ablation/tgcn/num_layers_2.yaml",
    "configs/ablation/tgat/ablation_A_increase_wcorr_topk.yaml": "configs/runs/ablation/tgat/wcorr_topk.yaml",
    "configs/ablation/tgat/ablation_B_loosen_max_edge_weight.yaml": "configs/runs/ablation/tgat/loosen_max_edge_weight.yaml",
    "configs/ablation/tgat/ablation_C_num_layers_2.yaml": "configs/runs/ablation/tgat/num_layers_2.yaml",
    # Legacy model-zoo paths
    "configs/models/tgcn_base.yaml": "configs/templates/models/tgcn_static.yaml",
    "configs/models/tgat_base.yaml": "configs/templates/models/tgat_static.yaml",
    "configs/models/tgcn/base.yaml": "configs/templates/models/tgcn_static.yaml",
    "configs/models/tgat/base.yaml": "configs/templates/models/tgat_static.yaml",
    "configs/models/tgcn/experiment.yaml": "configs/runs/exploratory/tgcn_static.yaml",
    "configs/models/tgat/experiment.yaml": "configs/runs/core/tgat_static_corr_only.yaml",
    "configs/models/tgcn/ablation/01_wcorr_topk.yaml": "configs/runs/ablation/tgcn/wcorr_topk.yaml",
    "configs/models/tgcn/ablation/02_loosen_max_edge_weight.yaml": "configs/runs/ablation/tgcn/loosen_max_edge_weight.yaml",
    "configs/models/tgcn/ablation/03_num_layers_2.yaml": "configs/runs/ablation/tgcn/num_layers_2.yaml",
    "configs/models/tgat/ablation/01_wcorr_topk.yaml": "configs/runs/ablation/tgat/wcorr_topk.yaml",
    "configs/models/tgat/ablation/02_loosen_max_edge_weight.yaml": "configs/runs/ablation/tgat/loosen_max_edge_weight.yaml",
    "configs/models/tgat/ablation/03_num_layers_2.yaml": "configs/runs/ablation/tgat/num_layers_2.yaml",
}


def _normalize_path_key(path: str | Path) -> str:
    p = Path(path)
    key = p.as_posix()
    if key.startswith("./"):
        key = key[2:]
    return key


def resolve_config_alias(path: str | Path) -> Optional[Path]:
    key = _normalize_path_key(path)
    mapped = LEGACY_CONFIG_ALIASES.get(key)
    if not mapped and key.startswith("configs/legacy/models/"):
        mapped = LEGACY_CONFIG_ALIASES.get(key.replace("configs/legacy/models/", "configs/models/"))
    if not mapped:
        return None
    return Path(mapped)
