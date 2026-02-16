"""Run a single training step for GNN to inspect losses/gradients/predictions.

Usage:
  python3 scripts/debug_gnn_one_step.py --config configs/runs/core/gcn_corr_sector_granger.yaml
"""
import sys
from pathlib import Path
import numpy as np
import torch

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache import cache_key, cache_path, cache_load, cache_save
from utils.config_normalize import load_config as load_normalized_config
from trainers.train_gnn import _build_snapshots_and_targets, _split_snapshots_by_date
from torch_geometric.loader import DataLoader as GeoDataLoader
from models.graph.static_gnn import StaticGNN


def load_cfg(path):
    return load_normalized_config(path, REPO_ROOT)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cpu')
    print('Building/loading snapshots...')
    snapshots, feat_cols, dates = _build_snapshots_and_targets(cfg)
    print(f'Got {len(snapshots)} snapshots')
    if not snapshots:
        print('No snapshots; abort')
        return

    train_snaps, val_snaps, test_snaps = _split_snapshots_by_date(snapshots, dates, cfg, label="gnn_debug", debug=True)
    print(f'train {len(train_snaps)} val {len(val_snaps)} test {len(test_snaps)}')
    if not train_snaps:
        print('No train snaps; abort')
        return

    loader = GeoDataLoader(train_snaps, batch_size=min(4, len(train_snaps)), shuffle=True)
    batch = next(iter(loader))
    model = StaticGNN(
        gnn_type=cfg['model']['type'],
        input_dim=len(feat_cols),
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=min(cfg['model']['num_layers'], 2),
        dropout=cfg['model']['dropout'],
        heads=cfg['model'].get('heads', 1),
        use_residual=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training'].get('weight_decay', 0.0)))
    loss_fn = torch.nn.MSELoss()

    batch = batch.to(device)
    model.train()
    optimizer.zero_grad()
    logits = model(batch.x, batch.edge_index, batch.edge_weight)
    mask = batch.valid_mask & torch.isfinite(batch.y)
    print(f'mask sum: {int(mask.sum())} / {batch.y.shape[0]}')
    if int(mask.sum()) == 0:
        print('No valid nodes in batch to compute loss. Exiting.')
        return
    loss = loss_fn(logits[mask], batch.y[mask])
    print('Loss before backward:', float(loss.item()))
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.norm().item())
    print('Sum of parameter grad norms:', grad_norm)
    # show sample preds vs targets
    preds = logits[mask].detach().cpu().numpy()[:10]
    targets = batch.y[mask].detach().cpu().numpy()[:10]
    print('Sample preds vs targets (first 10):')
    for p, t in zip(preds, targets):
        print(f'  pred={float(p):.6f} target={float(t):.6f}')

if __name__ == '__main__':
    main()
