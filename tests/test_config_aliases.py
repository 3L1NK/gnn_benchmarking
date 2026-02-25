from pathlib import Path

from utils.config_normalize import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_config_path_resolves_to_new_core_path():
    cfg_old = load_config("configs/gcn/gcn_corr_only.yaml", PROJECT_ROOT)
    cfg_new = load_config("configs/runs/core/gcn_corr_only.yaml", PROJECT_ROOT)
    assert cfg_old["model"]["type"] == cfg_new["model"]["type"] == "gcn"
    assert cfg_old["evaluation"]["out_dir"] == cfg_new["evaluation"]["out_dir"]


def test_granger_aliases_are_normalized():
    cfg = load_config("configs/runs/core/gcn_granger_only.yaml", PROJECT_ROOT)
    assert int(cfg["graph"]["granger_top_k"]) == 50
    assert int(cfg["granger"]["max_lag"]) == 5
    assert float(cfg["granger"]["p_threshold"]) == 0.01
