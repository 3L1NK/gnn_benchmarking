from pathlib import Path

import pandas as pd

from utils.config_normalize import load_config
from utils.targets import build_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_target_policy_matches_core_configs():
    cfg_xgb = load_config(PROJECT_ROOT / "configs/runs/core/xgb_raw.yaml", PROJECT_ROOT)
    cfg_lstm = load_config(PROJECT_ROOT / "configs/runs/core/lstm.yaml", PROJECT_ROOT)
    cfg_gcn = load_config(PROJECT_ROOT / "configs/runs/core/gcn_corr_only.yaml", PROJECT_ROOT)

    expected = {
        "target_type": "regression",
        "target_name": "log_return",
        "target_horizon": 1,
    }
    for cfg in (cfg_xgb, cfg_lstm, cfg_gcn):
        data_cfg = cfg.get("data", {})
        got = {
            "target_type": data_cfg.get("target_type"),
            "target_name": data_cfg.get("target_name"),
            "target_horizon": int(data_cfg.get("target_horizon")),
        }
        assert got == expected


def test_build_target_uses_next_day_log_return_shift():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "log_ret_1d": [0.01, 0.02, 0.03, -0.01, -0.02, -0.03],
        }
    )
    cfg = {
        "data": {
            "target_type": "regression",
            "target_name": "log_return",
            "target_horizon": 1,
        }
    }
    out, target_col = build_target(df, cfg, target_col="target")
    assert target_col == "target"
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Last row per ticker is dropped after shift(-1), so we keep two rows each.
    expected = [0.02, 0.03, -0.02, -0.03]
    assert out["target"].round(10).tolist() == expected
