import pytest

from utils.protocol import assert_canonical_protocol


def _base_cfg():
    return {
        "data": {
            "target_type": "regression",
            "target_name": "log_return",
            "target_horizon": 1,
        },
        "training": {
            "val_start": "2016-01-01",
            "test_start": "2020-01-01",
        },
        "evaluation": {
            "transaction_cost_bps": 0,
            "backtest_policies": [1, 5],
            "primary_rebalance_freq": 1,
        },
        "experiment": {
            "protocol_version": "v1_thesis_core",
            "enforce_protocol": True,
        },
    }


def test_protocol_guard_passes_for_canonical_config():
    cfg = _base_cfg()
    assert_canonical_protocol(cfg)


def test_protocol_guard_rejects_misaligned_horizon():
    cfg = _base_cfg()
    cfg["data"]["target_horizon"] = 5
    with pytest.raises(ValueError):
        assert_canonical_protocol(cfg)
