from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List

import pandas as pd


CANONICAL_PROTOCOL_VERSION = "v1_thesis_core"
CANONICAL_TARGET_TYPE = "regression"
CANONICAL_TARGET_NAME = "log_return"
CANONICAL_TARGET_HORIZON = 1
CANONICAL_VAL_START = "2016-01-01"
CANONICAL_TEST_START = "2020-01-01"
CANONICAL_BACKTEST_POLICIES = (1, 5)
CANONICAL_PRIMARY_REBALANCE = 1
CANONICAL_TRANSACTION_COST_BPS = 0.0


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_version: str
    target_type: str
    target_name: str
    target_horizon: int
    split_val_start: str
    split_test_start: str
    backtest_policies: List[int]
    primary_rebalance_freq: int
    baseline_version: str

    @property
    def split_train_end(self) -> str:
        val = pd.to_datetime(self.split_val_start)
        return str((val - pd.Timedelta(days=1)).date())

    @property
    def target_policy_hash(self) -> str:
        payload = {
            "protocol_version": self.protocol_version,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "target_horizon": self.target_horizon,
            "split_val_start": self.split_val_start,
            "split_test_start": self.split_test_start,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def as_result_fields(self, rebalance_freq: int) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "split_train_end": self.split_train_end,
            "split_val_start": self.split_val_start,
            "split_test_start": self.split_test_start,
            "rebalance_freq": int(rebalance_freq),
            "baseline_version": self.baseline_version,
            "target_policy_hash": self.target_policy_hash,
        }


def protocol_from_config(config: Dict[str, Any], baseline_version: str = "") -> ProtocolSpec:
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    exp_cfg = config.get("experiment", {})

    policies = [int(x) for x in eval_cfg.get("backtest_policies", [1, 5])]
    policies = [p for p in policies if p >= 1]
    if not policies:
        policies = [1, 5]
    primary = int(eval_cfg.get("primary_rebalance_freq", policies[0]))
    if primary not in policies:
        primary = policies[0]

    return ProtocolSpec(
        protocol_version=str(exp_cfg.get("protocol_version", "v1_thesis_core")),
        target_type=str(data_cfg.get("target_type", "regression")),
        target_name=str(data_cfg.get("target_name", "log_return")),
        target_horizon=int(data_cfg.get("target_horizon", 1)),
        split_val_start=str(train_cfg.get("val_start", "")),
        split_test_start=str(train_cfg.get("test_start", "")),
        backtest_policies=policies,
        primary_rebalance_freq=primary,
        baseline_version=str(baseline_version),
    )


def canonical_protocol_violations(config: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable violations for v1_thesis_core protocol."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    exp_cfg = config.get("experiment", {})
    violations: List[str] = []

    if str(exp_cfg.get("protocol_version", "")) != CANONICAL_PROTOCOL_VERSION:
        return violations

    target_type = str(data_cfg.get("target_type", ""))
    if target_type != CANONICAL_TARGET_TYPE:
        violations.append(f"data.target_type must be '{CANONICAL_TARGET_TYPE}', got '{target_type}'")

    target_name = str(data_cfg.get("target_name", ""))
    if target_name != CANONICAL_TARGET_NAME:
        violations.append(f"data.target_name must be '{CANONICAL_TARGET_NAME}', got '{target_name}'")

    horizon = int(data_cfg.get("target_horizon", -1))
    if horizon != CANONICAL_TARGET_HORIZON:
        violations.append(f"data.target_horizon must be {CANONICAL_TARGET_HORIZON}, got {horizon}")

    val_start = str(train_cfg.get("val_start", ""))
    if val_start != CANONICAL_VAL_START:
        violations.append(f"training.val_start must be '{CANONICAL_VAL_START}', got '{val_start}'")

    test_start = str(train_cfg.get("test_start", ""))
    if test_start != CANONICAL_TEST_START:
        violations.append(f"training.test_start must be '{CANONICAL_TEST_START}', got '{test_start}'")

    tx_cost = float(eval_cfg.get("transaction_cost_bps", float("nan")))
    if tx_cost != CANONICAL_TRANSACTION_COST_BPS:
        violations.append(
            f"evaluation.transaction_cost_bps must be {CANONICAL_TRANSACTION_COST_BPS:g}, got {tx_cost:g}"
        )

    policies = tuple(int(x) for x in eval_cfg.get("backtest_policies", []))
    if policies != CANONICAL_BACKTEST_POLICIES:
        violations.append(
            f"evaluation.backtest_policies must be {list(CANONICAL_BACKTEST_POLICIES)}, got {list(policies)}"
        )

    primary = int(eval_cfg.get("primary_rebalance_freq", -1))
    if primary != CANONICAL_PRIMARY_REBALANCE:
        violations.append(
            f"evaluation.primary_rebalance_freq must be {CANONICAL_PRIMARY_REBALANCE}, got {primary}"
        )

    return violations


def assert_canonical_protocol(config: Dict[str, Any]) -> None:
    """
    Validate canonical thesis protocol.
    Can be bypassed by setting experiment.enforce_protocol=false.
    """
    exp_cfg = config.get("experiment", {})
    enforce = bool(exp_cfg.get("enforce_protocol", True))
    if not enforce:
        return
    violations = canonical_protocol_violations(config)
    if violations:
        joined = "\n - ".join(violations)
        raise ValueError(f"Canonical protocol validation failed:\n - {joined}")
