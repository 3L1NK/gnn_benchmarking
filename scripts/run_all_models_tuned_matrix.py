#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_normalize import load_config


CORE_MATRIX_CONFIGS = [
    "configs/runs/core/xgb_raw.yaml",
    "configs/runs/core/mlp.yaml",
    "configs/runs/core/lstm.yaml",
    "configs/runs/core/xgb_node2vec_corr.yaml",
    "configs/runs/core/xgb_graphlasso_linear.yaml",
    "configs/runs/core/gcn_corr_only.yaml",
    "configs/runs/core/gcn_sector_only.yaml",
    "configs/runs/core/gcn_granger_only.yaml",
    "configs/runs/core/gcn_corr_sector_granger.yaml",
    "configs/runs/core/gat_corr_only.yaml",
    "configs/runs/core/gat_sector_only.yaml",
    "configs/runs/core/gat_granger_only.yaml",
    "configs/runs/core/gat_corr_sector_granger.yaml",
    "configs/runs/core/tgcn_static_corr_only.yaml",
    "configs/runs/core/tgat_static_corr_only.yaml",
]


def _budget_profile(name: str) -> dict:
    key = name.strip().lower()
    if key == "quick":
        return {"gnn_trials": 6, "xgb_trials": 10, "lstm_trials": 10, "mlp_trials": 12}
    if key == "heavy":
        return {"gnn_trials": 20, "xgb_trials": 30, "lstm_trials": 24, "mlp_trials": 24}
    return {"gnn_trials": 12, "xgb_trials": 24, "lstm_trials": 18, "mlp_trials": 18}


def _xgb_tuning_grid(run_tag: str) -> dict:
    if run_tag == "xgb_node2vec_corr":
        return {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.03],
            "n_estimators": [400, 600, 800],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [0.0, 1.0, 5.0],
        }
    return {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.03, 0.05],
        "n_estimators": [200, 400, 600],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [0.0, 1.0, 5.0],
    }


def _gnn_tuning_grid(model_type: str) -> dict:
    base = {
        "hidden_dim": [64, 128, 192],
        "num_layers": [2, 3],
        "dropout": [0.1, 0.2],
        "lr": [3e-4, 5e-4, 1e-3],
        "weight_decay": [1e-6, 1e-5, 5e-5],
        "gradient_clip": [0.8, 1.0, 1.2],
    }
    if model_type in {"gat", "tgat_static"}:
        base["heads"] = [1, 2, 4]
        base["attn_dropout"] = [0.0, 0.1, 0.2]
    return base


def _lstm_tuning_grid() -> dict:
    return {
        "hidden_dim": [32, 64, 96, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [3e-4, 5e-4, 1e-3],
    }


def _mlp_tuning_grid() -> dict:
    return {
        "hidden_dim_1": [64, 128, 192],
        "hidden_dim_2": [32, 64, 96],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [3e-4, 5e-4, 1e-3],
        "weight_decay": [0.0, 1e-5, 5e-5],
    }


def _inject_tuning(cfg: dict, *, run_tag: str, profile: dict) -> dict:
    tuned = deepcopy(cfg)
    model = tuned.get("model", {}) or {}
    family = str(model.get("family", "")).lower()
    model_type = str(model.get("type", "")).lower()
    tuned["tuning"] = {
        "enabled": True,
        "objective": "val_backtest_sharpe_annualized",
        "sample_mode": "random",
        "seed": 42,
    }

    if family == "gnn":
        tuned["tuning"]["max_trials"] = int(profile["gnn_trials"])
        tuned["tuning"]["tune_max_epochs"] = 12
        tuned["tuning"]["tune_patience"] = 4
        tuned["tuning"]["param_grid"] = _gnn_tuning_grid(model_type)
    elif family == "xgboost":
        tuned["tuning"]["max_trials"] = int(profile["xgb_trials"])
        tuned["tuning"]["param_grid"] = _xgb_tuning_grid(run_tag)
    elif family == "lstm":
        tuned["tuning"]["max_trials"] = int(profile["lstm_trials"])
        tuned["tuning"]["tune_max_epochs"] = 12
        tuned["tuning"]["tune_patience"] = 4
        tuned["tuning"]["param_grid"] = _lstm_tuning_grid()
    elif family == "mlp":
        tuned["tuning"]["max_trials"] = int(profile["mlp_trials"])
        tuned["tuning"]["tune_max_epochs"] = 12
        tuned["tuning"]["tune_patience"] = 4
        tuned["tuning"]["param_grid"] = _mlp_tuning_grid()
    else:
        tuned["tuning"]["max_trials"] = 8
        tuned["tuning"]["param_grid"] = {}
    return tuned


def _retag_paths(cfg: dict, *, src_tag: str) -> dict:
    tuned = deepcopy(cfg)
    tuned_tag = f"{src_tag}_tuned_all"
    tuned["experiment_name"] = tuned_tag
    eval_cfg = dict(tuned.get("evaluation", {}) or {})
    eval_cfg["results_path"] = "results/results_tuned_all.jsonl"
    eval_cfg["out_dir"] = f"experiments_tuned_all/{src_tag}"
    tuned["evaluation"] = eval_cfg
    return tuned


def _write_generated_configs(profile_name: str, profile: dict) -> list[Path]:
    out_dir = REPO_ROOT / "configs" / "runs" / "tuned_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for src in CORE_MATRIX_CONFIGS:
        src_path = REPO_ROOT / src
        cfg = load_config(str(src_path), REPO_ROOT)
        src_tag = str(cfg.get("experiment_name", src_path.stem))
        cfg = _retag_paths(cfg, src_tag=src_tag)
        cfg = _inject_tuning(cfg, run_tag=src_tag, profile=profile)
        cfg.setdefault("experiment", {})
        cfg["experiment"]["protocol_version"] = "v1_thesis_core"
        cfg["experiment"]["enforce_protocol"] = True
        cfg["tuning"]["profile"] = profile_name

        out_path = out_dir / f"{src_tag}_tuned_all.yaml"
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        generated.append(out_path)
    return generated


def _reset_outputs(report_out: str) -> None:
    results_file = REPO_ROOT / "results" / "results_tuned_all.jsonl"
    if results_file.exists():
        results_file.unlink()
    report_dir = REPO_ROOT / report_out
    if report_dir.exists():
        for p in report_dir.glob("*"):
            if p.is_file():
                p.unlink()


def run_matrix(config_paths: list[Path], rebuild_cache: bool = False) -> int:
    for cfg in config_paths:
        cmd = [sys.executable, "train.py", "--config", str(cfg)]
        if rebuild_cache:
            cmd.append("--rebuild-cache")
        print(f"[tuned-all] running {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            print(f"[tuned-all] failed at {cfg} with code {proc.returncode}")
            return proc.returncode
    print("[tuned-all] completed tuned-all matrix")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all core models with hyperparameter tuning enabled.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fresh-results", action="store_true")
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument("--report-out", type=str, default="results/reports/thesis_tuned_all")
    parser.add_argument(
        "--budget",
        type=str,
        default="medium",
        choices=["quick", "medium", "heavy"],
        help="Hyperparameter budget profile.",
    )
    args = parser.parse_args()

    profile = _budget_profile(args.budget)
    generated = _write_generated_configs(args.budget, profile)
    print(
        f"[tuned-all] generated {len(generated)} configs "
        f"(budget={args.budget}: gnn={profile['gnn_trials']} xgb={profile['xgb_trials']} "
        f"lstm={profile['lstm_trials']} mlp={profile['mlp_trials']})"
    )

    if args.fresh_results:
        _reset_outputs(args.report_out)

    code = run_matrix(generated, rebuild_cache=args.rebuild_cache)
    if code == 0 and args.with_report:
        dedup_cmd = [
            sys.executable,
            "scripts/deduplicate_results_ledger.py",
            "--results",
            "results/results_tuned_all.jsonl",
        ]
        print(f"[tuned-all] running {' '.join(dedup_cmd)}")
        proc = subprocess.run(dedup_cmd, cwd=REPO_ROOT)
        code = proc.returncode
    if code == 0 and args.with_report:
        report_cmd = [
            sys.executable,
            "scripts/generate_thesis_report.py",
            "--results",
            "results/results_tuned_all.jsonl",
            "--out",
            args.report_out,
            "--expected-runs",
            str(len(CORE_MATRIX_CONFIGS) * 2),
        ]
        print(f"[tuned-all] running {' '.join(report_cmd)}")
        proc = subprocess.run(report_cmd, cwd=REPO_ROOT)
        code = proc.returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()
