# Repository Full Model/Results Report

Generated: `2026-02-27 16:37:33`

## 1) Run sets in this repo
| source_ledger | rows | models | model_families | rebalance_counts | split_test_start | target_horizon |
| --- | --- | --- | --- | --- | --- | --- |
| results/results_tuned_all.jsonl | 30 | gat, gcn, graphlasso_linear, lstm, mlp, tgat_static, tgcn_static, xgb_node2vec, xgb_raw | gnn, lstm, mlp, xgboost | {"1": 15, "5": 15} | 2020-01-01 | 1 |
| results/results.jsonl | 28 | gat, gcn, lstm, tgat_static, tgcn_static, xgb_node2vec, xgb_raw | gnn, lstm, xgboost | {"1": 14, "5": 14} | 2020-01-01 | 1 |
| results/results_retune.jsonl | 2 | graphlasso_linear | xgboost | {"1": 1, "5": 1} | 2020-01-01 | 1 |

## 2) Canonical thesis protocol (current)
- Target: `log_return`, horizon `1` day (regression).
- Split: train to `2015-12-31`, validation from `2016-01-01`, test from `2020-01-01`.
- Canonical rebalances: `reb=1`, `reb=5`.
- Canonical transaction cost in main comparison: `0 bps` (gross).
- Cost robustness reported separately: `0/5/10 bps`.

## 3) Features used
- Feature count: `18`
- `ret_1d`, `ret_5d`, `ret_20d`, `log_ret_1d`, `mom_3d`, `mom_10`, `mom_21d`, `vol_5d`, `vol_20d`, `vol_60d`, `drawdown_20d`, `volume_pct_change`, `vol_z_5`, `vol_z_20`, `rsi_14`, `macd_line`, `macd_signal`, `macd_hist`

## 4) Strategy definitions in report
```markdown
# Strategy Definitions (Professor-Facing)

These definitions are used consistently in report tables, plots, and thesis text.

## Labels
- `Buy and hold (fixed shares)`
- `Equal weight (rebalanced, all assets)`
- `Top K long-only (equal weight within Top K)`
- `Top 3 long, bottom 3 short (market-neutral)`

## Construction Rules
1. `Buy and hold (fixed shares)`:
At `t0`, allocate equal dollars across all assets, buy fixed shares, and never rebalance.
Portfolio value changes only with asset prices.

2. `Equal weight (rebalanced, all assets)`:
On each rebalance date, set weights to `1/N` across all assets.
Between rebalance dates, weights drift with returns and are not reset daily.

3. `Top K long-only (equal weight within Top K)`:
On each rebalance date, rank by model signal, select Top K, weight selected assets equally, and set all others to zero.
Hold until the next rebalance date.

4. `Top 3 long, bottom 3 short (market-neutral)`:
On each rebalance date, long top 3 and short bottom 3 by model signal.
Equal weight within each leg, then scale to long sum `+0.5` and short sum `-0.5`.
Hold until the next rebalance date.

## Rebalance Notation
All tables and figure labels use `reb=1`, `reb=5`, `reb=21`.

## Turnover and Cost Convention
- `turnover_t = sum_i |w_{i,t} - w_{i,t-1}|`
- `cost_t = (bps / 10000) * turnover_t`
- Costs are applied on rebalance dates.
```

## 5) Model/config inventory
- Total configs in `configs/runs`: `59`
| run_group | n_configs |
| --- | --- |
| ablation | 6 |
| core | 15 |
| exploratory | 18 |
| retune_medium | 5 |
| tuned_all | 15 |

Core matrix high-level:
| experiment_name | model_family | model_type | lookback_window | corr_window | top_k | transaction_cost_bps | backtest_policies | primary_rebalance_freq | tuning_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gat_corr_only | gnn | gat | 60 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gat_corr_sector_granger | gnn | gat | 250 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gat_granger_only | gnn | gat | 250 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gat_sector_only | gnn | gat | 60 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gcn_corr_only | gnn | gcn | 60 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gcn_corr_sector_granger | gnn | gcn | 250 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gcn_granger_only | gnn | gcn | 250 | 60 | 20 | 0 | [1, 5] | 1 |  |
| gcn_sector_only | gnn | gcn | 60 | 60 | 20 | 0 | [1, 5] | 1 |  |
| tgat_static_corr_only | gnn | tgat_static | 60 | 60 | 20 | 0 | [1, 5] | 1 |  |
| tgcn_static_corr_only | gnn | tgcn_static | 60 | 60 | 20 | 0 | [1, 5] | 1 |  |
| lstm | lstm | lstm | 60 |  | 20 | 0 | [1, 5] | 1 | False |
| mlp | mlp | mlp | 60 |  | 20 | 0 | [1, 5] | 1 | False |
| graphlasso_linear | xgboost | graphlasso_linear | 60 |  | 20 | 0 | [1, 5] | 1 |  |
| xgb_node2vec_corr | xgboost | xgb_node2vec | 60 | 60 | 20 | 0 | [1, 5] | 1 | True |
| xgb_raw | xgboost | xgb_raw | 60 |  | 20 | 0 | [1, 5] | 1 | False |

## 6) Best runs by ledger/rebalance (Sharpe)
| source_ledger | rebalance_freq | run_tag | model_name | edge_type | portfolio_sharpe_annualized | portfolio_final_value | portfolio_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results/results.jsonl | 1 | xgb_raw | xgb_raw | none | 0.504663737480047 | 1.503081825414194 | 0.582722929936305 |
| results/results.jsonl | 5 | xgb_raw | xgb_raw | none | 0.6935574558215141 | 1.833378628074269 | 0.13925159235668702 |
| results/results_retune.jsonl | 1 | graphlasso_linear_retune_medium | graphlasso_linear | graphlasso | 0.415966338140386 | 1.401273809829456 | 0.555095541401273 |
| results/results_retune.jsonl | 5 | graphlasso_linear_retune_medium | graphlasso_linear | graphlasso | 0.602923245858214 | 1.720628900385555 | 0.135987261146496 |
| results/results_tuned_all.jsonl | 1 | mlp_tuned_all | mlp | none | 0.8702984391604671 | 2.220050429438613 | 0.509952229299363 |
| results/results_tuned_all.jsonl | 1 | xgb_node2vec_corr_tuned_all | xgb_node2vec | node2vec_correlation | 0.6601739988995341 | 1.7477071345368551 | 0.5656847133757961 |
| results/results_tuned_all.jsonl | 1 | gcn_sector_only_tuned_all | gcn | sector | 0.6060515515289381 | 1.684794165307252 | 0.576990445859872 |
| results/results_tuned_all.jsonl | 1 | gat_corr_only_tuned_all | gat | corr | 0.562531025672602 | 1.631072910204374 | 0.514570063694267 |
| results/results_tuned_all.jsonl | 1 | gcn_granger_only_tuned_all | gcn | granger | 0.554129052999207 | 1.564397468012839 | 0.477945859872611 |
| results/results_tuned_all.jsonl | 5 | xgb_node2vec_corr_tuned_all | xgb_node2vec | node2vec_correlation | 0.8058302190730461 | 2.011157102306581 | 0.136146496815286 |
| results/results_tuned_all.jsonl | 5 | mlp_tuned_all | mlp | none | 0.798272258944222 | 2.099649078618426 | 0.13805732484076402 |
| results/results_tuned_all.jsonl | 5 | xgb_raw_tuned_all | xgb_raw | none | 0.738978064491419 | 1.872559557908624 | 0.140207006369426 |
| results/results_tuned_all.jsonl | 5 | lstm_tuned_all | lstm | none | 0.7143462410918561 | 1.9007722009428711 | 0.12850318471337502 |
| results/results_tuned_all.jsonl | 5 | gat_corr_sector_granger_tuned_all | gat | corr+sector+granger | 0.702959157940076 | 1.853076738280619 | 0.146894904458598 |

## 7) Family aggregates
| source_ledger | model_family | rebalance_freq | n_runs | sharpe_mean | sharpe_max | ann_return_mean | max_dd_mean | turnover_mean | train_time_mean_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/results.jsonl | gnn | 1 | 10 |  |  | 0.0991122727291127 | -0.30957209786451445 | 0.5572531847133753 | 339.63043012619016 |
| results/results.jsonl | gnn | 5 | 10 |  |  | 0.13127289331893893 | -0.33753369977486863 | 0.14047770700636902 | 339.63043012619016 |
| results/results.jsonl | lstm | 1 | 1 |  |  | 0.128139159894537 | -0.31588249454780903 | 0.23749999999999902 | 243.98795676231384 |
| results/results.jsonl | lstm | 5 | 1 |  |  | 0.13981788108210402 | -0.35837177773974005 | 0.12348726114649601 | 243.98795676231384 |
| results/results.jsonl | xgboost | 1 | 3 | 0.504663737480047 | 0.504663737480047 | 0.08798261043603434 | -0.31471927695414303 | 0.5839171974522287 | 83.34412964185078 |
| results/results.jsonl | xgboost | 5 | 3 | 0.6935574558215141 | 0.6935574558215141 | 0.13507017023724435 | -0.32724082509432967 | 0.139331210191082 | 83.34412964185078 |
| results/results_retune.jsonl | xgboost | 1 | 1 | 0.415966338140386 | 0.415966338140386 | 0.07203084071039201 | -0.32232673502708303 | 0.555095541401273 | 25.50920796394348 |
| results/results_retune.jsonl | xgboost | 5 | 1 | 0.602923245858214 | 0.602923245858214 | 0.11711250581207601 | -0.357221223063237 | 0.135987261146496 | 25.50920796394348 |
| results/results_tuned_all.jsonl | gnn | 1 | 10 | 0.5059682646809922 | 0.6060515515289381 | 0.08832972258488261 | -0.30958676676561814 | 0.5424283439490443 | 14131.902932024002 |
| results/results_tuned_all.jsonl | gnn | 5 | 10 | 0.642628119513206 | 0.702959157940076 | 0.1212973867692809 | -0.33038350936003763 | 0.1380812101910823 | 14131.902932024002 |
| results/results_tuned_all.jsonl | lstm | 1 | 1 | 0.5322895406190641 | 0.5322895406190641 | 0.09750997100618701 | -0.35233756308434305 | 0.370541401273885 | 254.4479660987854 |
| results/results_tuned_all.jsonl | lstm | 5 | 1 | 0.7143462410918561 | 0.7143462410918561 | 0.139800946134733 | -0.347303382728054 | 0.12850318471337502 | 254.4479660987854 |
| results/results_tuned_all.jsonl | mlp | 1 | 1 | 0.8702984391604671 | 0.8702984391604671 | 0.17487210424978 | -0.28062674388166003 | 0.509952229299363 | 35.315099000930786 |
| results/results_tuned_all.jsonl | mlp | 5 | 1 | 0.798272258944222 | 0.798272258944222 | 0.161801527391667 | -0.33046400953286503 | 0.13805732484076402 | 35.315099000930786 |
| results/results_tuned_all.jsonl | xgboost | 1 | 3 | 0.5292088899404197 | 0.6601739988995341 | 0.09247402857440734 | -0.3028909316192783 | 0.5743365180467087 | 35.67769908905029 |
| results/results_tuned_all.jsonl | xgboost | 5 | 3 | 0.7159105098075598 | 0.8058302190730461 | 0.13465882568502402 | -0.3324412339112954 | 0.13744692144373602 | 35.67769908905029 |

## 8) Baselines/cost/audits from thesis_tuned_all
Baseline policy comparison:
| strategy_name | strategy_kind | rebalance_freq | rebalance_label | portfolio_sharpe_annualized | portfolio_annualized_return | portfolio_max_drawdown | portfolio_final_value | test_start_date | test_end_date | source_run_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Buy and hold (fixed shares) | baseline_buy_and_hold | 1 | reb=1 | 1.2272780893102675 | 0.3842240191009789 | -0.3471179958366627 | 5.049288556044275 | 2020-01-02 | 2024-12-27 | mlp_tuned_all |
| Equal weight (rebalanced, all assets) | baseline_equal_weight | 1 | reb=1 | 0.7667087720637379 | 0.1518790337272664 | -0.3464553449053246 | 2.0221621477103713 | 2020-01-02 | 2024-12-27 | mlp_tuned_all |
| Buy and hold (fixed shares) | baseline_buy_and_hold | 5 | reb=5 | 1.2272774449271957 | 0.3842240191009789 | -0.3471179958366627 | 5.049288556044275 | 2020-01-02 | 2024-12-27 | xgb_node2vec_corr_tuned_all |
| Equal weight (rebalanced, all assets) | baseline_equal_weight | 5 | reb=5 | 0.7667082048575795 | 0.1518790337272664 | -0.3464553449053246 | 2.0221621477103713 | 2020-01-02 | 2024-12-27 | xgb_node2vec_corr_tuned_all |

Professor main table:
| strategy_name | strategy_label | type | run_tag | run_key | rebalance_freq | rebalance_label | final_value | annual_return | annual_vol | sharpe_annualized | max_drawdown | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Buy and hold (fixed shares) | Buy and hold (fixed shares) | baseline |  | baseline_buy_and_hold_1 | 1 | reb=1 | 5.049288556044275 | 0.3842240191009789 | 0.3130700551469256 | 1.2272780893102675 | -0.3471179958366627 | 0.0 |
| Buy and hold (fixed shares) | Buy and hold (fixed shares) | baseline |  | baseline_buy_and_hold_5 | 5 | reb=5 | 5.049288556044275 | 0.3842240191009789 | 0.3130702195246257 | 1.2272774449271957 | -0.3471179958366627 | 0.0 |
| Equal weight (rebalanced, all assets) | Equal weight (rebalanced, all assets) | baseline |  | baseline_equal_weight_1 | 1 | reb=1 | 2.0221621477103713 | 0.1518790337272664 | 0.1980922082298028 | 0.7667087720637379 | -0.3464553449053246 | 0.0114616105832656 |
| Equal weight (rebalanced, all assets) | Equal weight (rebalanced, all assets) | baseline |  | baseline_equal_weight_5 | 5 | reb=5 | 2.0221621477103713 | 0.1518790337272664 | 0.1980923547772373 | 0.7667082048575795 | -0.3464553449053246 | 0.0056770380486862 |
| mlp_tuned_all (Top K long-only (equal weight within Top K)) | Top K long-only (equal weight within Top K) | model | mlp_tuned_all | d7b09cd9d2de | 5 | reb=5 | 2.0996490786184245 | 0.1618015273916671 | 0.2157322068989569 | 0.7982722589442207 | -0.3304640095328649 | 0.1380573248407643 |
| xgb_node2vec_corr_tuned_all (Top K long-only (equal weight within Top K)) | Top K long-only (equal weight within Top K) | model | xgb_node2vec_corr_tuned_all | 64952c498bf0 | 5 | reb=5 | 2.190442721435154 | 0.1717378476972582 | 0.1986808250162476 | 0.8919121202492275 | -0.3275175814621586 | 0.1361464968152866 |
| xgb_raw_tuned_all (Top K long-only (equal weight within Top K)) | Top K long-only (equal weight within Top K) | model | xgb_raw_tuned_all | 725b09e91479 | 5 | reb=5 | 2.044670355765633 | 0.1549979996958894 | 0.1967359378442815 | 0.8284608548461 | -0.3103494685450531 | 0.1402070063694267 |

Cost sensitivity summary:
| strategy | cost_bps | cost_label | sharpe_mean | ann_return_mean | max_dd_mean | turnover_mean | n_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_only_topk | 0 | gross (0 bps) | 0.7970677771570646 | 0.15636229306695 | -0.3166497786640526 | 0.33625 | 30 |
| long_only_topk | 5 | net (5 bps) | 0.5916085758986999 | 0.1086252051500235 | -0.3208006005797992 | 0.33625 | 30 |
| long_only_topk | 10 | net (10 bps) | 0.3861749181176022 | 0.0635402874607772 | -0.328367964724029 | 0.33625 | 30 |
| long_short_top3_bottom3 | 0 | gross (0 bps) | 0.1309744587234903 | 0.0096698735050562 | -0.2846519117770905 | 0.2052717049361098 | 30 |
| long_short_top3_bottom3 | 5 | net (5 bps) | -0.0474637398271233 | -0.0158560288145841 | -0.3286656597468125 | 0.2052717049361098 | 30 |
| long_short_top3_bottom3 | 10 | net (10 bps) | -0.2256464507124526 | -0.0405303357797325 | -0.3784755180424828 | 0.2052717049361098 | 30 |

Hard-gate audits:
| audit_name | status | fail_rows | warning_rows | detail |
| --- | --- | --- | --- | --- |
| equal_weight_rebalance_integrity | pass | 0 | 0 | fail if reb=1 and reb=5 EQW series are identical |
| graph_time_awareness | pass | 0 | 0 | fail if graph max timestamp exceeds policy bound |

Lookback subset status:
| run_tag | rebalance_freq | lookback_window | status | portfolio_sharpe_annualized | portfolio_annualized_return | portfolio_max_drawdown | portfolio_final_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_node2vec_corr_tuned_all | 5 | 14 | missing_rerun_required |  |  |  |  |
| xgb_node2vec_corr_tuned_all | 5 | 30 | missing_rerun_required |  |  |  |  |
| xgb_node2vec_corr_tuned_all | 5 | 60 | available | 0.8058302190730461 | 0.151949627758707 | -0.32841801947119 | 2.011157102306581 |
| xgb_raw_tuned_all | 5 | 14 | missing_rerun_required |  |  |  |  |
| xgb_raw_tuned_all | 5 | 30 | missing_rerun_required |  |  |  |  |
| xgb_raw_tuned_all | 5 | 60 | available | 0.738978064491419 | 0.134914343484289 | -0.311684459199459 | 1.872559557908624 |
| gat_corr_sector_granger_tuned_all | 5 | 14 | missing_rerun_required |  |  |  |  |
| gat_corr_sector_granger_tuned_all | 5 | 30 | missing_rerun_required |  |  |  |  |
| gat_corr_sector_granger_tuned_all | 5 | 60 | missing_rerun_required |  |  |  |  |

Monthly subset (first 20 rows):
| strategy_name | strategy_kind | strategy_label | rebalance_freq | rebalance_label | cost_bps | cost_label | portfolio_final_value | portfolio_annualized_return | portfolio_sharpe_annualized | portfolio_max_drawdown | portfolio_turnover | n_points | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_node2vec_corr_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 1 | reb=1 | 0 | gross (0 bps) | 2.492792342826601 | 0.2025331719114125 | 1.0166724408030363 | -0.2894475800381865 | 0.5656847133757962 | 1256 |  |
| xgb_node2vec_corr_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 5 | reb=5 | 0 | gross (0 bps) | 2.190442721435154 | 0.1717378476972582 | 0.8919121202492275 | -0.3275175814621586 | 0.1361464968152866 | 1256 |  |
| xgb_node2vec_corr_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 21 | reb=21 | 0 | gross (0 bps) | 1.8352914206413724 | 0.1308790285293306 | 0.7219843143306452 | -0.2969289081674049 | 0.0342356687898089 | 1256 |  |
| xgb_raw_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 1 | reb=1 | 0 | gross (0 bps) | 2.188605527063555 | 0.1708705651171855 | 0.8973677943326156 | -0.2850224955854418 | 0.6022292993630574 | 1256 |  |
| xgb_raw_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 5 | reb=5 | 0 | gross (0 bps) | 2.044670355765633 | 0.1549979996958894 | 0.8284608548461 | -0.3103494685450531 | 0.1402070063694267 | 1256 |  |
| xgb_raw_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 21 | reb=21 | 0 | gross (0 bps) | 1.8816403803491195 | 0.1359021243328413 | 0.7784035469070005 | -0.2744031743594393 | 0.0359076433121019 | 1256 |  |
| gat_corr_sector_granger_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 1 | reb=1 | 0 | gross (0 bps) | 2.0725638177389407 | 0.1589823262524534 | 0.808884119047596 | -0.3158435118350763 | 0.5822452229299364 | 1256 |  |
| gat_corr_sector_granger_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 5 | reb=5 | 0 | gross (0 bps) | 2.031962195048989 | 0.1543908686893969 | 0.7923950541295828 | -0.3079428822626847 | 0.1468949044585987 | 1256 |  |
| gat_corr_sector_granger_tuned_all | model_topk_long_only | Top K long-only (equal weight within Top K) | 21 | reb=21 | 0 | gross (0 bps) | 1.9519441104291009 | 0.1451229541343226 | 0.7653220072302555 | -0.2983651268824964 | 0.0371815286624203 | 1256 |  |
| Equal weight (rebalanced, all assets) | baseline_equal_weight | Equal weight (rebalanced, all assets) | 1 | reb=1 | 0 | gross (0 bps) | 1.9868601593707904 | 0.147686771123656 | 0.8041563093083701 | -0.3112345749237915 | 0.0114616105832656 | 1256 |  |
| Buy and hold (fixed shares) | baseline_buy_and_hold | Buy and hold (fixed shares) | 1 | reb=1 | 0 | gross (0 bps) | 2.407432576265708 | 0.1929297070461948 | 0.9566326214293598 | -0.2986234113856566 | 0.0 | 1256 |  |
| Equal weight (rebalanced, all assets) | baseline_equal_weight | Equal weight (rebalanced, all assets) | 5 | reb=5 | 0 | gross (0 bps) | 1.9695125920214247 | 0.1456692112801692 | 0.7976064107755235 | -0.3136352988773329 | 0.0056770380486862 | 1256 |  |
| Buy and hold (fixed shares) | baseline_buy_and_hold | Buy and hold (fixed shares) | 5 | reb=5 | 0 | gross (0 bps) | 2.407432576265708 | 0.1929297070461948 | 0.9566326214293598 | -0.2986234113856566 | 0.0 | 1256 |  |
| Equal weight (rebalanced, all assets) | baseline_equal_weight | Equal weight (rebalanced, all assets) | 21 | reb=21 | 0 | gross (0 bps) | 1.9685480556673909 | 0.1455566173426528 | 0.8033473131931261 | -0.3095585835922766 | 0.0032683370151062 | 1256 |  |
| Buy and hold (fixed shares) | baseline_buy_and_hold | Buy and hold (fixed shares) | 21 | reb=21 | 0 | gross (0 bps) | 2.407432576265708 | 0.1929297070461948 | 0.9566326214293598 | -0.2986234113856566 | 0.0 | 1256 |  |

## 9) Caveats
- Ledger rows do not store `transaction_cost_bps` explicitly; cost assumptions come from config/protocol + cost-sensitivity CSVs.
- Tuning grids are stored in configs; selected best params are not uniformly persisted across all model families.
- `reb=21` currently exists in report-level subset outputs; canonical ledgers are still `reb=1` and `reb=5`.
