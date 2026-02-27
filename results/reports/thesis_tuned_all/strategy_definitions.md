# Strategy Definitions

- `Buy and hold (fixed shares)`: equal dollars at t0, fixed shares, no rebalance.
- `Equal weight (rebalanced, all assets)`: set 1/N on rebalance dates, hold and let weights drift between rebalances.
- `Top K long-only (equal weight within Top K)`: select Top K by model score on rebalance dates, equal weight within Top K, hold between rebalances.
- `Top 3 long, bottom 3 short (market-neutral)`: top 3 long and bottom 3 short, equal within legs, scaled to +0.5 / -0.5.

Rebalance notation: `reb=1`, `reb=5`, `reb=21`.

Turnover/cost convention:
- `turnover_t = sum_i |w_{i,t} - w_{i,t-1}|`
- `cost_t = (bps/10000) * turnover_t` on rebalance dates.
