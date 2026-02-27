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
