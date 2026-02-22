import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze(pred_csv, top_k=20):
    df = pd.read_csv(pred_csv)
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique())
    stats = []
    zero_std_days = 0
    days_no_select = 0
    sample_days = []
    for d in dates:
        g = df[df['date'] == d].sort_values('pred', ascending=False)
        preds = g['pred'].values
        if len(preds) == 0:
            days_no_select += 1
            continue
        if np.nanstd(preds) < 1e-10:
            zero_std_days += 1
        selected = g.head(top_k)
        n_selected = selected.shape[0]
        if n_selected == 0:
            days_no_select += 1
        stats.append({'date': d, 'n_selected': n_selected, 'pred_mean': float(np.nanmean(preds)), 'pred_std': float(np.nanstd(preds)), 'selected_mean': float(selected['pred'].mean()) if n_selected>0 else np.nan})
        if len(sample_days) < 5 and n_selected>0:
            sample_days.append((d, selected['ticker'].tolist()[:min(10,len(selected))], float(selected['pred'].iloc[0]) if n_selected>0 else None))

    stats_df = pd.DataFrame(stats)
    print(f"Total days: {len(dates)}, days with no rows: {days_no_select}, days with near-constant preds: {zero_std_days}")
    if not stats_df.empty:
        print(stats_df[['n_selected','pred_mean','pred_std','selected_mean']].describe())
    print('\nSample selected tickers on first few days:')
    for d, tlist, topval in sample_days:
        print(d.date(), 'top tickers:', tlist, 'top pred', topval)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/diagnose_trades.py path/to/<model>_predictions.csv [top_k]')
        raise SystemExit(1)
    pred_csv = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if not Path(pred_csv).exists():
        print('File not found:', pred_csv); raise SystemExit(1)
    analyze(pred_csv, top_k=top_k)
