import matplotlib.pyplot as plt
from pathlib import Path


def plot_equity_curve(equity_series, title, out_path: Path):
    plt.figure()
    equity_series.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
