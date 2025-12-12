import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# ============================================================
# 1. EQUITY CURVE PLOT
# ============================================================

def plot_equity_curve(equity_series, title, out_path: Path, use_log_scale=False):
    """
    Plot cumulative portfolio value over time.
    """

    plt.figure(figsize=(12, 5))

    plt.plot(
        equity_series.index,
        equity_series.values,
        linewidth=2,
        color="royalblue",
        alpha=0.9,
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    if use_log_scale:
        plt.yscale("log")

    plt.grid(alpha=0.3, linestyle="--")

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# 2. DAILY IC TIME SERIES PLOT
# ============================================================

def plot_daily_ic(ic_df, out_path: Path):
    """
    Plot daily Information Coefficient (IC) over time.
    
    ic_df must contain:
        - "date"
        - "ic"
    """

    plt.figure(figsize=(12, 4))

    plt.plot(
        ic_df["date"],
        ic_df["ic"],
        linewidth=1.2,
        color="darkgreen",
        alpha=0.8,
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Daily Information Coefficient", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("IC")

    plt.grid(alpha=0.3)
    plt.gcf().autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# 3. IC DISTRIBUTION HISTOGRAM
# ============================================================

def plot_ic_hist(ic_df, out_path: Path):
    """
    Plot histogram of daily IC values.
    
    ic_df must contain:
        - "ic"
    """

    plt.figure(figsize=(6, 4))

    plt.hist(
        ic_df["ic"].dropna(),
        bins=40,
        alpha=0.85,
        color="steelblue",
        edgecolor="black",
    )

    plt.title("Distribution of Daily IC", fontsize=13)
    plt.xlabel("IC")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ============================================================
# 4.  EQUITY CURVE COMPARISON PLOT
# ============================================================

def plot_equity_comparison(model_curve, bh_curve, title, out_path: Path, use_log_scale=False):
    """
    Plot model vs buy-and-hold equity curves on the same chart.
    """

    plt.figure(figsize=(12, 5))

    # Model curve
    plt.plot(
        model_curve.index,
        model_curve.values,
        label="Model",
        linewidth=2,
        color="royalblue",
    )

    # Buy and hold
    plt.plot(
        bh_curve.index,
        bh_curve.values,
        label="Buy & Hold",
        linewidth=2,
        color="orange",
        alpha=0.9,
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    if use_log_scale:
        plt.yscale("log")

    plt.grid(alpha=0.3, linestyle="--")

    plt.legend()

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
