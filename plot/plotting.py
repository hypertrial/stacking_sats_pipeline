# ---------------------------
# core/plots.py
# ---------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import configuration constants and helper functions
from config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS, MIN_WEIGHT
from strategy.strategy_template import construct_features


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_cycle_label(start_year: int, cycle_idx: int) -> str:
    """
    Pretty cycle label for arbitrary cycle length.
    """
    first = start_year + CYCLE_YEARS * cycle_idx
    if CYCLE_YEARS == 1:
        return f"{first}"
    return f"{first}–{first + CYCLE_YEARS - 1}"


def _cycle_idx(timestamp, start_year):
    """
    Integer cycle index for a timestamp relative to the first back-test year.
    """
    return (timestamp.year - start_year) // CYCLE_YEARS


# ──────────────────────────────────────────────────────────────────────────────
# plotting utilities
# ──────────────────────────────────────────────────────────────────────────────


def plot_features(
    df,
    weights=None,
    *,
    start_date: str | pd.Timestamp = BACKTEST_START,
    end_date: str | pd.Timestamp = BACKTEST_END,
):
    """
    Plot BTC price vs. the first derived feature within the chosen back-test window.

    Parameters
    ----------
    df : pd.DataFrame
        Raw BTC price dataframe (`PriceUSD` must be present).
    weights : pd.Series or None, optional
        Daily weight series aligned on the same index as `df` (used only for
        marker styling). If provided, it is automatically trimmed to the same
        date window.
    start_date, end_date : str or pd.Timestamp, optional
        Window to plot.  Defaults to BACKTEST_START / BACKTEST_END from config.
    """
    # Build features first, then trim to the requested window
    df = construct_features(df).loc[start_date:end_date]

    # Trim weights (if any) to the same index
    if weights is not None:
        weights = weights.loc[df.index]

    feature_name = df.columns[1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"BTC Price and {feature_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

    # Main series
    ax.plot(df.index, df["PriceUSD"], label="BTC Price", color="black", alpha=0.7)
    ax.plot(df.index, df[feature_name], label=feature_name, color="orange", alpha=0.7)

    # Highlight regions where the feature value exceeds price
    signal = df["PriceUSD"] < df[feature_name]
    ax.fill_between(
        df.index,
        df["PriceUSD"],
        df[feature_name],
        where=signal,
        color="green",
        alpha=0.1,
    )

    # Optional weight markers (colour-coded by the same signal for now)
    if weights is not None:
        ax.scatter(
            df.index[~signal],
            df.loc[~signal, "PriceUSD"],
            marker="o",
            facecolors="none",
            edgecolors="blue",
            label="Uniform",
        )
        ax.scatter(
            df.index[signal],
            df.loc[signal, "PriceUSD"],
            marker="o",
            color="red",
            label="Dynamic",
        )

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_weights(
    weights: pd.Series,
    *,
    start_date: str | pd.Timestamp = BACKTEST_START,
    cycle_years: int = CYCLE_YEARS,
):
    """
    Plot daily position weights with one curve per investment cycle and show
    a horizontal dashed line at the global MIN_WEIGHT threshold.
    """

    # ── helpers ─────────────────────────────────────────────────────────────
    def _cycle_idx(ts, first_year):
        return (ts.year - first_year) // cycle_years

    def _cycle_label(first_year, idx):
        first = first_year + cycle_years * idx
        return f"{first}" if cycle_years == 1 else f"{first}–{first + cycle_years - 1}"

    # ── compute cycle labels ───────────────────────────────────────────────
    start_year = pd.to_datetime(start_date).year
    cycle_indices = weights.index.to_series().apply(
        lambda dt: _cycle_idx(dt, start_year)
    )

    # ── plot ───────────────────────────────────────────────────────────────
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, group in weights.groupby(cycle_indices):
        label = _cycle_label(start_year, idx)
        ax.plot(group.index, group.values, label=label, color=cmap(idx % 10))

        # per-cycle uniform weight baseline
        uniform = 1.0 / len(group)
        ax.hlines(
            uniform,
            group.index.min(),
            group.index.max(),
            color=cmap(idx % 10),
            linestyle="--",
            alpha=0.5,
        )

    # global MIN_WEIGHT threshold (one line across the whole plot)
    ax.axhline(
        MIN_WEIGHT,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"MIN_WEIGHT = {MIN_WEIGHT:g}",
    )

    ax.set_title("Final Daily Weights")

    labels = list(weights.groupby(cycle_indices).groups.keys())
    n_labels = len(labels) + 1  # +1 for MIN_WEIGHT line

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=n_labels,
        fontsize="small",
        frameon=True,
        handlelength=1.5,
        columnspacing=1.2,
    )

    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weight_sums_by_cycle(
    weights,
    *,
    start_date: str | pd.Timestamp = BACKTEST_START,
    cycle_years: int = CYCLE_YEARS,
):
    """
    Bar-plot showing the sum of daily weights inside each investment cycle.

    Parameters
    ----------
    weights : pd.Series
        Daily weight series indexed by date.
    start_date : str or pd.Timestamp, optional
        First day of the back-test (default = BACKTEST_START).
    cycle_years : int, optional
        Length of each investment cycle (default = CYCLE_YEARS).
    """

    # ── helpers ───────────────────────────────────────────────────────────
    def _cycle_idx(ts, first_year):
        return (ts.year - first_year) // cycle_years

    def _cycle_label(first_year, idx):
        first = first_year + cycle_years * idx
        return f"{first}" if cycle_years == 1 else f"{first}–{first + cycle_years - 1}"

    # ── aggregate sums ────────────────────────────────────────────────────
    start_year = pd.to_datetime(start_date).year
    cycle_indices = weights.index.to_series().apply(
        lambda dt: _cycle_idx(dt, start_year)
    )
    weight_sums = weights.groupby(cycle_indices).sum()

    # ── console printout ──────────────────────────────────────────────────
    print("Cycle Weight Sums:")
    for idx, total in weight_sums.items():
        print(f"  {_cycle_label(start_year, idx)}: {total:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────
    labels = [_cycle_label(start_year, idx) for idx in weight_sums.index]
    plt.bar(labels, weight_sums.values, alpha=0.7)
    plt.axhline(1.0, linestyle="--", color="black", label="Target: 1.0")
    plt.title("Weight Sums by Cycle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spd_comparison(
    df_res: pd.DataFrame,
    strategy_name: str = "Dynamic",
    *,
    cycle_years: int = CYCLE_YEARS,
):
    """
    Compare uniform vs. static DCA vs. dynamic DCA in sats-per-dollar space and percentile space.
    """
    x = np.arange(len(df_res))
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_yscale("log")

    # ── SPD curves in desired order: max ▸ dynamic ▸ static_dca ▸ uniform ▸ min ──────────
    lines = ax1.plot(
        x,
        df_res["max_spd"],
        "o-",
        x,
        df_res["dynamic_spd"],
        "o-",
        x,
        df_res["static_dca_spd"],
        "s-",  # Square markers for static DCA
        x,
        df_res["uniform_spd"],
        "o-",
        x,
        df_res["min_spd"],
        "o-",
    )

    ax1.set_title(f"Uniform vs Static DCA (30%) vs {strategy_name} DCA (SPD)")
    ax1.set_ylabel("Sats per Dollar (log scale)")
    ax1.set_xlabel("Cycle" if cycle_years == 1 else f"Cycle ({cycle_years}-yr)")
    ax1.grid(True, linestyle="--", linewidth=0.5)

    ax1.legend(
        lines,
        [
            "Max spd (Low)",
            strategy_name,
            "Static DCA (30%)",
            "Uniform DCA",
            "Min spd (High)",
        ],
        loc="upper left",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_res.index, rotation=15, ha="right")

    # ── Percentile bars ────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    bar_w = 0.25  # Narrower bars to fit three categories
    bar1 = ax2.bar(
        x - bar_w, df_res["uniform_pct"], width=bar_w, alpha=0.3, label="Uniform %"
    )
    bar2 = ax2.bar(
        x, df_res["static_dca_pct"], width=bar_w, alpha=0.3, label="Static DCA (30%)"
    )
    bar3 = ax2.bar(
        x + bar_w,
        df_res["dynamic_pct"],
        width=bar_w,
        alpha=0.3,
        label=f"{strategy_name} %",
    )

    ax2.set_ylabel("SPD Percentile (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(
        [bar1, bar2, bar3],
        ["Uniform %", "Static DCA (30%)", f"{strategy_name} %"],
        loc="upper right",
    )

    plt.tight_layout()
    plt.show()
