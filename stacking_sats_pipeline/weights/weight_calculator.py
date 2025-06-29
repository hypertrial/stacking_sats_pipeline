"""
Simple weight calculator for historical Bitcoin allocation.

This module integrates the strategy and data modules to generate
allocation weights for historical periods using coinmetrics data.
"""

import logging

import pandas as pd

from ..config import BACKTEST_END, BACKTEST_START
from ..data import load_data
from ..strategy import compute_weights
from ..strategy.strategy_template import construct_features

logger = logging.getLogger(__name__)

# =============================================================================
# DATA VALIDATION AND LOADING
# =============================================================================


def validate_date_range(btc_df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """Validate that requested dates are within available historical data."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Handle empty DataFrame
    if btc_df.empty:
        raise ValueError("No historical data available")

    data_start = btc_df.index.min()
    data_end = btc_df.index.max()

    # Handle timezone-aware data (only if index is DatetimeIndex)
    if hasattr(btc_df.index, "tz") and btc_df.index.tz is not None:
        # Convert timestamps to match the timezone of the data index
        start_ts = start_ts.tz_localize(btc_df.index.tz)
        end_ts = end_ts.tz_localize(btc_df.index.tz)

    if start_ts < data_start:
        raise ValueError(
            f"Start date {start_date} is before available data starts "
            f"({data_start.strftime('%Y-%m-%d')})"
        )

    if end_ts > data_end:
        raise ValueError(
            f"End date {end_date} is after available data ends ({data_end.strftime('%Y-%m-%d')})"
        )

    logger.info(f"Date range {start_date} to {end_date} validated against available data")


def get_historical_btc_data() -> pd.DataFrame:
    """Load historical BTC data from coinmetrics dump."""
    logger.info("Loading historical BTC price data from coinmetrics...")
    btc_df = load_data()
    logger.info(
        f"Loaded data from {btc_df.index.min().strftime('%Y-%m-%d')} to "
        f"{btc_df.index.max().strftime('%Y-%m-%d')}"
    )
    return btc_df


# =============================================================================
# STRATEGY COMPUTATION WITH DYNAMIC DATE RANGES
# =============================================================================


def compute_weights_for_period(btc_df: pd.DataFrame, end_date: str) -> pd.Series:
    """Compute weights using strategy, extending beyond BACKTEST_END if needed."""
    backtest_end_ts = pd.Timestamp(BACKTEST_END)
    end_ts = pd.Timestamp(end_date)

    if end_ts <= backtest_end_ts:
        logger.info("Computing allocation weights...")
        return compute_weights(btc_df)

    logger.info(f"Extending computation beyond BACKTEST_END ({BACKTEST_END}) to {end_date}")

    # Use strategy components with extended date range
    import numpy as np

    from ..config import CYCLE_YEARS, MIN_WEIGHT

    # Strategy parameters
    REBALANCE_WINDOW = max(int(CYCLE_YEARS * 365) // 2, 1)
    ALPHA = 1.25

    # Compute features and weights for extended range
    df_feat = construct_features(btc_df).loc[BACKTEST_START:end_date]
    weights = pd.Series(index=df_feat.index, dtype=float)

    start_year = pd.to_datetime(BACKTEST_START).year
    cycle_labels = df_feat.index.to_series().apply(lambda ts: (ts.year - start_year) // CYCLE_YEARS)

    # Apply strategy logic per cycle
    for _, cycle_df in df_feat.groupby(cycle_labels):
        N = len(cycle_df)
        base_wt = 1.0 / N
        temp_wt = np.full(N, base_wt)

        for day in range(N):
            price, ma, sd = (
                cycle_df["PriceUSD"].iat[day],
                cycle_df["ma200"].iat[day],
                cycle_df["std200"].iat[day],
            )
            if pd.isna(ma) or pd.isna(sd) or sd == 0 or price >= ma:
                continue

            z = (ma - price) / sd
            boosted = temp_wt[day] * (1 + ALPHA * z)
            excess = boosted - temp_wt[day]

            start_r = max(N - REBALANCE_WINDOW, day + 1)
            idx_r = np.arange(start_r, N)
            if len(idx_r) == 0:
                continue
            reduction = excess / len(idx_r)
            if np.all(temp_wt[idx_r] - reduction >= MIN_WEIGHT):
                temp_wt[day] = boosted
                temp_wt[idx_r] -= reduction

        weights.loc[cycle_df.index] = temp_wt

    return weights


def get_historical_data_for_period(start_date: str, end_date: str) -> pd.DataFrame:
    """Load and validate historical BTC data for the specified period."""
    btc_df = get_historical_btc_data()
    validate_date_range(btc_df, start_date, end_date)
    return btc_df


# =============================================================================
# MAIN WEIGHT CALCULATION FUNCTIONS
# =============================================================================


def get_weights_for_period(start_date: str, end_date: str) -> pd.Series:
    """Generate allocation weights for the specified date range."""
    btc_df = get_historical_data_for_period(start_date, end_date)
    weights = compute_weights_for_period(btc_df, end_date)

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    # Handle timezone-aware data
    if weights.index.tz is not None:
        # Convert timestamps to match the timezone of the weights index
        start_ts = start_ts.tz_localize(weights.index.tz)
        end_ts = end_ts.tz_localize(weights.index.tz)

    logger.info(
        f"Filtering weights for period: {start_ts.strftime('%Y-%m-%d')} to "
        f"{end_ts.strftime('%Y-%m-%d')}"
    )

    period_weights = weights.loc[start_ts:end_ts]
    logger.info(f"Found {len(period_weights)} weight entries for specified period")
    return period_weights


def display_weights(budget: float, start_date: str, end_date: str):
    """Display weights and daily allocations for the specified period and budget."""
    weights = get_weights_for_period(start_date, end_date)
    btc_df = get_historical_data_for_period(start_date, end_date)

    print(f"\n=== Allocation Weights from {start_date} to {end_date} ===")
    print(f"Total budget: ${budget:,.2f}")
    print(f"Total days: {len(weights)}")
    print(f"Total weight: {weights.sum():.4f}")
    print(f"Average daily weight: {weights.mean():.4f}")
    if len(weights) > 0:
        print(f"Average daily allocation: ${budget / len(weights):,.2f}")

    print("\nDaily Breakdown:")
    print("-" * 80)
    print(f"{'Date':<12} {'Weight %':<10} {'USD Amount':<12} {'BTC Price':<12} {'BTC Amount':<12}")
    print("-" * 80)

    total_btc = 0
    total_period_weight = weights.sum()

    for date, weight in weights.items():
        weight_pct = (weight / total_period_weight * 100) if total_period_weight > 0 else 0
        daily_usd = budget * weight

        # Get BTC price for this date if available
        price_str = btc_amount_str = "N/A"
        if date in btc_df.index:
            price = btc_df.loc[date, "PriceUSD"]
            if pd.notna(price):
                price_str = f"${price:.2f}"
                btc_amount = daily_usd / price
                btc_amount_str = f"{btc_amount:.6f}"
                total_btc += btc_amount

        print(
            f"{date.strftime('%Y-%m-%d'):<12} {weight_pct:<10.2f}% "
            f"${daily_usd:<11.2f} {price_str:<12} {btc_amount_str:<12}"
        )

    print("-" * 80)
    print(f"{'TOTAL':<12} {100.00:<10.2f}% ${budget:<11.2f} {'':12} {total_btc:<12.6f}")


def save_weights_to_csv(
    budget: float, start_date: str, end_date: str, filename: str | None = None
) -> str:
    """Save weights and allocations to CSV file."""
    if filename is None:
        filename = f"weights_{start_date}_to_{end_date}.csv"

    df = _create_weights_dataframe(budget, start_date, end_date)
    df.to_csv(filename)
    logger.info(f"Saved weights and allocations to {filename}")
    return filename


def save_weights_to_parquet(
    budget: float, start_date: str, end_date: str, filename: str | None = None
) -> str:
    """Save weights and allocations to Parquet file."""
    if filename is None:
        filename = f"weights_{start_date}_to_{end_date}.parquet"

    df = _create_weights_dataframe(budget, start_date, end_date)
    df.to_parquet(filename)
    logger.info(f"Saved weights and allocations to {filename}")
    return filename


def save_weights(
    budget: float,
    start_date: str,
    end_date: str,
    filename: str | None = None,
    file_format: str = "csv",
) -> str:
    """Save weights and allocations to file in specified format."""
    if file_format.lower() == "parquet":
        return save_weights_to_parquet(budget, start_date, end_date, filename)
    else:
        return save_weights_to_csv(budget, start_date, end_date, filename)


def _create_weights_dataframe(budget: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Create a DataFrame with weights, allocations, and prices for export."""
    weights = get_weights_for_period(start_date, end_date)
    btc_df = get_historical_data_for_period(start_date, end_date)

    # Create DataFrame with weights, allocations, and prices
    df = pd.DataFrame(
        {
            "date": weights.index,
            "weight": weights.values,
            "weight_percent": weights.values * 100,
            "usd_allocation": weights.values * budget,
        }
    )

    # Add BTC prices and amounts
    prices, btc_amounts = [], []
    for date in weights.index:
        if date in btc_df.index:
            price = btc_df.loc[date, "PriceUSD"]
            if pd.notna(price):
                prices.append(price)
                daily_usd = weights.loc[date] * budget
                btc_amounts.append(daily_usd / price)
            else:
                prices.append(None)
                btc_amounts.append(None)
        else:
            prices.append(None)
            btc_amounts.append(None)

    df["btc_price_usd"] = prices
    df["btc_amount"] = btc_amounts
    df.set_index("date", inplace=True)

    return df


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate Bitcoin allocation weights for a specific period and budget"
    )
    parser.add_argument("budget", type=float, help="Total USD budget to allocate across the period")
    parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--save", "-s", action="store_true", help="Save weights to CSV file")
    parser.add_argument("--filename", "-f", type=str, help="CSV filename (optional)")

    args = parser.parse_args()

    try:
        display_weights(args.budget, args.start_date, args.end_date)

        if args.save:
            csv_file = save_weights_to_csv(
                args.budget, args.start_date, args.end_date, args.filename
            )
            print(f"\nWeights saved to: {csv_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
