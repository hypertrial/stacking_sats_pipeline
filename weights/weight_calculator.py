"""
Simple weight calculator for current month Bitcoin allocation.

This module integrates the strategy and data modules to generate
allocation weights for the current month.
"""

import logging

import pandas as pd

from data import load_data
from strategy import compute_weights

logger = logging.getLogger(__name__)


def get_weights_for_period(start_date: str, end_date: str) -> pd.Series:
    """
    Generate allocation weights for the specified date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        pandas Series with dates as index and weights as values for specified period
    """
    # Load BTC price data
    logger.info("Loading BTC price data...")
    btc_df = load_data()

    # Compute weights using the strategy
    logger.info("Computing allocation weights...")
    weights = compute_weights(btc_df)

    # Parse date strings
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    logger.info(
        f"Filtering weights for period: {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}"
    )

    # Filter weights for specified period
    period_weights = weights.loc[start_ts:end_ts]

    logger.info(f"Found {len(period_weights)} weight entries for specified period")
    return period_weights


def display_weights(budget: float, start_date: str, end_date: str):
    """
    Display weights and daily allocations for the specified period and budget.

    Args:
        budget: Total USD budget to allocate across the period
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    weights = get_weights_for_period(start_date, end_date)
    btc_df = load_data()

    print(f"\n=== Allocation Weights from {start_date} to {end_date} ===")
    print(f"Total budget: ${budget:,.2f}")
    print(f"Total days: {len(weights)}")
    print(f"Total weight: {weights.sum():.4f}")
    print(f"Average daily weight: {weights.mean():.4f}")
    if len(weights) > 0:
        print(f"Average daily allocation: ${budget / len(weights):,.2f}")

    print("\nDaily Breakdown:")
    print("-" * 90)
    print(
        f"{'Date':<12} {'Weight':<8} {'Weight %':<10} {'USD Amount':<12} {'BTC Price':<12} {'BTC Amount':<12}"
    )
    print("-" * 90)

    total_btc = 0
    for date, weight in weights.items():
        weight_pct = weight * 100
        daily_usd = budget * weight

        # Get BTC price for this date if available
        price_str = "N/A"
        btc_amount_str = "N/A"
        if date in btc_df.index:
            price = btc_df.loc[date, "PriceUSD"]
            if pd.notna(price):
                price_str = f"${price:.2f}"
                btc_amount = daily_usd / price
                btc_amount_str = f"{btc_amount:.6f}"
                total_btc += btc_amount

        print(
            f"{date.strftime('%Y-%m-%d'):<12} {weight:<8.4f} {weight_pct:<10.2f}% ${daily_usd:<11.2f} {price_str:<12} {btc_amount_str:<12}"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<12} {weights.sum():<8.4f} {weights.sum() * 100:<10.2f}% ${budget:<11.2f} {'':12} {total_btc:<12.6f}"
    )


def save_weights_to_csv(
    budget: float, start_date: str, end_date: str, filename: str = None
) -> str:
    """
    Save weights and allocations to CSV file.

    Args:
        budget: Total USD budget to allocate
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        filename: Optional filename. If None, uses default naming.

    Returns:
        Path to the saved CSV file
    """
    if filename is None:
        filename = f"weights_{start_date}_to_{end_date}.csv"

    weights = get_weights_for_period(start_date, end_date)
    btc_df = load_data()

    # Create DataFrame with weights, allocations, and prices
    df = pd.DataFrame(
        {
            "date": weights.index,
            "weight": weights.values,
            "weight_percent": weights.values * 100,
            "usd_allocation": weights.values * budget,
        }
    )

    # Add BTC prices and BTC amounts where available
    prices = []
    btc_amounts = []
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

    df.to_csv(filename)
    logger.info(f"Saved weights and allocations to {filename}")
    return filename


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate Bitcoin allocation weights for a specific period and budget"
    )
    parser.add_argument(
        "budget", type=float, help="Total USD budget to allocate across the period"
    )
    parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save weights to CSV file"
    )
    parser.add_argument("--filename", "-f", type=str, help="CSV filename (optional)")

    args = parser.parse_args()

    try:
        # Display weights
        display_weights(args.budget, args.start_date, args.end_date)

        # Save to CSV if requested
        if args.save:
            csv_file = save_weights_to_csv(
                args.budget, args.start_date, args.end_date, args.filename
            )
            print(f"\nWeights saved to: {csv_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
