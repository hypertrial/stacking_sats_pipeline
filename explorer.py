#!/usr/bin/env python3
"""
Explorer script for the merged cryptocurrency and financial data.
"""

from pathlib import Path

import pandas as pd


def explore_parquet_data():
    """Load and explore the merged parquet data."""

    # Load the parquet file
    parquet_file = Path("merged_crypto_data.parquet")

    if not parquet_file.exists():
        print(f"❌ Parquet file not found: {parquet_file}")
        print("Run 'stacking-sats --extract-data parquet' first to create the file.")
        return

    print(f"🔍 Loading data from: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # Assert daily-level data integrity
    print("\n🔍 Validating daily-level data structure...")

    # Check that all timestamps are at midnight UTC
    non_midnight_count = (df.index.hour != 0).sum()
    assert non_midnight_count == 0, (
        f"Found {non_midnight_count} non-midnight timestamps! All data should be at daily level."
    )

    # Check for duplicate dates
    duplicate_dates = df.index.duplicated().sum()
    assert duplicate_dates == 0, (
        f"Found {duplicate_dates} duplicate dates! Each day should appear only once."
    )

    # Calculate expected vs actual date coverage
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D", tz="UTC")
    total_possible_days = len(date_range)
    actual_days = len(df)
    coverage_pct = (actual_days / total_possible_days) * 100

    print("✅ Daily data validation passed!")
    print(f"   Total possible days in range: {total_possible_days:,}")
    print(f"   Actual days with data: {actual_days:,}")
    print(f"   Date coverage: {coverage_pct:.1f}%")

    # Assert reasonable coverage (at least 50% of days should have some data)
    assert coverage_pct >= 50.0, (
        f"Date coverage too low ({coverage_pct:.1f}%)! Expected at least 50% of days to have data."
    )

    print("\n📊 Dataset Overview:")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"   Date Range: {df.index.min()} to {df.index.max()}")
    print(f"   Index Type: {type(df.index).__name__}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print("\n📅 HEAD (First 5 rows):")
    print("=" * 80)
    print(df.head())

    print("\n📅 TAIL (Last 5 rows):")
    print("=" * 80)
    print(df.tail())

    print("\n🏷️  Column Categories:")
    print("=" * 80)

    # Group columns by source/category
    coinmetrics_cols = [col for col in df.columns if "coinmetrics" in col.lower()]
    feargreed_cols = [col for col in df.columns if "feargreed" in col.lower()]
    yfinance_cols = [
        col
        for col in df.columns
        if any(symbol in col for symbol in ["BTC-USD", "ETH-USD", "AAPL", "SPY", "GLD"])
    ]
    fred_cols = [col for col in df.columns if "fred" in col.lower()]

    print(f"   CoinMetrics columns: {len(coinmetrics_cols)}")
    if coinmetrics_cols:
        print(f"      Sample: {coinmetrics_cols[:3]}")

    print(f"   Fear & Greed columns: {len(feargreed_cols)}")
    if feargreed_cols:
        print(f"      Sample: {feargreed_cols[:3]}")

    print(f"   Yahoo Finance columns: {len(yfinance_cols)}")
    if yfinance_cols:
        print(f"      Sample: {yfinance_cols[:3]}")

    print(f"   FRED columns: {len(fred_cols)}")
    if fred_cols:
        print(f"      Sample: {fred_cols[:3]}")

    print("\n📈 Sample Data Types:")
    print("=" * 80)
    print(df.dtypes.value_counts())

    print("\n📊 Missing Data Summary:")
    print("=" * 80)
    missing_data = df.isnull().sum().sort_values(ascending=False)
    top_missing = missing_data.head(10)
    if top_missing.sum() > 0:
        print("Top 10 columns with missing data:")
        for col, missing_count in top_missing.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                print(f"   {col}: {missing_count:,} ({percentage:.1f}%)")
    else:
        print("✅ No missing data found!")


if __name__ == "__main__":
    explore_parquet_data()
