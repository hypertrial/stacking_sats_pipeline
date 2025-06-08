"""
Bitcoin accumulation simulation comparing uniform DCA vs custom strategies.

This module simulates how much Bitcoin would be accumulated over time
with a fixed annual budget using different DCA strategies.
"""

from typing import Callable

import pandas as pd

from config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS


def simulate_bitcoin_accumulation(
    df: pd.DataFrame,
    strategy_fn: Callable,
    annual_budget_usd: float = 10_000_000,
    *,
    cycle_years: int = CYCLE_YEARS,
) -> pd.DataFrame:
    """
    Simulate Bitcoin accumulation using uniform DCA vs custom strategy.

    Args:
        df: DataFrame with Bitcoin price data
        strategy_fn: Function that computes weights for the custom strategy
        annual_budget_usd: Annual budget in USD (default: $10,000,000)
        cycle_years: Length of each cycle in years

    Returns:
        DataFrame with simulation results by cycle
    """
    df_backtest = df.loc[BACKTEST_START:BACKTEST_END]
    cycle_length = pd.DateOffset(years=cycle_years)

    # Get strategy weights
    strategy_weights = strategy_fn(df)

    current = df_backtest.index.min()
    results = []

    while current <= df_backtest.index.max():
        cycle_end = current + cycle_length - pd.Timedelta(days=1)
        end_date = min(cycle_end, df_backtest.index.max())

        cycle_mask = (df_backtest.index >= current) & (df_backtest.index <= end_date)
        cycle_data = df_backtest.loc[cycle_mask]

        if cycle_data.empty:
            break

        # Create cycle label
        if cycle_years == 1:
            cycle_label = f"{current.year}"
        else:
            cycle_label = f"{current.year}–{end_date.year}"

        # Get weights for this cycle
        cycle_weights = strategy_weights.loc[cycle_data.index]
        uniform_weights = pd.Series(1.0 / len(cycle_data), index=cycle_data.index)

        # Calculate daily USD amounts
        strategy_daily_usd = cycle_weights * annual_budget_usd
        uniform_daily_usd = uniform_weights * annual_budget_usd

        # Calculate Bitcoin purchased each day
        prices = cycle_data["PriceUSD"]
        strategy_btc_daily = strategy_daily_usd / prices
        uniform_btc_daily = uniform_daily_usd / prices

        # Total Bitcoin accumulated in this cycle
        strategy_btc_total = strategy_btc_daily.sum()
        uniform_btc_total = uniform_btc_daily.sum()

        # Calculate average price paid
        strategy_avg_price = annual_budget_usd / strategy_btc_total
        uniform_avg_price = annual_budget_usd / uniform_btc_total

        # Calculate price statistics for the cycle
        min_price = prices.min()
        max_price = prices.max()
        mean_price = prices.mean()

        # Calculate how much better/worse the strategy performed
        btc_difference = strategy_btc_total - uniform_btc_total
        btc_improvement_pct = (btc_difference / uniform_btc_total) * 100

        results.append(
            {
                "cycle": cycle_label,
                "annual_budget_usd": annual_budget_usd,
                "uniform_btc_accumulated": uniform_btc_total,
                "strategy_btc_accumulated": strategy_btc_total,
                "btc_difference": btc_difference,
                "btc_improvement_pct": btc_improvement_pct,
                "uniform_avg_price": uniform_avg_price,
                "strategy_avg_price": strategy_avg_price,
                "cycle_min_price": min_price,
                "cycle_max_price": max_price,
                "cycle_mean_price": mean_price,
            }
        )

        current += cycle_length

    return pd.DataFrame(results).set_index("cycle")


def print_simulation_summary(
    simulation_results: pd.DataFrame, strategy_label: str = "Custom Strategy"
) -> None:
    """
    Print a summary of the Bitcoin accumulation simulation results.

    Args:
        simulation_results: DataFrame from simulate_bitcoin_accumulation
        strategy_label: Name of the strategy for display
    """
    total_uniform_btc = simulation_results["uniform_btc_accumulated"].sum()
    total_strategy_btc = simulation_results["strategy_btc_accumulated"].sum()
    total_difference = total_strategy_btc - total_uniform_btc
    total_improvement_pct = (total_difference / total_uniform_btc) * 100

    annual_budget = simulation_results["annual_budget_usd"].iloc[0]
    total_invested = annual_budget * len(simulation_results)

    print(f"\n{'=' * 60}")
    print("BITCOIN ACCUMULATION SIMULATION RESULTS")
    print(f"Strategy: {strategy_label}")
    print(f"Annual Budget: ${annual_budget:,.0f}")
    print(f"Total Invested: ${total_invested:,.0f}")
    print(f"Simulation Period: {len(simulation_results)} cycles")
    print(f"{'=' * 60}")

    print("\nTOTAL BITCOIN ACCUMULATED:")
    print(f"  Uniform DCA:     {total_uniform_btc:.8f} BTC")
    print(f"  {strategy_label}: {total_strategy_btc:.8f} BTC")
    print(f"  Difference:      {total_difference:.8f} BTC")
    print(f"  Improvement:     {total_improvement_pct:+.2f}%")

    # Calculate average prices
    avg_uniform_price = total_invested / total_uniform_btc
    avg_strategy_price = total_invested / total_strategy_btc

    print("\nAVERAGE PRICE PAID:")
    print(f"  Uniform DCA:     ${avg_uniform_price:,.2f}")
    print(f"  {strategy_label}: ${avg_strategy_price:,.2f}")
    print(f"  Price Difference: ${avg_strategy_price - avg_uniform_price:+,.2f}")

    print("\nPER-CYCLE BREAKDOWN:")
    print(
        f"{'Cycle':<8} {'Uniform DCA':<15} {'Custom Strategy':<15} {'Excess Bitcoin':<15}"
    )
    print(f"{'':8} {'(BTC)':<15} {'(BTC)':<15} {'(BTC)':<15}")
    print(f"{'-' * 65}")

    for cycle, row in simulation_results.iterrows():
        print(
            f"{cycle:<8} {row['uniform_btc_accumulated']:<15.6f} "
            f"{row['strategy_btc_accumulated']:<15.6f} "
            f"{row['btc_difference']:<15.6f}"
        )

    # Summary statistics
    improvements = simulation_results["btc_improvement_pct"]
    print("\nIMPROVEMENT STATISTICS:")
    print(f"  Best cycle:      {improvements.max():+.2f}%")
    print(f"  Worst cycle:     {improvements.min():+.2f}%")
    print(f"  Average:         {improvements.mean():+.2f}%")
    print(f"  Median:          {improvements.median():+.2f}%")
    print(f"  Std deviation:   {improvements.std():.2f}%")


def run_full_simulation(
    df: pd.DataFrame,
    strategy_fn: Callable,
    strategy_label: str = "Custom Strategy",
    annual_budget_usd: float = 10_000_000,
    *,
    cycle_years: int = CYCLE_YEARS,
) -> pd.DataFrame:
    """
    Run complete Bitcoin accumulation simulation and print results.

    Args:
        df: DataFrame with Bitcoin price data
        strategy_fn: Function that computes weights for the custom strategy
        strategy_label: Name of the strategy for display
        annual_budget_usd: Annual budget in USD
        cycle_years: Length of each cycle in years

    Returns:
        DataFrame with detailed simulation results
    """
    results = simulate_bitcoin_accumulation(
        df, strategy_fn, annual_budget_usd, cycle_years=cycle_years
    )

    print_simulation_summary(results, strategy_label)

    return results
