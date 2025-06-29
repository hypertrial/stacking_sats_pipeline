"""
Tutorial examples demonstrating the new PyPI library experience for backtesting strategies.
Marimo notebook style with reactive cells for interactive strategy development.

These examples follow the same structure as strategy_template.py to ensure proper validation.
All strategies use the cycle-based weight allocation pattern for compliance.

To run this as a marimo notebook:
1. Install marimo: pip install marimo
2. Run: marimo edit tutorials/examples.py
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    """Setup and imports cell"""
    import os
    import sys

    import numpy as np
    import pandas as pd

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from backtest import backtest, quick_backtest, strategy
    from config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS

    print("🚀 Stacking Sats Pipeline - Interactive Tutorial")
    print("=" * 60)
    print("📊 Ready for strategy development and backtesting!")

    return (
        BACKTEST_END,
        BACKTEST_START,
        CYCLE_YEARS,
        backtest,
        np,
        pd,
        quick_backtest,
        strategy,
    )


@app.cell
def _(BACKTEST_END, BACKTEST_START, CYCLE_YEARS, pd):
    """Uniform Strategy (Baseline) - Perfect DCA benchmark"""

    def construct_features_uniform(df):
        """Construct features for uniform strategy (just returns price data)."""
        df = df.copy()
        return df[["PriceUSD"]]

    def uniform_strategy(df, *, cycle_years=CYCLE_YEARS):
        """
        Uniform allocation strategy following the template pattern.
        This should match the uniform DCA benchmark exactly.
        """
        # Features
        df_feat = construct_features_uniform(df).loc[BACKTEST_START:BACKTEST_END]
        weights = pd.Series(index=df_feat.index, dtype=float)

        start_year = pd.to_datetime(BACKTEST_START).year
        cycle_labels = df_feat.index.to_series().apply(
            lambda ts: (ts.year - start_year) // cycle_years
        )

        # Loop over cycles - uniform allocation
        for _, cycle_df in df_feat.groupby(cycle_labels):
            N = len(cycle_df)
            base_wt = 1.0 / N
            weights.loc[cycle_df.index] = base_wt

        return weights

    print("✅ Uniform Strategy defined (baseline)")
    return (uniform_strategy,)


@app.cell
def _(BACKTEST_END, BACKTEST_START, CYCLE_YEARS, np, pd):
    """Simple Moving Average Strategy"""

    def construct_features_ma(df):
        """
        Construct features for the moving average strategy.
        Uses only past data for calculations to avoid look-ahead bias.
        """
        df = df.copy()
        df = df[["PriceUSD"]]

        # Shift the PriceUSD column by one to use only past data
        past_price = df["PriceUSD"].shift(1)
        # Calculate 200-day moving average
        df["ma200"] = past_price.rolling(window=200, min_periods=1).mean()
        return df

    def simple_ma_strategy(df, *, cycle_years=CYCLE_YEARS):
        """
        Simple moving average strategy following the template pattern.
        Buy more when price is below 200-day MA.
        """
        # Parameters
        BOOST_FACTOR = 1.5  # Buy 50% more when below MA

        # Features
        df_feat = construct_features_ma(df).loc[BACKTEST_START:BACKTEST_END]
        weights = pd.Series(index=df_feat.index, dtype=float)

        start_year = pd.to_datetime(BACKTEST_START).year
        cycle_labels = df_feat.index.to_series().apply(
            lambda ts: (ts.year - start_year) // cycle_years
        )

        # Loop over cycles
        for _, cycle_df in df_feat.groupby(cycle_labels):
            N = len(cycle_df)
            base_wt = 1.0 / N
            temp_wt = np.full(N, base_wt)

            for day in range(N):
                price, ma = (
                    cycle_df["PriceUSD"].iat[day],
                    cycle_df["ma200"].iat[day],
                )

                # Simple logic: buy more when below MA
                if not pd.isna(ma) and price < ma:
                    temp_wt[day] *= BOOST_FACTOR

            # Normalize to sum to 1 within the cycle
            temp_wt = temp_wt / temp_wt.sum()
            weights.loc[cycle_df.index] = temp_wt

        return weights

    print("✅ Simple MA Strategy defined")
    return (simple_ma_strategy,)


@app.cell
def _(BACKTEST_END, BACKTEST_START, CYCLE_YEARS, np, pd, strategy):
    """RSI Strategy with Decorator"""

    def construct_features_rsi(df):
        """
        Construct features for the RSI strategy.
        Uses only past data for calculations to avoid look-ahead bias.
        """
        df = df.copy()
        df = df[["PriceUSD"]]

        # Calculate RSI using past data
        past_price = df["PriceUSD"].shift(1)
        delta = past_price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @strategy(name="RSI Strategy", auto_backtest=False)
    def rsi_strategy(df, *, cycle_years=CYCLE_YEARS):
        """
        RSI-based strategy following the template pattern.
        Buy more when oversold (RSI < 30), less when overbought (RSI > 70).
        """
        # Parameters
        OVERSOLD_BOOST = 1.8
        OVERBOUGHT_REDUCE = 0.6

        # Features
        df_feat = construct_features_rsi(df).loc[BACKTEST_START:BACKTEST_END]
        weights = pd.Series(index=df_feat.index, dtype=float)

        start_year = pd.to_datetime(BACKTEST_START).year
        cycle_labels = df_feat.index.to_series().apply(
            lambda ts: (ts.year - start_year) // cycle_years
        )

        # Loop over cycles
        for _, cycle_df in df_feat.groupby(cycle_labels):
            N = len(cycle_df)
            base_wt = 1.0 / N
            temp_wt = np.full(N, base_wt)

            for day in range(N):
                rsi = cycle_df["rsi"].iat[day]

                if not pd.isna(rsi):
                    if rsi < 30:  # Oversold
                        temp_wt[day] *= OVERSOLD_BOOST
                    elif rsi > 70:  # Overbought
                        temp_wt[day] *= OVERBOUGHT_REDUCE

            # Normalize to sum to 1 within the cycle
            temp_wt = temp_wt / temp_wt.sum()
            weights.loc[cycle_df.index] = temp_wt

        return weights

    print("✅ RSI Strategy defined (with decorator)")
    return (rsi_strategy,)


@app.cell
def _(BACKTEST_END, BACKTEST_START, CYCLE_YEARS, np, pd, strategy):
    """Combined Multi-Indicator Strategy"""

    def construct_features_combined(df):
        """
        Construct features for the combined strategy.
        Uses only past data for calculations to avoid look-ahead bias.
        """
        df = df.copy()
        df = df[["PriceUSD"]]

        # Shift the PriceUSD column by one to use only past data
        past_price = df["PriceUSD"].shift(1)

        # Moving averages
        df["ma50"] = past_price.rolling(window=50, min_periods=1).mean()
        df["ma200"] = past_price.rolling(window=200, min_periods=1).mean()

        # Bollinger Bands
        df["bb_middle"] = past_price.rolling(window=20, min_periods=1).mean()
        df["bb_std"] = past_price.rolling(window=20, min_periods=1).std()
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

        return df

    @strategy(name="Combined Strategy")
    def combined_strategy(df, *, cycle_years=CYCLE_YEARS):
        """
        Combined strategy using multiple indicators following the template pattern.
        """
        # Parameters
        TREND_BOOST = 1.3  # When in downtrend
        TREND_REDUCE = 0.8  # When in uptrend
        BB_BOOST = 1.5  # When below lower Bollinger Band

        # Features
        df_feat = construct_features_combined(df).loc[BACKTEST_START:BACKTEST_END]
        weights = pd.Series(index=df_feat.index, dtype=float)

        start_year = pd.to_datetime(BACKTEST_START).year
        cycle_labels = df_feat.index.to_series().apply(
            lambda ts: (ts.year - start_year) // cycle_years
        )

        # Loop over cycles
        for _, cycle_df in df_feat.groupby(cycle_labels):
            N = len(cycle_df)
            base_wt = 1.0 / N
            temp_wt = np.full(N, base_wt)

            for day in range(N):
                price, ma50, ma200, bb_lower = (
                    cycle_df["PriceUSD"].iat[day],
                    cycle_df["ma50"].iat[day],
                    cycle_df["ma200"].iat[day],
                    cycle_df["bb_lower"].iat[day],
                )

                multiplier = 1.0

                # MA trend signal
                if not pd.isna(ma50) and not pd.isna(ma200):
                    if ma50 > ma200:  # Uptrend
                        multiplier *= TREND_REDUCE
                    else:  # Downtrend
                        multiplier *= TREND_BOOST

                # Bollinger Band signal
                if not pd.isna(bb_lower) and price < bb_lower:
                    multiplier *= BB_BOOST

                temp_wt[day] *= multiplier

            # Normalize to sum to 1 within the cycle
            temp_wt = temp_wt / temp_wt.sum()
            weights.loc[cycle_df.index] = temp_wt

        return weights

    print("✅ Combined Strategy defined (with decorator)")
    return (combined_strategy,)


@app.cell
def _(backtest, uniform_strategy):
    """Run Baseline - Uniform Strategy Backtest"""
    print("🔍 Running Uniform Strategy (baseline)...")

    results_uniform = backtest(uniform_strategy, strategy_name="Uniform Strategy")

    print("\n📊 UNIFORM STRATEGY RESULTS")
    print("=" * 40)
    results_uniform.summary()

    return


@app.cell
def _(backtest, simple_ma_strategy):
    """Run Simple MA Strategy Backtest"""
    print("🔍 Running Simple MA Strategy...")

    results_ma = backtest(simple_ma_strategy, strategy_name="Simple MA Strategy")

    print("\n📊 SIMPLE MA STRATEGY RESULTS")
    print("=" * 40)
    results_ma.summary()

    return


@app.cell
def _(rsi_strategy):
    """Run RSI Strategy using Decorator Method"""
    print("🔍 Running RSI Strategy (decorator method)...")

    results_rsi = rsi_strategy.backtest()

    print("\n📊 RSI STRATEGY RESULTS")
    print("=" * 40)
    results_rsi.summary()

    return


@app.cell
def _(combined_strategy):
    """Run Combined Strategy using Decorator Method"""
    print("🔍 Running Combined Strategy (decorator method)...")

    results_combined = combined_strategy.backtest()

    print("\n📊 COMBINED STRATEGY RESULTS")
    print("=" * 40)
    results_combined.summary()

    return


@app.cell
def _(
    combined_strategy,
    pd,
    quick_backtest,
    rsi_strategy,
    simple_ma_strategy,
    uniform_strategy,
):
    """Quick Performance Comparison"""
    print("⚡ Quick Strategy Comparison")
    print("=" * 40)

    # Run quick backtests for comparison
    uniform_excess = quick_backtest(uniform_strategy)
    ma_excess = quick_backtest(simple_ma_strategy)
    rsi_excess = quick_backtest(rsi_strategy)
    combined_excess = quick_backtest(combined_strategy)

    # Create comparison table
    comparison_data = {
        "Strategy": ["Uniform", "Simple MA", "RSI", "Combined"],
        "Excess SPD (%)": [uniform_excess, ma_excess, rsi_excess, combined_excess],
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Excess SPD (%)", ascending=False)

    print("\n🏆 STRATEGY RANKING (by Excess SPD)")
    print(comparison_df.to_string(index=False))

    # Best strategy
    best_strategy = comparison_df.iloc[0]["Strategy"]
    best_performance = comparison_df.iloc[0]["Excess SPD (%)"]

    print(f"\n🥇 Best Strategy: {best_strategy} (+{best_performance:.2f}%)")

    return (comparison_df,)


@app.cell
def _(backtest, simple_ma_strategy):
    """Custom Parameters Example"""
    print("🔧 Custom Parameters Example - MA Strategy (2022-2024)")
    print("=" * 50)

    results_custom = backtest(
        simple_ma_strategy,
        start_date="2022-01-01",
        end_date="2024-01-01",
        cycle_years=1,
        strategy_name="MA Strategy (2022-2024)",
    )

    print("\n📊 CUSTOM PERIOD RESULTS")
    print("=" * 40)
    results_custom.summary()

    return


@app.cell
def _(comparison_df):
    """Results Summary and Next Steps"""
    print("🎉 TUTORIAL COMPLETE!")
    print("=" * 50)

    print("\n📈 SUMMARY OF ALL RESULTS:")
    print("-" * 30)
    for strategy_name, excess in zip(
        comparison_df["Strategy"], comparison_df["Excess SPD (%)"], strict=False
    ):
        emoji = "🥇" if strategy_name == comparison_df.iloc[0]["Strategy"] else "📊"
        print(f"{emoji} {strategy_name}: {excess:+.2f}% vs uniform DCA")

    print("\n🔍 KEY INSIGHTS:")
    print("• All strategies follow the strategy_template.py pattern")
    print("• construct_features() functions use only past data")
    print("• Weights are normalized per cycle to sum to 1.0")
    print("• Strategies show varying performance across different periods")

    print("\n🚀 NEXT STEPS:")
    print("• Modify strategy parameters in cells above and see results update")
    print("• Try: results_ma.plot() to generate visualizations")
    print("• Access detailed data: results_ma.spd_table")
    print("• Check validation: results_ma.validation")
    print("• Create your own strategy by modifying the cells above!")

    print("\n💡 INTERACTIVE FEATURES:")
    print("• Change parameters in any cell and see results update automatically")
    print("• Experiment with different indicators and logic")
    print("• Compare strategies side-by-side in real-time")

    # Available results objects
    available_results = {
        "results_uniform": "Uniform strategy results",
        "results_ma": "Simple MA strategy results",
        "results_rsi": "RSI strategy results",
        "results_combined": "Combined strategy results",
    }

    print("\n📊 AVAILABLE RESULT OBJECTS:")
    for var, desc in available_results.items():
        print(f"• {var}: {desc}")

    return


if __name__ == "__main__":
    app.run()
