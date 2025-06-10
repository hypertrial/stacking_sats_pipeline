"""
Simple backtest runner for strategy functions.
Provides a clean PyPI library experience for backtesting.
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd

try:
    # Try relative imports first (when used as a package)
    from ..config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS
    from ..data.data_loader import load_data, validate_price_data
    from .checks import backtest_dynamic_dca, validate_strategy_comprehensive
except ImportError:
    # Fall back to absolute imports (when run directly)
    from backtest.checks import backtest_dynamic_dca, validate_strategy_comprehensive
    from data.data_loader import load_data, validate_price_data

    from ..config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS


class BacktestResults:
    """Container for backtest results with convenient access methods."""

    def __init__(
        self, strategy_fn: Callable, df: pd.DataFrame, results: Dict[str, Any]
    ):
        self.strategy_fn = strategy_fn
        self.df = df
        self.results = results
        self.weights = strategy_fn(df)
        # Store backtest period weights for easy access
        self._backtest_weights = None

    @property
    def backtest_weights(self) -> pd.Series:
        """Get weights filtered to the backtest period."""
        if self._backtest_weights is None:
            self._backtest_weights = self.weights.loc[self.df.index]
        return self._backtest_weights

    @property
    def spd_table(self) -> pd.DataFrame:
        """Get the SPD comparison table."""
        return self.results.get("spd_table")

    @property
    def validation(self) -> Dict[str, Any]:
        """Get validation results."""
        return self.results.get("validation", {})

    @property
    def passed_validation(self) -> bool:
        """Check if strategy passed all validation checks."""
        return self.validation.get("validation_passed", False)

    def save_weights_to_csv(
        self, filename: Optional[str] = None, budget: Optional[float] = None
    ) -> str:
        """
        Save backtest period weights to CSV file.

        Args:
            filename: Output filename. If None, auto-generates based on strategy name and dates
            budget: Optional budget to calculate USD allocations. If None, uses normalized weights

        Returns:
            Path to the saved CSV file
        """

        # Auto-generate filename if not provided
        if filename is None:
            strategy_name = getattr(self.strategy_fn, "__name__", "strategy")
            start_date = self.df.index.min().strftime("%Y-%m-%d")
            end_date = self.df.index.max().strftime("%Y-%m-%d")
            filename = f"{strategy_name}_weights_{start_date}_to_{end_date}.csv"

        weights = self.backtest_weights

        # Create DataFrame with weights and metadata
        df_export = pd.DataFrame(
            {
                "date": weights.index,
                "weight": weights.values,
                "weight_percent": (weights.values / weights.sum()) * 100,
            }
        )

        # Add budget-based allocations if budget provided
        if budget is not None:
            df_export["usd_allocation"] = weights.values * budget

            # Add BTC prices and amounts if available
            if "PriceUSD" in self.df.columns:
                prices = []
                btc_amounts = []
                for date in weights.index:
                    if date in self.df.index:
                        price = self.df.loc[date, "PriceUSD"]
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

                df_export["btc_price_usd"] = prices
                df_export["btc_amount"] = btc_amounts

        df_export.set_index("date", inplace=True)
        df_export.to_csv(filename)

        print(f"Model weights saved to: {filename}")
        return filename

    def display_weight_statistics(self, budget: Optional[float] = None):
        """
        Display statistical summary of the model weights for the backtest period.

        Args:
            budget: Optional budget to display allocation amounts
        """
        weights = self.backtest_weights

        print(f"\n{'=' * 60}")
        print("MODEL WEIGHTS STATISTICS")
        print(f"Strategy: {getattr(self.strategy_fn, '__name__', 'Strategy')}")
        print(
            f"Period: {self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}"
        )
        print(f"{'=' * 60}")

        print(f"Total days: {len(weights)}")
        print(f"Total weight sum: {weights.sum():.6f}")
        print(f"Mean daily weight: {weights.mean():.6f}")
        print(f"Median daily weight: {weights.median():.6f}")
        print(f"Min daily weight: {weights.min():.6f}")
        print(f"Max daily weight: {weights.max():.6f}")
        print(f"Weight std deviation: {weights.std():.6f}")

        if budget is not None:
            print(f"\nWith budget of ${budget:,.2f}:")
            print(f"Average daily allocation: ${(budget * weights.mean()):,.2f}")
            print(f"Min daily allocation: ${(budget * weights.min()):,.2f}")
            print(f"Max daily allocation: ${(budget * weights.max()):,.2f}")

            if "PriceUSD" in self.df.columns:
                # Calculate total BTC that would be accumulated
                total_btc = 0
                for date in weights.index:
                    if date in self.df.index:
                        price = self.df.loc[date, "PriceUSD"]
                        if pd.notna(price):
                            daily_usd = weights.loc[date] * budget
                            total_btc += daily_usd / price

                print(f"Total BTC accumulated: {total_btc:.8f}")
                print(f"Average price paid: ${budget / total_btc:.2f}")

    def export_weights_summary(self, filename: Optional[str] = None) -> str:
        """
        Export a summary of model weights by cycle to CSV.

        Args:
            filename: Output filename. If None, auto-generates

        Returns:
            Path to the saved CSV file
        """
        if filename is None:
            strategy_name = getattr(self.strategy_fn, "__name__", "strategy")
            filename = f"{strategy_name}_weights_summary.csv"

        try:
            from ..config import CYCLE_YEARS
        except ImportError:
            CYCLE_YEARS = 4  # Default fallback

        weights = self.backtest_weights

        # Group weights by cycle
        start_year = weights.index.min().year
        cycle_labels = weights.index.to_series().apply(
            lambda ts: (ts.year - start_year) // CYCLE_YEARS
        )

        summary_data = []
        for cycle_id, cycle_weights in weights.groupby(cycle_labels):
            cycle_start = cycle_weights.index.min()
            cycle_end = cycle_weights.index.max()

            summary_data.append(
                {
                    "cycle": f"Cycle {cycle_id}",
                    "start_date": cycle_start.strftime("%Y-%m-%d"),
                    "end_date": cycle_end.strftime("%Y-%m-%d"),
                    "days": len(cycle_weights),
                    "total_weight": cycle_weights.sum(),
                    "mean_weight": cycle_weights.mean(),
                    "median_weight": cycle_weights.median(),
                    "min_weight": cycle_weights.min(),
                    "max_weight": cycle_weights.max(),
                    "std_weight": cycle_weights.std(),
                }
            )

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(filename, index=False)

        print(f"Weights summary saved to: {filename}")
        return filename

    def get_weights_dataframe(self, budget: Optional[float] = None) -> pd.DataFrame:
        """
        Get model weights as a pandas DataFrame with additional computed columns.

        Args:
            budget: Optional budget to calculate USD allocations

        Returns:
            DataFrame with weights and computed columns
        """
        weights = self.backtest_weights

        df = pd.DataFrame(
            {
                "date": weights.index,
                "weight": weights.values,
                "weight_percent": (weights.values / weights.sum()) * 100,
            }
        )

        # Add budget-based columns if budget provided
        if budget is not None:
            df["usd_allocation"] = weights.values * budget

            # Add BTC data if available
            if "PriceUSD" in self.df.columns:
                df["btc_price"] = [
                    self.df.loc[date, "PriceUSD"] if date in self.df.index else None
                    for date in weights.index
                ]
                df["btc_amount"] = [
                    (weights.loc[date] * budget) / self.df.loc[date, "PriceUSD"]
                    if date in self.df.index and pd.notna(self.df.loc[date, "PriceUSD"])
                    else None
                    for date in weights.index
                ]

        return df.set_index("date")

    def plot(self, show: bool = True):
        """Generate plots for the backtest results."""
        try:
            from ..plot.plotting import (
                plot_features,
                plot_final_weights,
                plot_spd_comparison,
                plot_weight_sums_by_cycle,
            )
        except ImportError:
            from plot.plotting import (
                plot_features,
                plot_final_weights,
                plot_spd_comparison,
                plot_weight_sums_by_cycle,
            )

        plot_features(
            self.df,
            weights=self.weights,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
        )
        plot_final_weights(self.weights, start_date=BACKTEST_START)
        plot_spd_comparison(self.spd_table)
        plot_weight_sums_by_cycle(self.weights, start_date=BACKTEST_START)

    def summary(self):
        """Print a summary of backtest results."""
        print(f"\n{'=' * 60}")
        print("BACKTEST SUMMARY")
        print(f"{'=' * 60}")

        if self.passed_validation:
            print("✅ Strategy passed all validation checks")
        else:
            print("❌ Strategy failed validation checks")

        if self.spd_table is not None:
            mean_excess = self.spd_table["excess_pct"].mean()
            print(f"Mean excess SPD vs uniform DCA: {mean_excess:.2f}%")
            print(f"Number of cycles tested: {len(self.spd_table)}")

        # Add information about model weights
        weights = self.backtest_weights
        print(f"\nModel weights computed for {len(weights)} days")
        print(f"Weight range: {weights.min():.6f} to {weights.max():.6f}")
        print(f"Mean daily weight: {weights.mean():.6f}")

        print("\n💡 To export model weights:")
        print("   results.save_weights_to_csv()  # Save to CSV")
        print("   results.display_weight_statistics()  # Show statistics")
        print("   results.export_weights_summary()  # Save cycle summary")


def backtest(
    strategy_fn: Callable,
    *,
    data: Optional[pd.DataFrame] = None,
    start_date: str = BACKTEST_START,
    end_date: str = BACKTEST_END,
    cycle_years: int = CYCLE_YEARS,
    validate: bool = True,
    verbose: bool = True,
    strategy_name: Optional[str] = None,
    export_weights: bool = False,
    export_dir: Optional[str] = None,
    export_budget: Optional[float] = None,
) -> BacktestResults:
    """
    Backtest a strategy function with a clean, simple interface.

    Args:
        strategy_fn: Function that computes weights from price data
        data: Optional DataFrame with price data. If None, loads default data
        start_date: Start date for backtesting
        end_date: End date for backtesting
        cycle_years: Length of investment cycles in years
        validate: Whether to run comprehensive validation
        verbose: Whether to print results
        strategy_name: Name for the strategy (defaults to function name)
        export_weights: Whether to automatically export weights to CSV files
        export_dir: Directory to export weights to (defaults to 'stacking_sats_pipeline/data')
        export_budget: Budget to use for weight export calculations (optional)

    Returns:
        BacktestResults object with all results and convenience methods

    Example:
        >>> def my_strategy(df):
        ...     # Your strategy logic here
        ...     return weights
        >>>
        >>> results = backtest(my_strategy)
        >>> results.summary()
        >>> results.plot()
        >>>
        >>> # Auto-export weights to data directory
        >>> results = backtest(my_strategy, export_weights=True, export_budget=10_000_000)
    """

    # Load data if not provided
    if data is None:
        data = load_data()
        validate_price_data(data)

    # Filter to backtest period
    df_backtest = data.loc[start_date:end_date]

    # Get strategy name
    if strategy_name is None:
        strategy_name = getattr(strategy_fn, "__name__", "Strategy")

    results = {}

    # Run validation if requested
    if validate:
        if verbose:
            print(f"Running validation for {strategy_name}...")
        validation_results = validate_strategy_comprehensive(
            df_backtest, strategy_fn, cycle_years=cycle_years, return_details=True
        )
        results["validation"] = validation_results

        if verbose:
            if validation_results["validation_passed"]:
                print("✅ Validation passed")
            else:
                print("❌ Validation failed - check results.validation for details")

    # Run backtest
    if verbose:
        print(f"Running backtest for {strategy_name}...")

    spd_results = backtest_dynamic_dca(
        df_backtest, strategy_fn, strategy_label=strategy_name, cycle_years=cycle_years
    )
    results["spd_table"] = spd_results

    # Create BacktestResults object
    backtest_results = BacktestResults(strategy_fn, df_backtest, results)

    # Auto-export weights if requested
    if export_weights:
        from pathlib import Path

        # Set default export directory
        if export_dir is None:
            export_dir = "stacking_sats_pipeline/data"

        # Create directory if it doesn't exist
        Path(export_dir).mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n📊 Auto-exporting weights to {export_dir}/...")

        # Export basic weights
        basic_filename = (
            f"{export_dir}/{strategy_name.lower().replace(' ', '_')}_weights.csv"
        )
        backtest_results.save_weights_to_csv(filename=basic_filename)

        # Export weights with budget if provided
        if export_budget is not None:
            budget_filename = f"{export_dir}/{strategy_name.lower().replace(' ', '_')}_weights_budget.csv"
            backtest_results.save_weights_to_csv(
                filename=budget_filename, budget=export_budget
            )

            if verbose:
                print(f"💰 Exported weights with ${export_budget:,.0f} budget")

        # Export cycle summary
        summary_filename = f"{export_dir}/{strategy_name.lower().replace(' ', '_')}_weights_summary.csv"
        backtest_results.export_weights_summary(filename=summary_filename)

        if verbose:
            print("✅ Weight export complete!")

    return backtest_results


def quick_backtest(strategy_fn: Callable, **kwargs) -> float:
    """
    Quick backtest that returns just the mean excess SPD percentage.
    Useful for optimization or quick comparisons.

    Args:
        strategy_fn: Function that computes weights from price data
        **kwargs: Additional arguments passed to backtest()

    Returns:
        Mean excess SPD percentage vs uniform DCA

    Example:
        >>> excess_spd = quick_backtest(my_strategy)
        >>> print(f"Strategy beats uniform DCA by {excess_spd:.2f}%")
    """
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("validate", False)

    results = backtest(strategy_fn, **kwargs)
    return results.spd_table["excess_pct"].mean()


# Decorator version for even cleaner syntax
def strategy(
    *,
    name: Optional[str] = None,
    cycle_years: int = CYCLE_YEARS,
    auto_backtest: bool = False,
):
    """
    Decorator to mark a function as a strategy and optionally auto-backtest it.

    Args:
        name: Name for the strategy
        cycle_years: Investment cycle length in years
        auto_backtest: Whether to automatically run backtest when function is defined

    Example:
        >>> @strategy(name="My Amazing Strategy", auto_backtest=True)
        ... def my_strategy(df):
        ...     # Your strategy logic
        ...     return weights
        >>>
        >>> # Backtest results automatically printed
        >>> # You can also call: backtest(my_strategy)
    """

    def decorator(func):
        # Store metadata on the function
        func._strategy_name = name or func.__name__
        func._cycle_years = cycle_years

        # Add convenience method to the function
        def run_backtest(**kwargs):
            kwargs.setdefault("strategy_name", func._strategy_name)
            kwargs.setdefault("cycle_years", func._cycle_years)
            return backtest(func, **kwargs)

        func.backtest = run_backtest

        # Auto-backtest if requested
        if auto_backtest:
            try:
                results = run_backtest(verbose=True)
                results.summary()
            except Exception as e:
                print(f"Auto-backtest failed for {func._strategy_name}: {e}")

        return func

    return decorator
