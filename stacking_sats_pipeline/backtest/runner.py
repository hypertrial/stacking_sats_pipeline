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

    return BacktestResults(strategy_fn, df_backtest, results)


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
