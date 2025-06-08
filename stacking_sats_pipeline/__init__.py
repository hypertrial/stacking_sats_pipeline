"""
Stacking Sats Pipeline - Bitcoin DCA Strategy Backtesting Framework
==================================================================

This package provides tools for backtesting Bitcoin DCA strategies with a clean,
PyPI library-like experience.

Quick Start:
    >>> from stacking_sats_pipeline import backtest
    >>>
    >>> def my_strategy(df):
    ...     # Your strategy logic here
    ...     return weights
    >>>
    >>> results = backtest(my_strategy)
    >>> results.summary()
    >>> results.plot()

Or using the decorator approach:
    >>> from stacking_sats_pipeline import strategy
    >>>
    >>> @strategy(name="My Strategy", auto_backtest=True)
    ... def my_strategy(df):
    ...     # Your strategy logic here
    ...     return weights
"""

# New simplified interface (recommended)
# Legacy imports for backward compatibility
from .backtest import (
    BacktestResults,
    backtest,
    backtest_dynamic_dca,
    check_strategy_submission_ready,
    compute_cycle_spd,
    quick_backtest,
    strategy,
)
from .config import (
    BACKTEST_END,
    BACKTEST_START,
    CYCLE_YEARS,
    MIN_WEIGHT,
    PURCHASE_FREQ,
)
from .data import (
    extract_btc_data_to_csv,
    load_btc_data_from_web,
    load_data,
    validate_price_data,
)
from .plot import (
    plot_features,
    plot_final_weights,
    plot_spd_comparison,
    plot_weight_sums_by_cycle,
)
from .strategy import compute_weights, construct_features

__version__ = "0.0.1"

__all__ = [
    # New simplified interface (recommended)
    "backtest",
    "quick_backtest",
    "strategy",
    "BacktestResults",
    # Config
    "BACKTEST_START",
    "BACKTEST_END",
    "CYCLE_YEARS",
    "PURCHASE_FREQ",
    "MIN_WEIGHT",
    # Data loading
    "extract_btc_data_to_csv",
    "load_data",
    "load_btc_data_from_web",
    "validate_price_data",
    # Features
    "construct_features",
    # Strategy (legacy)
    "compute_weights",
    # Backtesting (legacy)
    "compute_cycle_spd",
    "backtest_dynamic_dca",
    "check_strategy_submission_ready",
    # Plotting
    "plot_features",
    "plot_final_weights",
    "plot_weight_sums_by_cycle",
    "plot_spd_comparison",
]
