"""
Stacking Sats Challenge – Strategy & Utilities Package
=====================================================

This package provides utilities for Bitcoin DCA strategy backtesting.
"""

from .backtest import (
    backtest_dynamic_dca,
    check_strategy_submission_ready,
    compute_cycle_spd,
)
from .config import (
    BACKTEST_END,
    BACKTEST_START,
    CYCLE_YEARS,
    MIN_WEIGHT,
    PURCHASE_FREQ,
)
from .data import extract_btc_data_to_csv, load_data, validate_price_data
from .plot import (
    plot_features,
    plot_final_weights,
    plot_spd_comparison,
    plot_weight_sums_by_cycle,
)
from .strategy import compute_weights, construct_features

__all__ = [
    # Config
    "BACKTEST_START",
    "BACKTEST_END",
    "CYCLE_YEARS",
    "PURCHASE_FREQ",
    "MIN_WEIGHT",
    # Data loading
    "extract_btc_data_to_csv",
    "load_data",
    "validate_price_data",
    # Features
    "construct_features",
    # Strategy
    "compute_weights",
    # Backtesting
    "compute_cycle_spd",
    "backtest_dynamic_dca",
    "check_strategy_submission_ready",
    # Plotting
    "plot_features",
    "plot_final_weights",
    "plot_weight_sums_by_cycle",
    "plot_spd_comparison",
]
