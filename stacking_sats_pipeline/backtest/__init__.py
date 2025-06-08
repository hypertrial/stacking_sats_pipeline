"""
Backtesting and strategy validation utilities.
"""

# Import all implementations from checks module
from .checks import (
    backtest_dynamic_dca,
    check_strategy_submission_ready,
    compute_cycle_spd,
    validate_strategy_comprehensive,
)

# Import the new simplified interface
from .runner import (
    BacktestResults,
    backtest,
    quick_backtest,
    strategy,
)

__all__ = [
    # New simplified interface (recommended)
    "backtest",
    "quick_backtest",
    "strategy",
    "BacktestResults",
    # Legacy interface (for backward compatibility)
    "backtest_dynamic_dca",
    "check_strategy_submission_ready",
    "compute_cycle_spd",
    "validate_strategy_comprehensive",
]
