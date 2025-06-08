# ---------------------------
# backtesting.py
# ---------------------------

# ╔═════════════╗
# ║  Imports    ║
# ╚═════════════╝

# Import all backtesting and validation functions from checks module
from .checks import (
    backtest_dynamic_dca,
    check_strategy_submission_ready,
    compute_cycle_spd,
    validate_strategy_comprehensive,
)

# Re-export all functions for backward compatibility
__all__ = [
    "backtest_dynamic_dca",
    "check_strategy_submission_ready",
    "compute_cycle_spd",
    "validate_strategy_comprehensive",
]
