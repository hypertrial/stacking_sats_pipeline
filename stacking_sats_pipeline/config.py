"""
Global configuration and constants for the Stacking Sats Challenge.
"""

import logging

# ---------- constants --------------------------------------------------------
BACKTEST_START: str = "2020-01-01"  # Updated for grid search period
BACKTEST_END: str = "2024-12-31"
CYCLE_YEARS: int = 1  # investment duration
PURCHASE_FREQ: str = "Daily"  # (unused for now)
MIN_WEIGHT: float = 1e-5  # minimum daily allocation

# Configure logging once
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
