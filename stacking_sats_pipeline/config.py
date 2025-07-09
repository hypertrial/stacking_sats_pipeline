"""
Global configuration for the Stacking Sats Pipeline.
"""

import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configuration constants for backwards compatibility with tests and tutorials
BACKTEST_START = "2016-01-01"
BACKTEST_END = "2024-01-01"
