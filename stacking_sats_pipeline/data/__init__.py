"""
Data loading and extraction utilities.
"""

from .coinmetrics_loader import CoinMetricsLoader
from .data_loader import (
    MultiSourceDataLoader,
    extract_btc_data_to_csv,
    load_and_merge_data,
    load_btc_data_from_web,
    load_data,
    validate_price_data,
)

__all__ = [
    # Main classes
    "MultiSourceDataLoader",
    "CoinMetricsLoader",
    # Main functions
    "load_data",
    "load_and_merge_data",
    "validate_price_data",
    # Backward compatibility
    "extract_btc_data_to_csv",
    "load_btc_data_from_web",
]
